"""Ultra master cut — one final ~30-min narration of the corpus.

Four passes layered on the per-episode digests already on disk (no new
transcription, extraction, or audio cutting):

0. KNOWN-STATE baseline (eps 1-40): the already-listened episodes are mined
   first (cached permanently as ``ultra_insights_001_040.json``). Their
   insights seed every subsequent mining prompt's previously-found list, so
   a candidate that merely restates something the listener already knows is
   marked ``covered_candidates`` instead of re-emitted as new. The narration
   also gets the baseline titles to connect to without re-explaining.
1. Chunked novelty mining (DeepSeek): each chunk (~10-12 digests) yields only
   the paradigm-shifting insights NEW relative to the known baseline — the
   "many gods vs one Yahweh" tier. Sequential chunks pass the previously-found
   insight list forward, so repetition collapses. Per-chunk cached, resumable.
2. Coalesce + deterministic dedup + rank (DeepSeek + code): one call fuses
   the per-chunk insights into the global list; ``_dedupe_insights`` folds
   true duplicates in code. **No arbitrary caps** — this dedup is the only
   deterministic guard, and it is the anti-repetition mechanism itself, not a
   volume cap. The volume ceiling is DeepSeek's own ~8k output-token limit.
3. One final narration (DeepSeek → edge-tts): the ranked outline becomes one
   flowing ~5,000-word spoken exposition (open → develop → close), rendered to
   ``output/ultra_cut.mp3`` at 1x — 1x IS the product. ~5,000 words ≈ ~28-30
   min at 1x (measured TTS rate ~175-180 wpm).

Follows the same fail-loud / resumable / cache conventions as the rest of the
pipeline (``super_cut.py`` caches, ``narration.py`` batch, ``summary_doc.py``
digests). Everything downstream of the mining phase is deterministic from the
persisted caches; ``--skip-tts`` lets you regenerate the narration text
without re-spending any DeepSeek calls.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from podcastcondensor.llm.deepseek import DeepSeekClient, resolve_api_key
from podcastcondensor.summary_doc import (
    _default_output_root,
    _episode_digest,
    _episode_dir_numbers,
    _episode_title,
)
from podcastcondensor.theme_extraction import _repair_truncated_json, _try_parse_json
from podcastcondensor.tts import synthesize_narration

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
)
_MINE_PROMPT_PATH = os.path.join(_PROMPT_DIR, "mine_insights.txt")
_COALESCE_PROMPT_PATH = os.path.join(_PROMPT_DIR, "coalesce_insights.txt")
_NARRATE_PROMPT_PATH = os.path.join(_PROMPT_DIR, "narrate_ultra_cut.txt")

_INSIGHTS_FILENAME_TMPL = "ultra_insights_{start:03d}_{end:03d}.json"
_OUTLINE_FILENAME_TMPL = "ultra_outline_{start:03d}_{end:03d}.json"
_NARRATION_FILENAME = "ultra_cut_narration.txt"
_MP3_FILENAME = "ultra_cut.mp3"

# DeepSeek chat-completions output ceiling; the mining/coalesce/narration
# responses are sized to stay under it (a ~5,000-word narration is ~6.5-7k
# tokens).
_MAX_TOKENS = 8192


def _load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Phase 1: chunked novelty mining
# ---------------------------------------------------------------------------


def build_chunks(ep_nums: List[int], chunk_size: int) -> List[List[int]]:
    """Split the episode list into fixed-size chunks (sequential mining)."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [ep_nums[i:i + chunk_size] for i in range(0, len(ep_nums), chunk_size)]


def _build_mine_prompt(
    digests: List[Tuple[int, str]],
    previously_found: List[dict],
) -> str:
    """Assemble the mining prompt: base + optional prior insights + digests."""
    base = _load_prompt(_MINE_PROMPT_PATH)
    parts = [base]
    if previously_found:
        parts.append("Previously found insights:")
        parts.append(json.dumps(
            [{"id": i.get("id"), "title": i.get("title"), "claim": i.get("claim")}
             for i in previously_found],
            indent=2,
        ))
    parts.append("Episode digests:")
    parts.append(json.dumps(
        [{"episode": ep, "digest": text} for ep, text in digests],
        indent=2,
    ))
    return "\n\n".join(parts)


def _parse_mine_response(raw: str) -> Tuple[List[dict], List[str]]:
    """Parse the mining JSON response into (insights, covered_candidates).

    Mirrors theme_extraction robustness: strips markdown fences, repairs
    trailing commas and truncated JSON. Malformed entries are skipped.
    """
    if not raw:
        return [], []
    data = _parse_json_object(raw)
    if not data:
        logger.warning(
            "Failed to parse mine JSON (first 200 chars): %s", raw[:200]
        )
        return [], []

    insights: List[dict] = []
    for item in data.get("insights", []):
        try:
            episodes_raw = item.get("episodes", [])
            if not isinstance(episodes_raw, list):
                raise ValueError("episodes must be a list")
            episodes = sorted({
                int(e) for e in episodes_raw
                if isinstance(e, int) or str(e).lstrip("-").isdigit()
            })
            claim = str(item.get("claim", "")).strip()
            if not claim:
                raise ValueError("missing claim")
            insights.append({
                "id": str(item.get("id", "unknown")).strip() or "unknown",
                "title": str(item.get("title", "Untitled")).strip(),
                "claim": str(item.get("claim", "")).strip(),
                "why_novel": str(item.get("why_novel", "")).strip(),
                "detail": str(item.get("detail", "")).strip(),
                "episodes": episodes,
            })
        except (TypeError, ValueError) as e:  # noqa: BLE001
            logger.warning("Skipping malformed mined insight: %s", e)
    covered = [
        str(c).strip() for c in data.get("covered_candidates", []) if str(c).strip()
    ]
    return insights, covered


def _parse_json_object(raw: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response and parse it.

    Handles markdown fences, trailing commas, and truncated responses
    (same robustness as super_cut / theme_extraction).
    """
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        clean = []
        in_block = False
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean).strip()
    start = text.find("{")
    if start < 0:
        return None
    candidate = text[start:]
    data = _try_parse_json(candidate)
    if data is None:
        repaired = _repair_truncated_json(candidate)
        if repaired and repaired != candidate:
            data = _try_parse_json(repaired)
            if data:
                logger.info("Repaired truncated JSON response")
    return data if isinstance(data, dict) else None


def _insights_cache_path(output_root: str, start: int, end: int) -> str:
    return os.path.join(
        output_root, _INSIGHTS_FILENAME_TMPL.format(start=start, end=end)
    )


def _load_insights_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "chunks" in data:
                return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt insights cache, starting fresh: %s", path)
    return {"chunks": {}}


def _save_insights_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def mine_chunk(
    output_root: str,
    chunk_eps: List[int],
    previously_found: List[dict],
    client: DeepSeekClient,
    model: str = "deepseek-chat",
    timeout: int = 600,
) -> Tuple[List[dict], List[str]]:
    """One chunk's novelty mining call; returns (insights, covered_candidates)."""
    digests = [
        (n, _episode_digest(output_root, n, _episode_title(output_root, n),
                            with_segments=False))
        for n in chunk_eps
    ]
    prompt = _build_mine_prompt(digests, previously_found)
    logger.info(
        "Mining chunk eps %s (%d chars prompt)",
        chunk_eps, len(prompt),
    )
    raw = client.generate(
        prompt=prompt,
        model=model,
        timeout=timeout,
        temperature=0.3,
        max_tokens=_MAX_TOKENS,
        force_json=True,
    )
    return _parse_mine_response(raw)


def mine_all_chunks(
    output_root: str,
    ep_nums: List[int],
    chunk_size: int,
    client: DeepSeekClient,
    model: str = "deepseek-chat",
    timeout: int = 600,
    cache_path: str = "",
    seed: Optional[List[dict]] = None,
) -> dict:
    """Mine every chunk sequentially, resuming from the cache.

    ``seed`` (default []) is the KNOWN-STATE baseline: insights already
    established in an earlier, already-listened range (e.g. eps 1-40). Every
    chunk's previously-found list starts from the seed, so a candidate that
    merely restates something the listener already knows is marked
    ``covered_candidates`` instead of being re-emitted as new. The list then
    accumulates across chunks (and across resumed runs — cached chunks feed
    it), so repetition collapses without ever re-spending a call on an
    already-mined chunk. Returns the insights cache dict:
    ``{"chunks": {"0": {"episodes": [...], "insights": [...]}, ...}}``.
    """
    path = cache_path or _insights_cache_path(output_root, ep_nums[0], ep_nums[-1])
    cache = _load_insights_cache(path)
    chunks = build_chunks(ep_nums, chunk_size)

    # Seed previously-found from the known-state baseline + cached chunks
    # (resume correctness).
    seen: List[dict] = list(seed or [])
    for ci, _ in enumerate(chunks):
        seen.extend(cache["chunks"].get(str(ci), {}).get("insights", []))

    for ci, chunk_eps in enumerate(chunks):
        if str(ci) in cache["chunks"]:
            logger.info("chunk %d (eps %s): cached, skipping", ci, chunk_eps)
            continue
        insights, covered = mine_chunk(
            output_root, chunk_eps, seen, client, model=model, timeout=timeout
        )
        logger.info(
            "chunk %d (eps %s): %d new insights, %d covered candidates",
            ci, chunk_eps, len(insights), len(covered),
        )
        cache["chunks"][str(ci)] = {"episodes": chunk_eps, "insights": insights}
        seen.extend(insights)
        _save_insights_cache(path, cache)
    return cache


def collect_insights(cache: dict) -> List[dict]:
    """Flatten all chunk insights into one list with namespaced ids (c00-...)."""
    insights: List[dict] = []
    for ci in sorted(cache.get("chunks", {}), key=int):
        for ins in cache["chunks"][ci].get("insights", []):
            namespaced = dict(ins)
            namespaced["id"] = f"c{int(ci):02d}-{ins.get('id')}"
            insights.append(namespaced)
    return insights


# ---------------------------------------------------------------------------
# Phase 2: coalesce + deterministic dedup + rank
# ---------------------------------------------------------------------------


def _build_coalesce_prompt(insights: List[dict]) -> str:
    base = _load_prompt(_COALESCE_PROMPT_PATH)
    return base + "\n\n" + json.dumps({"insights": insights}, indent=2)


def _parse_coalesce_response(raw: str) -> List[dict]:
    if not raw:
        return []
    data = _parse_json_object(raw)
    if not data:
        logger.warning(
            "Failed to parse coalesce JSON (first 200 chars): %s", raw[:200]
        )
        return []

    themes: List[dict] = []
    for item in data.get("themes", []):
        try:
            themes.append({
                "id": str(item.get("id", "unknown")).strip() or "unknown",
                "title": str(item.get("title", "Untitled")).strip(),
                "claim": str(item.get("claim", "")).strip(),
                "why_novel": str(item.get("why_novel", "")).strip(),
                "detail": str(item.get("detail", "")).strip(),
                "importance": float(item.get("importance", 0.5)),
                "episodes": sorted({
                    int(e) for e in item.get("episodes", [])
                    if isinstance(e, int) or str(e).lstrip("-").isdigit()
                }),
                "insight_ids": [
                    str(x) for x in item.get("insight_ids", []) if str(x).strip()
                ],
            })
        except (TypeError, ValueError) as e:  # noqa: BLE001
            logger.warning("Skipping malformed coalesced theme: %s", e)
    return themes


def _normalize_title(title: str) -> str:
    return "".join(c for c in title.lower() if c.isalnum())


def dedupe_insights(themes: List[dict]) -> List[dict]:
    """Deterministic dedup of coalesce themes — **no count cap**.

    Drops themes with no source chunk insights or empty claims, merges themes
    whose normalized titles are identical (keeping the most important claim,
    union of episodes and source insight ids), and re-ranks by importance.
    This is the ONLY deterministic guard — it is the anti-repetition
    mechanism, not a volume constraint. Duplicate themes that DeepSeek failed
    to fold are the exact repetition this pass exists to kill.
    """
    kept = [
        t for t in themes
        if t.get("insight_ids") and t.get("claim", "").strip()
    ]
    if not kept:
        return []

    by_title: Dict[str, dict] = {}
    for theme in sorted(kept, key=lambda t: -t.get("importance", 0.0)):
        key = _normalize_title(theme.get("title", ""))
        if not key:
            continue
        if key not in by_title:
            by_title[key] = dict(theme)
            continue
        merged = by_title[key]
        # Keep the higher-importance claim/why/detail; union the rest.
        if theme["importance"] > merged["importance"]:
            merged["claim"] = theme["claim"]
            merged["why_novel"] = theme["why_novel"]
            merged["detail"] = theme["detail"]
            merged["importance"] = theme["importance"]
        merged["episodes"] = sorted(set(merged["episodes"]) | set(theme["episodes"]))
        merged["insight_ids"] = sorted(set(merged["insight_ids"]) | set(theme["insight_ids"]))

    ranked = sorted(by_title.values(), key=lambda t: -t.get("importance", 0.0))
    logger.info(
        "Coalesce dedup: %d → %d global insights (no cap)",
        len(themes), len(ranked),
    )
    return ranked


def coalesce_insights(
    insights: List[dict],
    client: DeepSeekClient,
    model: str = "deepseek-chat",
    timeout: int = 600,
) -> List[dict]:
    """One DeepSeek call fusing all chunk insights into the global list."""
    if not insights:
        return []
    prompt = _build_coalesce_prompt(insights)
    logger.info(
        "Coalescing %d chunk insights (%d chars prompt)",
        len(insights), len(prompt),
    )
    raw = client.generate(
        prompt=prompt,
        model=model,
        timeout=timeout,
        temperature=0.3,
        max_tokens=_MAX_TOKENS,
        force_json=True,
    )
    themes = _parse_coalesce_response(raw)
    if not themes:
        logger.warning("Coalesce returned no themes")
        return []

    known = {i.get("id") for i in insights}
    for t in themes:
        t["insight_ids"] = [sid for sid in t["insight_ids"] if sid in known]
    return dedupe_insights(themes)


def _outline_cache_path(output_root: str, start: int, end: int) -> str:
    return os.path.join(
        output_root, _OUTLINE_FILENAME_TMPL.format(start=start, end=end)
    )


def save_outline(outline: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# Phase 3: one final narration + TTS
# ---------------------------------------------------------------------------


def narrate_ultra_cut(
    outline: List[dict],
    client: DeepSeekClient,
    out_path: str,
    known_titles: Optional[List[str]] = None,
    model: str = "deepseek-chat",
    timeout: int = 600,
    max_tokens: int = _MAX_TOKENS,
) -> str:
    """Write the final flowing narration from the outline; return the text.

    ``known_titles`` (optional) is the known-state baseline — the insights the
    listener already absorbed from eps 1-40. They are fed to the LLM so the
    narration can *connect* the new 41-144 material to them without
    re-explaining them (the listener already knows them).

    The ~5,000-word guide is in the prompt, NOT enforced here — DeepSeek's own
    output-token ceiling is the real bound (documented behavior: the model
    ignores volume constraints, so we let it fly). Retries once on failure,
    mirroring ``narration.py``.
    """
    prompt = _load_prompt(_NARRATE_PROMPT_PATH)
    if known_titles:
        prompt += (
            "\n\nKnown from earlier episodes (the listener has already absorbed"
            " these; do NOT re-explain them, but you may reference them briefly"
            " to connect):\n" + json.dumps(known_titles, indent=2)
        )
    prompt += "\n\nThe outline:\n" + json.dumps(outline, indent=2)
    logger.info("Narrating ultra cut from %d outline themes", len(outline))
    last_error = None
    narration = ""
    for _attempt in range(2):
        try:
            narration = client.generate(
                prompt=prompt,
                model=model,
                timeout=timeout,
                temperature=0.3,
                max_tokens=max_tokens,
            ).strip()
            if not narration:
                raise RuntimeError("empty ultra-cut narration response")
            break
        except Exception as e:  # noqa: BLE001 — mirror narration.py retry
            last_error = e
            logger.warning("Ultra-cut narration call failed: %s — retrying", e)
    else:
        raise RuntimeError(f"Ultra-cut narration failed: {last_error}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(narration + "\n")
    logger.info(
        "Wrote ultra-cut narration: %s (%d words)",
        out_path, len(narration.split()),
    )
    return narration


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _require_global_states(root: str, ep_nums: List[int], label: str) -> None:
    """Fail-loud: every episode in ``ep_nums`` must have ``global_state.json``."""
    missing = [
        n for n in ep_nums
        if not os.path.exists(os.path.join(root, f"ep-{n:03d}", "global_state.json"))
    ]
    if missing:
        raise RuntimeError(
            f"Ultra cut {label} incomplete: {len(missing)} episodes missing "
            f"global_state.json (never report a partial cut as success): "
            + ", ".join(f"ep {n}" for n in missing[:10])
        )


def build_ultra_cut(
    output_root: str = "",
    start: int = 41,  # eps 1-40 already listened — the ultra cut covers 41-144
    end: int = 144,
    chunk_size: int = 12,
    client: Optional[DeepSeekClient] = None,
    skip_tts: bool = False,
    speed: float = 1.0,
    model: str = "deepseek-chat",
    timeout: int = 600,
    prior_start: int = 1,
    prior_end: int = 40,
) -> dict:
    """Run the full ultra-cut pipeline; return a phases/timings summary.

    The KNOWN-STATE baseline (``prior_start``..``prior_end``, default 1-40 —
    the episodes the listener has already absorbed) is mined FIRST and cached
    permanently as ``ultra_insights_001_040.json``. Its insights seed the
    target-range mining's previously-found list, so only material genuinely
    NEW relative to what the listener already knows survives to the outline.
    The narration is also handed the baseline titles so it can connect the new
    material without re-explaining what's already known.

    Fail-loud: any non-Q&A episode in either range missing
    ``global_state.json`` aborts before any LLM call. Q&A episodes are skipped
    (the canonical ``_QA_EPISODES`` set via ``_episode_dir_numbers``).
    """
    root = output_root or _default_output_root()
    ep_nums = [n for n in _episode_dir_numbers(root) if start <= n <= end]
    if not ep_nums:
        raise RuntimeError(
            f"No non-Q&A episodes in range {start}-{end} under {root!r}"
        )
    _require_global_states(root, ep_nums, label=f"{start}-{end}")

    if client is None:
        client = DeepSeekClient(api_key=resolve_api_key())

    phases: List[dict] = []
    results: dict = {}

    # Phase 0: known-state baseline (already-listened eps 1-40, cached once).
    # This is the novelty prior — candidates merely restating it are NOT new.
    seed: List[dict] = []
    known_cache_path = ""
    prior_eps = [
        n for n in _episode_dir_numbers(root) if prior_start <= n <= prior_end
    ]
    if prior_eps:
        _require_global_states(root, prior_eps, label=f"known baseline {prior_start}-{prior_end}")
        t0 = time.time()
        known_cache_path = _insights_cache_path(root, prior_start, prior_end)
        prior_cache = mine_all_chunks(
            root, prior_eps, chunk_size, client, model=model, timeout=timeout,
            cache_path=known_cache_path,
        )
        seed = collect_insights(prior_cache)
        results["known_cache"] = known_cache_path
        phases.append({
            "phase": "known_state",
            "elapsed_sec": time.time() - t0,
            "prior_episodes": len(prior_eps),
            "known_insight_count": len(seed),
        })

    # Phase 1: chunked novelty mining, seeded with the known baseline
    t0 = time.time()
    cache_path = _insights_cache_path(root, start, end)
    cache = mine_all_chunks(
        root, ep_nums, chunk_size, client, model=model, timeout=timeout,
        cache_path=cache_path, seed=seed,
    )
    all_insights = collect_insights(cache)
    results["insights_cache"] = cache_path
    phases.append({
        "phase": "mine_insights",
        "elapsed_sec": time.time() - t0,
        "chunks": len(cache.get("chunks", {})),
        "insight_count": len(all_insights),
    })

    # Phase 2: coalesce + dedup + rank
    t0 = time.time()
    outline = coalesce_insights(all_insights, client, model=model, timeout=timeout)
    outline_path = save_outline(outline, _outline_cache_path(root, start, end))
    results["outline"] = outline_path
    phases.append({
        "phase": "coalesce",
        "elapsed_sec": time.time() - t0,
        "theme_count": len(outline),
    })

    # Phase 3: one final narration
    t0 = time.time()
    narration_path = os.path.join(root, _NARRATION_FILENAME)
    narration = narrate_ultra_cut(
        outline, client, narration_path,
        known_titles=[i.get("title") for i in seed if i.get("title")],
        model=model, timeout=timeout,
    )
    results["narration"] = narration_path
    phases.append({
        "phase": "narrate",
        "elapsed_sec": time.time() - t0,
        "words": len(narration.split()),
    })

    # Phase 4: TTS (1x IS the product; --speed adds an optional convenience copy)
    mp3_path = ""
    speed_path = ""
    if not skip_tts:
        t0 = time.time()
        mp3_path = synthesize_narration(
            narration, os.path.join(root, _MP3_FILENAME)
        )
        if speed and abs(speed - 1.0) > 1e-9:
            from podcastcondensor.narration import _apply_speed
            speed_path = _apply_speed(
                mp3_path, os.path.join(root, f"ultra_cut_{speed}x.mp3"), speed
            )
        phases.append({
            "phase": "tts",
            "elapsed_sec": time.time() - t0,
            "mp3_path": mp3_path,
            "speed_path": speed_path,
        })

    results["phases"] = phases
    results["mp3"] = mp3_path
    results["mp3_speed"] = speed_path
    return results
