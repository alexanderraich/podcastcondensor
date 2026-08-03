"""Super master cut — full-corpus thematic anthology (eps 1-144).

The ``build-super-cut`` pipeline is the offline counterpart to
``build-master-cut``: instead of downloading/transcribing, it reads the
per-episode ``global_state.json`` files already on disk and produces a
per-theme MP3 for every TOP theme of the whole series.

Phases:
  1. Merge cumulative universe from per-episode global_state.json (0 API calls)
  2. Chunked theme extraction — existing ``extract_themes`` per ~12-ep chunk
  3. Coalesce — one DeepSeek call fuses all chunk themes into the top global
     themes of the series (the "top topics over all the universe")
  4. Resolve + episode-diversity cap — each global theme's candidates span
     many episodes, capped so the selection prompt stays in context
  5. Per-theme LLM selection (reused ``select_segments_for_master_cut``,
     NO budget → LLM-owned volume, same mechanism as the range cuts)
  6. Assemble one MP3 per theme (reused ``assemble_master_cut``)

Fully offline from disk; never contacts YouTube.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from podcastcondensor.config import Config
from podcastcondensor.download_pool import _scan_existing_episodes
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.master_cut import (
    ThemeWithSegments,
    ThemeSegment,
    Selection,
    MasterCutPlan,
    resolve_theme_segments_from_state,
    select_segments_for_master_cut,
    assemble_master_cut,
    _compute_selection_warnings,
)
from podcastcondensor.theme_extraction import (
    Theme,
    extract_themes,
    _try_parse_json,
    _repair_truncated_json,
)
from podcastcondensor.universe_state import _fresh_default, merge_episode_knowledge

logger = logging.getLogger(__name__)

CATEGORIES = ("entities", "concepts", "claims", "scriptural_links", "glossary")

COALESCE_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts", "coalesce_themes.txt",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChunkThemeRecord:
    """A theme discovered inside one chunk, namespaced and collision-proof."""
    chunk_index: int
    theme_id: str            # namespaced: f"c{chunk_index:02d}-{theme.id}"
    title: str
    description: str
    importance: float
    related_item_ids: List[str]
    episode_span: Tuple[int, int]  # [min_ep, max_ep] of the theme's items


@dataclass
class GlobalTheme:
    """A top topic of the whole series, aggregated from chunk themes."""
    id: str
    title: str
    description: str
    importance: float
    source_theme_ids: List[str] = field(default_factory=list)
    natural_intro_theme_ids: List[str] = field(default_factory=list)
    # Filled during resolve phase:
    related_item_ids: List[str] = field(default_factory=list)
    candidate_count: int = 0
    episode_count: int = 0
    episode_numbers: List[int] = field(default_factory=list)  # sorted distinct eps of candidates
    selected_count: int = 0


# ---------------------------------------------------------------------------
# Phase 1: Cumulative universe merge (offline)
# ---------------------------------------------------------------------------


def build_universe_from_global_states(
    output_root: str,
    start_ep: int,
    end_ep: int,
    state_path: str = "",
) -> dict:
    """Merge every per-episode ``global_state.json`` in [start_ep, end_ep] into
    one cumulative universe dict. Reuses ``merge_episode_knowledge`` (same
    id-dedup / episode_numbers / segment-union semantics as the pipeline).

    Episodes without a global_state.json (e.g. Q&A, or eps 1-30 before the
    regeneration step) are skipped silently. Returns the merged dict.
    """
    data = _fresh_default()
    merged: List[int] = []
    for ep_num in range(start_ep, end_ep + 1):
        gs_path = os.path.join(output_root, f"ep-{ep_num:03d}", "global_state.json")
        if not os.path.exists(gs_path):
            continue
        with open(gs_path, encoding="utf-8") as f:
            global_data = json.load(f)
        knowledge = {
            "summary": global_data.get("summary", ""),
            "entities": global_data.get("entities", []),
            "concepts": global_data.get("concepts", []),
            "claims": global_data.get("claims", []),
            "scriptural_links": global_data.get("scriptural_links", []),
            "glossary": global_data.get("glossary", []),
        }
        merge_episode_knowledge(data, ep_num, knowledge)
        merged.append(ep_num)

    data["metadata"]["episodes_built_from"] = merged
    data["metadata"]["last_built_episode"] = max(merged) if merged else 0

    if state_path:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Wrote cumulative universe: %s (%d episodes)", state_path, len(merged))
    return data


# ---------------------------------------------------------------------------
# Phase 2: Chunked theme extraction
# ---------------------------------------------------------------------------


def build_chunks(
    output_root: str,
    start_ep: int,
    end_ep: int,
    chunk_size: int = 12,
) -> List[List[int]]:
    """Bin episodes that have a ``global_state.json`` into chunks of ``chunk_size``.

    Episodes without global_state (Q&A, unregenerated eps 1-30) are excluded
    automatically. The last chunk may be smaller.
    """
    eps = []
    for ep_num in range(start_ep, end_ep + 1):
        gs_path = os.path.join(output_root, f"ep-{ep_num:03d}", "global_state.json")
        if os.path.exists(gs_path):
            eps.append(ep_num)
    return [eps[i:i + chunk_size] for i in range(0, len(eps), chunk_size)]


def _build_chunk_universe(ep_list: List[int], output_root: str) -> dict:
    """Merge a chunk's global_states in memory (chunk-local episode_numbers)."""
    data = _fresh_default()
    merged: List[int] = []
    for ep_num in ep_list:
        gs_path = os.path.join(output_root, f"ep-{ep_num:03d}", "global_state.json")
        if not os.path.exists(gs_path):
            continue
        with open(gs_path, encoding="utf-8") as f:
            global_data = json.load(f)
        knowledge = {
            "summary": global_data.get("summary", ""),
            "entities": global_data.get("entities", []),
            "concepts": global_data.get("concepts", []),
            "claims": global_data.get("claims", []),
            "scriptural_links": global_data.get("scriptural_links", []),
            "glossary": global_data.get("glossary", []),
        }
        merge_episode_knowledge(data, ep_num, knowledge)
        merged.append(ep_num)
    data["metadata"]["episodes_built_from"] = merged
    data["metadata"]["last_built_episode"] = max(merged) if merged else 0
    return data


def _find_item(universe_data: dict, item_id: str) -> Optional[dict]:
    """Return an item by id from any category in a universe dict (or None)."""
    for category in CATEGORIES:
        for item in universe_data.get(category, []):
            if item.get("id") == item_id:
                return item
    return None


def extract_chunk_themes(
    ep_list: List[int],
    output_root: str,
    client,
    cfg: Config,
    chunk_index: int,
) -> List[ChunkThemeRecord]:
    """Run the existing ``extract_themes`` on one chunk, namespacing theme ids.

    Returns ChunkThemeRecords with collision-proof ``c{chunk_index:02d}-``
    prefixed ids and the episode span of each theme's items.
    """
    if not ep_list:
        return []
    chunk_universe = _build_chunk_universe(ep_list, output_root)
    themes = extract_themes(
        universe_data=chunk_universe,
        client=client,
        model=cfg.deepseek_model,
        timeout=cfg.deepseek_timeout,
    )

    records: List[ChunkThemeRecord] = []
    for t in themes:
        spans: List[int] = []
        for iid in t.related_item_ids:
            item = _find_item(chunk_universe, iid)
            if item:
                spans.extend(item.get("episode_numbers", []))
        ep_span = (min(spans), max(spans)) if spans else (min(ep_list), max(ep_list))
        records.append(ChunkThemeRecord(
            chunk_index=chunk_index,
            theme_id=f"c{chunk_index:02d}-{t.id}",
            title=t.title,
            description=t.description,
            importance=t.importance,
            related_item_ids=list(t.related_item_ids),
            episode_span=ep_span,
        ))
    return records


# ---------------------------------------------------------------------------
# Phase 3: Coalesce chunk themes into global themes
# ---------------------------------------------------------------------------


def _build_coalesce_prompt(
    records: List[ChunkThemeRecord],
    chunk_episodes: Dict[int, List[int]],
) -> str:
    """Assemble the coalesce prompt: base prompt + compact chunk-theme JSON."""
    with open(COALESCE_PROMPT_PATH, encoding="utf-8") as f:
        base_prompt = f.read().strip()
    payload: dict = {"chunks": []}
    for ci in sorted(chunk_episodes.keys()):
        recs = [r for r in records if r.chunk_index == ci]
        payload["chunks"].append({
            "chunk_index": ci,
            "episodes": chunk_episodes[ci],
            "themes": [
                {
                    "id": r.theme_id,
                    "title": r.title,
                    "importance": r.importance,
                    "description": r.description,
                    "item_count": len(r.related_item_ids),
                    "episode_span": list(r.episode_span),
                }
                for r in recs
            ],
        })
    return base_prompt + "\n\n" + json.dumps(payload, indent=2)


def _parse_coalesce_response(raw: str) -> List[GlobalTheme]:
    """Parse the coalesce JSON response into GlobalTheme objects.

    Mirrors theme_extraction's robustness: strips markdown fences, repairs
    trailing commas and truncated JSON.
    """
    if not raw:
        return []
    text = raw.strip()

    # Strip markdown code fences
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
        logger.warning("No JSON object found in coalesce response")
        return []
    candidate = text[start:]

    data = _try_parse_json(candidate)
    if data is None:
        repaired = _repair_truncated_json(candidate)
        if repaired and repaired != candidate:
            data = _try_parse_json(repaired)
            if data:
                logger.info("Repaired truncated coalesce JSON")
    if data is None or not isinstance(data, dict):
        logger.warning(
            "Failed to parse coalesce JSON (first 200 chars): %s",
            candidate[:200],
        )
        return []

    themes: List[GlobalTheme] = []
    for rt in data.get("themes", []):
        try:
            themes.append(GlobalTheme(
                id=str(rt.get("id", "unknown")),
                title=str(rt.get("title", "Untitled")),
                description=str(rt.get("description", "")),
                importance=float(rt.get("importance", 0.5)),
                source_theme_ids=[str(x) for x in rt.get("source_theme_ids", [])],
                natural_intro_theme_ids=[
                    str(x) for x in rt.get("natural_intro_theme_ids", [])
                ],
            ))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Skipping malformed global theme: %s", e)
    return themes


def _dedupe_and_cap_global_themes(
    global_themes: List[GlobalTheme],
    max_themes: int = 25,
) -> List[GlobalTheme]:
    """Deterministically deduplicate and cap the coalesce output.

    DeepSeek does NOT reliably honor the "15-25 themes" count in the coalesce
    prompt (it produced 101 from 198 chunk themes in the first full dry run,
    with many near-duplicates). This is the reliable layer:

      1. Drop themes with no valid source chunk themes.
      2. Merge themes sharing the EXACT same source chunk-theme set (a chunk
         theme should map to at most one global theme; duplicates keep the
         highest importance and union sources/intros).
      3. Merge themes whose normalized titles are identical.
      4. Cap to ``max_themes`` by importance descending.
    """
    kept = [gt for gt in global_themes if gt.source_theme_ids]
    if not kept:
        return []

    def _merge(gt: GlobalTheme, other: GlobalTheme) -> GlobalTheme:
        gt.source_theme_ids = sorted(set(gt.source_theme_ids) | set(other.source_theme_ids))
        gt.natural_intro_theme_ids = list(
            dict.fromkeys(gt.natural_intro_theme_ids + other.natural_intro_theme_ids)
        )
        if other.importance > gt.importance:
            gt.importance = other.importance
            gt.description = other.description
        return gt

    # 2. Dedup by exact source set (keep highest importance)
    by_source: Dict[tuple, GlobalTheme] = {}
    for gt in sorted(kept, key=lambda g: -g.importance):
        key = tuple(sorted(set(gt.source_theme_ids)))
        if key in by_source:
            _merge(by_source[key], gt)
        else:
            by_source[key] = gt

    # 3. Dedup by normalized title
    by_title: Dict[str, GlobalTheme] = {}
    for gt in sorted(by_source.values(), key=lambda g: -g.importance):
        tkey = re.sub(r"[^a-z0-9]+", "", gt.title.lower())
        if tkey in by_title:
            _merge(by_title[tkey], gt)
        else:
            by_title[tkey] = gt

    # 4. Cap by importance
    ranked = sorted(by_title.values(), key=lambda g: -g.importance)
    logger.info(
        "Coalesce dedup: %d → %d global themes (capped at %d)",
        len(global_themes), len(ranked[:max_themes]), max_themes,
    )
    return ranked[:max_themes]


def coalesce_themes(
    records: List[ChunkThemeRecord],
    chunk_episodes: Dict[int, List[int]],
    client,
    cfg: Config,
    max_themes: int = 25,
) -> List[GlobalTheme]:
    """One DeepSeek call fusing all chunk themes into the top global themes.

    Drops source_theme_ids that don't reference a known namespaced chunk theme,
    then deterministically dedupes and caps via ``_dedupe_and_cap_global_themes``
    (DeepSeek over-produces without it).
    """
    if not records:
        return []
    prompt = _build_coalesce_prompt(records, chunk_episodes)
    logger.info(
        "Coalescing %d chunk themes into global themes (%d chars prompt)",
        len(records), len(prompt),
    )
    try:
        raw = client.generate(
            prompt=prompt,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout,
            temperature=0.3,
            max_tokens=16000,
            force_json=True,
        )
    except Exception as e:
        logger.error("Coalesce LLM call failed: %s", e)
        return []

    global_themes = _parse_coalesce_response(raw)
    if not global_themes:
        logger.warning("Coalesce returned no global themes")
        return []

    known = {r.theme_id for r in records}
    for gt in global_themes:
        gt.source_theme_ids = [sid for sid in gt.source_theme_ids if sid in known]
    logger.info("Coalesced into %d raw global themes", len(global_themes))
    return _dedupe_and_cap_global_themes(global_themes, max_themes=max_themes)


# ---------------------------------------------------------------------------
# Discovery cache (chunk themes + coalesced global themes)
# ---------------------------------------------------------------------------


def _discovery_cache_path(output_root: str, start_ep: int, end_ep: int) -> str:
    return os.path.join(
        output_root, f"super_cut_discovery_{start_ep:03d}_{end_ep:03d}.json"
    )


def _save_discovery_cache(
    path: str,
    chunk_episodes: Dict[int, List[int]],
    records: List[ChunkThemeRecord],
    global_themes: List[GlobalTheme],
) -> None:
    """Persist discovery so subsequent --theme cuts skip chunk+coalesce calls."""
    payload = {
        "chunk_episodes": {str(k): v for k, v in chunk_episodes.items()},
        "chunk_themes": [
            {
                "chunk_index": r.chunk_index,
                "theme_id": r.theme_id,
                "title": r.title,
                "description": r.description,
                "importance": r.importance,
                "related_item_ids": r.related_item_ids,
                "episode_span": list(r.episode_span),
            }
            for r in records
        ],
        "global_themes": [
            {
                "id": gt.id,
                "title": gt.title,
                "description": gt.description,
                "importance": gt.importance,
                "source_theme_ids": gt.source_theme_ids,
                "natural_intro_theme_ids": gt.natural_intro_theme_ids,
            }
            for gt in global_themes
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote discovery cache: %s", path)


def _load_discovery_cache(
    path: str,
) -> Optional[Tuple[Dict[int, List[int]], List[ChunkThemeRecord], List[GlobalTheme]]]:
    """Load a discovery cache, or None if absent/unusable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        records = [
            ChunkThemeRecord(
                chunk_index=int(r["chunk_index"]),
                theme_id=str(r["theme_id"]),
                title=str(r["title"]),
                description=str(r.get("description", "")),
                importance=float(r.get("importance", 0.5)),
                related_item_ids=[str(x) for x in r.get("related_item_ids", [])],
                episode_span=tuple(int(x) for x in r.get("episode_span", [])),
            )
            for r in payload.get("chunk_themes", [])
        ]
        global_themes = [
            GlobalTheme(
                id=str(gt["id"]),
                title=str(gt["title"]),
                description=str(gt.get("description", "")),
                importance=float(gt.get("importance", 0.5)),
                source_theme_ids=[str(x) for x in gt.get("source_theme_ids", [])],
                natural_intro_theme_ids=[
                    str(x) for x in gt.get("natural_intro_theme_ids", [])
                ],
            )
            for gt in payload.get("global_themes", [])
        ]
        chunk_episodes = {int(k): v for k, v in payload.get("chunk_episodes", {}).items()}
        if not records or not global_themes:
            return None
        return chunk_episodes, records, global_themes
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Discovery cache unusable (%s) — will re-run discovery", e)
        return None


# ---------------------------------------------------------------------------
# Candidates cache (per-theme capped candidate segments)
# ---------------------------------------------------------------------------


def _retry_empty_chunks(
    chunk_episodes: Dict[int, List[int]],
    all_records: List[ChunkThemeRecord],
    output_root: str,
    client,
    cfg: Config,
) -> int:
    """Re-attempt extraction ONLY for chunks that produced 0 themes.

    `extract_themes` returns [] when its DeepSeek JSON fails to parse, so a
    chunk can silently contribute nothing (eps missing from coalesce). This
    retries just those chunks — a targeted recovery, NOT a full re-run.

    Mutates ``all_records`` in place (appends recovered chunk themes).
    Returns the number of chunks retried.
    """
    counts: Dict[int, int] = defaultdict(int)
    for r in all_records:
        counts[r.chunk_index] += 1
    empty = sorted(ci for ci in chunk_episodes if counts[ci] == 0)
    for ci in empty:
        ep_list = chunk_episodes[ci]
        logger.info(
            "Retrying empty chunk %d (eps %d-%d)...",
            ci, ep_list[0], ep_list[-1],
        )
        records = extract_chunk_themes(ep_list, output_root, client, cfg, ci)
        if records:
            all_records.extend(records)
            logger.info("  → chunk %d retry: %d themes", ci, len(records))
        else:
            logger.warning("  → chunk %d retry still empty (will be skipped)", ci)
    return len(empty)


def _candidates_cache_path(output_root: str, start_ep: int, end_ep: int) -> str:
    return os.path.join(
        output_root, f"super_cut_candidates_{start_ep:03d}_{end_ep:03d}.json"
    )


def _save_candidates_cache(
    path: str,
    global_themes: List[GlobalTheme],
    themes_with_segments: List[ThemeWithSegments],
) -> None:
    """Persist capped candidate segments per theme (deterministic, 0 calls)."""
    by_theme = {}
    for gt, tws in zip(global_themes, themes_with_segments):
        by_theme[gt.id] = {
            "title": gt.title,
            "importance": gt.importance,
            "episode_numbers": gt.episode_numbers,
            "candidate_count": len(tws.segments),
            "candidates": [
                {
                    "episode": s.episode_number,
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "duration": round(s.duration, 2),
                    "audio_path": s.audio_path,
                    "relevance": s.relevance_score,
                    "is_intro": s.is_intro,
                }
                for s in tws.segments
            ],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(by_theme, f, ensure_ascii=False, indent=2)
    logger.info("Wrote candidates cache: %s (%d themes)", path, len(by_theme))


def _load_candidates_cache(path: str) -> dict:
    """Load candidates cache, or {} if absent/unusable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Candidates cache unusable (%s)", e)
        return {}


# ---------------------------------------------------------------------------
# Selections cache (per-theme LLM-selected segments)
# ---------------------------------------------------------------------------


def _selections_cache_path(output_root: str, start_ep: int, end_ep: int) -> str:
    return os.path.join(
        output_root, f"super_cut_selections_{start_ep:03d}_{end_ep:03d}.json"
    )


def _load_selections_cache(path: str) -> Dict[str, List[dict]]:
    """Load {theme_id: [selection dicts]}, or {} if absent/unusable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Selections cache unusable (%s)", e)
        return {}


def _merge_selections_cache(
    path: str,
    by_theme: Dict[str, List[Selection]],
) -> None:
    """Load existing cache, merge new selections in, write back."""
    existing = _load_selections_cache(path)
    for tid, sels in by_theme.items():
        existing[tid] = [
            {
                "theme_id": s.theme_id,
                "theme_title": s.theme_title,
                "episode": s.segment.episode_number,
                "audio_path": s.segment.audio_path,
                "start": round(s.segment.start, 2),
                "end": round(s.segment.end, 2),
                "text_preview": s.segment.text_preview,
                "beep_before": s.beep_before,
            }
            for s in sels
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    logger.info("Wrote selections cache: %s (%d themes)", path, len(existing))


def _selections_from_dicts(ds: List[dict]) -> List[Selection]:
    """Reconstruct Selection objects from cached dicts for assembly."""
    sels = []
    for d in ds:
        seg = ThemeSegment(
            theme_id=d["theme_id"],
            episode_number=d["episode"],
            audio_path=d["audio_path"],
            start=d["start"],
            end=d["end"],
            text_preview=d.get("text_preview", ""),
        )
        sels.append(Selection(
            segment=seg,
            theme_title=d.get("theme_title", d["theme_id"]),
            theme_id=d["theme_id"],
            beep_before=d.get("beep_before", "none"),
        ))
    return sels


def _resnap_selections(
    selections: List[Selection],
    output_root: str,
) -> List[Selection]:
    """Re-snap Selection boundaries to sentence blocks (0 LLM calls).

    Defensive against a stale selections cache: boundaries written before or
    without the sentence-block snap would otherwise ship mid-sentence cuts on
    the cache-HIT path. Selections whose SRT can't be loaded are kept as-is.
    """
    from podcastcondensor.minimal_theme_cut import (
        snap_to_sentence_blocks,
        RefinedSelection,
    )

    refs = [
        RefinedSelection(
            episode_number=s.segment.episode_number,
            audio_path=s.segment.audio_path,
            start=s.segment.start,
            end=s.segment.end,
            reason=s.segment.text_preview,
        )
        for s in selections
    ]
    snapped = snap_to_sentence_blocks(refs, output_root)

    out = []
    for orig, rs in zip(selections, snapped):
        seg = ThemeSegment(
            theme_id=orig.segment.theme_id,
            episode_number=rs.episode_number,
            audio_path=rs.audio_path,
            start=rs.start,
            end=rs.end,
            text_preview=rs.reason or orig.segment.text_preview,
            is_intro=orig.segment.is_intro,
            relevance_score=orig.segment.relevance_score,
            match_count=orig.segment.match_count,
        )
        out.append(Selection(
            segment=seg,
            theme_title=orig.theme_title,
            theme_id=orig.theme_id,
            beep_before=orig.beep_before,
        ))
    return out


# ---------------------------------------------------------------------------
# Bracket analysis (read-only, 0 LLM calls)
# ---------------------------------------------------------------------------


def analyze_brackets(
    candidates_by_theme: dict,
    themes_info: List[dict],
    brackets: List[Tuple[str, set]],
    top_n: int = 5,
) -> List[dict]:
    """Top themes per episode bracket, computed from the candidates cache.

    ``themes_info`` is [{id, title, importance}] (from discovery cache).
    ``brackets`` is [(name, set_of_episodes)]. "rest" = complement of the
    specified brackets. Ranked by in-bracket candidate count, tie-broken by
    in-bracket duration.
    """
    results = []
    for name, eps_set in brackets:
        scores = []
        for info in themes_info:
            cands = candidates_by_theme.get(info["id"], {}).get("candidates", [])
            in_bracket = [c for c in cands if c.get("episode") in eps_set]
            if not in_bracket:
                continue
            scores.append({
                "theme_id": info["id"],
                "title": info["title"],
                "importance": info["importance"],
                "candidates": len(in_bracket),
                "duration_sec": round(sum(c.get("duration", 0) for c in in_bracket), 1),
                "episodes": sorted({c["episode"] for c in in_bracket}),
            })
        scores.sort(key=lambda s: (-s["candidates"], -s["duration_sec"]))
        results.append({"bracket": name, "top": scores[:top_n]})
    return results


def _parse_episode_range(spec: str) -> set:
    """Parse 'a-b' or 'a' into a set of episode numbers."""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return set(range(int(a), int(b) + 1))
    return {int(spec)}


def run_super_cut_brackets(
    output_root: str,
    start_episode: int = 1,
    end_episode: int = 144,
    bracket_specs: Optional[List[str]] = None,
    top_n: int = 5,
) -> dict:
    """Read-only bracket analysis from persisted artefacts (0 LLM calls).

    ``bracket_specs`` are episode-range strings like "40-80" / "81-120"; the
    "rest" bracket = all episodes not covered by any specified bracket.
    Defaults to 40-80 / 81-120 / rest. Requires the discovery + candidates
    caches on disk (built by one ``build-super-cut`` run or ``--dry-run``).
    """
    if not output_root:
        output_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output",
        )
    discovery_path = _discovery_cache_path(output_root, start_episode, end_episode)
    candidates_path = _candidates_cache_path(output_root, start_episode, end_episode)
    cache = _load_discovery_cache(discovery_path)
    if cache is None:
        raise RuntimeError(
            f"Discovery cache missing: {discovery_path} — run build-super-cut "
            "once (or --dry-run) to build it."
        )
    candidates = _load_candidates_cache(candidates_path)
    if not candidates:
        raise RuntimeError(
            f"Candidates cache missing: {candidates_path} — run build-super-cut "
            "once (or --dry-run) to build it."
        )
    chunk_episodes, records, global_themes = cache
    themes_info = [
        {"id": gt.id, "title": gt.title, "importance": gt.importance}
        for gt in global_themes
    ]
    all_eps = set(range(start_episode, end_episode + 1))
    if not bracket_specs:
        bracket_specs = ["40-80", "81-120"]
    brackets: List[Tuple[str, set]] = []
    covered: set = set()
    for spec in bracket_specs:
        eps = _parse_episode_range(spec) & all_eps
        covered |= eps
        brackets.append((spec, eps))
    rest = all_eps - covered
    if rest:
        brackets.append(("rest", rest))
    return {"results": analyze_brackets(candidates, themes_info, brackets, top_n=top_n)}


# ---------------------------------------------------------------------------
# Phase 4: Resolve + episode-diversity cap
# ---------------------------------------------------------------------------


def cap_candidate_segments(
    segments: List[ThemeSegment],
    *,
    max_total: int = 48,
    max_per_episode: int = 3,
    reserve_top: int = 12,
) -> List[ThemeSegment]:
    """Keep a diversity-spread subset of candidate segments for one theme.

    Goal: the selection LLM should see segments from MANY episodes (the
    "throughout the run" requirement) without blowing the prompt context, but
    a theme's episode-core content must never be starved.

    Algorithm:
      1. Rank by (-relevance, -duration, ep, start).
      2. RESERVE the top ``reserve_top`` globally (episode-agnostic) — this
         protects a theme's anchor episode.
      3. From the rest, round-robin across episodes to fill up to ``max_total``,
         per-episode capped at ``max_per_episode`` — forces episode spread.
      4. Guarantee the theme's ``is_intro`` segment survives (swap it in for
         the lowest-relevance pick if the pool is full).
      5. Return a FILTER of the input in original order (build_selection_prompt
         assigns seg_id positionally, so order must be preserved).

    All parameters are tunable (a dry run can tune them).
    """
    if not segments:
        return []

    n = len(segments)
    # 1. Rank
    ranked = sorted(
        enumerate(segments),
        key=lambda ie: (
            -ie[1].relevance_score,
            -ie[1].duration,
            ie[1].episode_number,
            ie[1].start,
        ),
    )

    # 2. Reserve top-N by relevance (episode-agnostic)
    picked = {i for i, _ in ranked[:reserve_top]}
    ep_used: Dict[int, int] = defaultdict(int)
    for i in picked:
        ep_used[segments[i].episode_number] += 1

    # 3. Round-robin fill from the remaining pool
    remaining = [i for i, _ in ranked[reserve_top:]]
    by_episode: Dict[int, List[int]] = defaultdict(list)
    for i in remaining:
        by_episode[segments[i].episode_number].append(i)

    round_idx = 0
    while len(picked) < max_total:
        added_any = False
        for ep in sorted(by_episode.keys()):
            if len(picked) >= max_total:
                break
            if ep_used[ep] >= max_per_episode:
                continue
            idxs = by_episode[ep]
            if round_idx < len(idxs) and idxs[round_idx] not in picked:
                picked.add(idxs[round_idx])
                ep_used[ep] += 1
                added_any = True
        if not added_any:
            break
        round_idx += 1

    # 4. is_intro must survive
    intro = next((i for i, s in enumerate(segments) if s.is_intro), None)
    if intro is not None and intro not in picked:
        if len(picked) < max_total:
            picked.add(intro)
        else:
            lowest = min(
                picked,
                key=lambda i: (segments[i].relevance_score, -segments[i].duration),
            )
            picked.discard(lowest)
            picked.add(intro)

    # 5. Filter preserving input order
    return [s for i, s in enumerate(segments) if i in picked]


def resolve_global_theme_candidates(
    global_themes: List[GlobalTheme],
    record_by_id: Dict[str, ChunkThemeRecord],
    universe_data: dict,
    manifests,
    output_root: str,
) -> List[ThemeWithSegments]:
    """Resolve each global theme's candidate segments from the cumulative universe.

    Builds a ``Theme`` per global theme whose ``related_item_ids`` is the union
    of its constituent chunk themes' items, then reuses the master-cut
    ``resolve_theme_segments_from_state`` (dedups by ep:start:end, marks
    is_intro, scores relevance). Finally drops missing-audio candidates and
    applies the episode-diversity cap.
    """
    themes: List[Theme] = []
    for gt in global_themes:
        item_ids: List[str] = []
        for src_id in gt.source_theme_ids:
            rec = record_by_id.get(src_id)
            if not rec:
                continue
            for iid in rec.related_item_ids:
                if iid not in item_ids:
                    item_ids.append(iid)
        gt.related_item_ids = item_ids
        themes.append(Theme(
            id=gt.id,
            title=gt.title,
            description=gt.description,
            importance=gt.importance,
            related_item_ids=item_ids,
            natural_intro_items=[],
        ))

    results = resolve_theme_segments_from_state(
        themes=themes,
        universe_data=universe_data,
        manifests=manifests,
        output_root=output_root,
    )

    for gt, tws in zip(global_themes, results):
        tws.segments = [
            s for s in tws.segments
            if s.audio_path and os.path.exists(s.audio_path)
        ]
        tws.segments = cap_candidate_segments(tws.segments)
        gt.candidate_count = len(tws.segments)
        gt.episode_numbers = sorted({s.episode_number for s in tws.segments})
        gt.episode_count = len(gt.episode_numbers)
        logger.info(
            "  %s: %d candidates across %d episodes (%s)",
            gt.id, gt.candidate_count, gt.episode_count,
            _episode_span_str(gt.episode_numbers),
        )
    return results


# ---------------------------------------------------------------------------
# Phase 6: Assemble per-theme MP3s
# ---------------------------------------------------------------------------


def fill_selection_volume(
    selections: List[Selection],
    themes_with_segments: List[ThemeWithSegments],
    *,
    min_duration_floor: float = 0.0,
    max_per_episode: int = 2,
) -> List[Selection]:
    """Deterministically top up each theme's selection to a duration floor.

    The LLM's kept set is the quality ranking and narrative spine, but DeepSeek
    under-selects volume unreliably (measured 18 → 5 → 9 segments across
    identical setups). When a theme's total is below the floor, append the
    theme's best UN-KEPT candidate segments — per-episode capped, preferring
    LATER episodes so a late-arc theme surveys the run's second half — until
    the floor is met. Universal (no per-theme logic).

    Topped-up segments use their resolved candidate boundaries and pass through
    the same sentence-block snap as the LLM's picks downstream, so the
    no-mid-sentence-cuts guarantee holds. ``min_duration_floor=0`` disables
    filling (pure LLM volume, the archive default).
    """
    if min_duration_floor <= 0 or not selections:
        return selections

    by_theme: Dict[str, List[Selection]] = defaultdict(list)
    for s in selections:
        by_theme[s.theme_id].append(s)
    pool_by_theme: Dict[str, ThemeWithSegments] = {
        tws.theme.id: tws for tws in themes_with_segments
    }

    out: List[Selection] = []
    for tid, group in by_theme.items():
        total = sum(s.segment.duration for s in group)
        tws = pool_by_theme.get(tid)
        if total >= min_duration_floor or tws is None or not tws.segments:
            out.extend(group)
            continue
        used = {(s.segment.episode_number, round(s.segment.start, 1)) for s in group}
        ep_used = defaultdict(int)
        for s in group:
            ep_used[s.segment.episode_number] += 1
        # Best un-kept candidates: LATER episodes first (the series' most
        # developed treatment), then higher relevance.
        cands = sorted(
            [c for c in tws.segments
             if (c.episode_number, round(c.start, 1)) not in used],
            key=lambda c: (-c.episode_number, -c.relevance_score),
        )
        added: List[Selection] = []
        for c in cands:
            if total >= min_duration_floor:
                break
            if ep_used[c.episode_number] >= max_per_episode:
                continue
            added.append(Selection(
                segment=c,
                theme_title=tws.theme.title,
                theme_id=tid,
                beep_before="single",
            ))
            ep_used[c.episode_number] += 1
            used.add((c.episode_number, round(c.start, 1)))
            total += c.duration
        if added:
            logger.info(
                "Volume floor: theme '%s' %.0fs → %.0fs (+%d segment(s) "
                "from later episodes)",
                tid, total - sum(a.segment.duration for a in added),
                total, len(added),
            )
        out.extend(group + added)
    return out


def _place_conclusion_selections(
    selections: List[Selection],
) -> List[Selection]:
    """Selection-level variant of ``_place_conclusion_segments``: move theme-
    conclusion segments (signalled by the kept reason in ``text_preview``) to
    the END of playback order. Applied right before assembly so it covers the
    cache-HIT and volume-filled paths too, not just fresh LLM selections.
    """
    from podcastcondensor.minimal_theme_cut import _is_conclusion_reason

    conc_ids = {
        id(s) for s in selections if _is_conclusion_reason(s.segment.text_preview)
    }
    if not conc_ids:
        return selections
    return (
        [s for s in selections if id(s) not in conc_ids]
        + [s for s in selections if id(s) in conc_ids]
    )


def order_combined_selections(
    selections: List[Selection],
    theme_order: List[str],
) -> List[Selection]:
    """Reorder selections to playback order and assign beeps.

    ``theme_order`` is the resolved theme ids in the user's ``--theme`` arg
    order (the order the listener should hear them). Selections for a theme
    keep their internal (LLM-narrative) order. Beeps: none before the very
    first segment, triple at each theme transition, single within a theme.

    Pure/deterministic — no LLM, no audio. Mutates the returned Selection
    objects' ``beep_before`` in place.
    """
    ordered: List[Selection] = []
    for tid in theme_order:
        for s in selections:
            if s.theme_id == tid:
                ordered.append(s)
    if ordered:
        ordered[0].beep_before = "none"
        prev = ordered[0].theme_id
        for s in ordered[1:]:
            s.beep_before = "triple" if s.theme_id != prev else "single"
            prev = s.theme_id
    return ordered


def assemble_super_cut(
    selections: List[Selection],
    output_dir: str,
    cfg: Config,
) -> Dict[str, str]:
    """Group selections by theme and assemble one MP3 per theme (single beeps).

    Reuses ``assemble_master_cut`` per theme with normalized beeps (none before
    the first segment, single between). Returns {theme_id: mp3_path}.
    """
    by_theme: Dict[str, List[Selection]] = defaultdict(list)
    for s in selections:
        by_theme[s.theme_id].append(s)

    outputs: Dict[str, str] = {}
    used: set = set()
    for tid, group in by_theme.items():
        group[0].beep_before = "none"
        for s in group[1:]:
            s.beep_before = "single"

        base = re.sub(r"[^a-z0-9-]+", "-", tid.lower()).strip("-") or "theme"
        out_name = base + ".mp3"
        counter = 1
        while out_name in used:
            out_name = f"{base}-{counter}.mp3"
            counter += 1
        used.add(out_name)
        out_path = os.path.join(output_dir, out_name)

        assemble_master_cut(
            selections=group,
            output_path=out_path,
            sample_rate=cfg.audio_sample_rate,
            bitrate=cfg.audio_bitrate,
            speed=cfg.audio_speed,
            parallel_workers=4,
            keep_temp=cfg.keep_temp,
        )
        outputs[tid] = out_path
        logger.info(
            "Theme '%s' → %s (%.0fs)",
            tid, out_path, sum(s.segment.duration for s in group),
        )
    return outputs


def verify_cut_boundaries(
    selections: List[Selection],
    output_root: str,
) -> List[dict]:
    """Verify every selection lands on complete-sentence boundaries.

    Loads each episode's SRT, rebuilds sentence blocks, and checks that each
    selection's start coincides with a block START and its end with a block
    END — the hard "no mid-sentence cuts" guarantee. Records the literal last
    SRT line inside each cut as proof of what the listener actually hears.

    Never raises: a selection whose SRT can't be loaded is reported as
    UNVERIFIED, not dropped. Run on the POST-snap selections so it reflects
    exactly what gets assembled.
    """
    from podcastcondensor.subtitles import load_subtitles, build_sentence_blocks

    report: List[dict] = []
    loaded_eps: Dict[int, List[dict]] = {}
    for sel in selections:
        ep = sel.segment.episode_number
        if ep not in loaded_eps:
            srt_path = os.path.join(
                output_root, f"ep-{ep:03d}", "source_subtitles.srt"
            )
            try:
                loaded_eps[ep] = load_subtitles(srt_path, reindex=False)
            except FileNotFoundError:
                loaded_eps[ep] = []
        entries = loaded_eps.get(ep, [])
        start_ok = end_ok = False
        last_line = ""
        note = ""
        if entries:
            blocks = build_sentence_blocks(entries)
            # The "lands on a sentence-block boundary" check is membership in
            # the SET of block starts/ends — NOT "which block contains the
            # boundary". Blocks are contiguous, so a boundary (e.g. 1259.0)
            # lies at the end of one block AND the start of the next; a
            # contains-check matches the wrong block and yields false FAILs.
            start_ok = any(
                abs(b["start"] - sel.segment.start) < 0.01 for b in blocks
            )
            end_ok = any(
                abs(b["end"] - sel.segment.end) < 0.01 for b in blocks
            )
            # Literal last line = the text of the block that ENDS at the
            # selection's end boundary (the sentence the listener hears close).
            e_block = next(
                (b for b in blocks if abs(b["end"] - sel.segment.end) < 0.01),
                None,
            )
            if e_block:
                last_line = e_block.get("text", "")
        else:
            note = "SRT unavailable — unverified"
        report.append({
            "theme_id": sel.theme_id,
            "episode": ep,
            "start": round(sel.segment.start, 1),
            "end": round(sel.segment.end, 1),
            "duration": round(sel.segment.duration, 1),
            "start_ok": start_ok,
            "end_ok": end_ok,
            "last_line": last_line,
            "note": note,
        })
    return report


def write_boundary_report(
    report: List[dict],
    output_root: str,
    label: str,
) -> str:
    """Write a human-readable boundary-verification report to ``output_root``.

    ``label`` becomes the file suffix (e.g. ``flight`` →
    ``super_cut_verify_flight_001_144.txt``). Returns the report path.
    """
    lines = [
        f"Boundary verification — {label}",
        f"Checked {len(report)} segment(s) against sentence-block boundaries.",
        "Each segment must START at a complete-sentence block start and END at",
        "a complete-sentence block end — no mid-sentence cuts.",
        "",
    ]
    bad = [r for r in report if not (r["start_ok"] and r["end_ok"]) and not r["note"]]
    unverified = [r for r in report if r["note"]]
    ok = len(report) - len(bad) - len(unverified)
    lines.append(f"OK: {ok}   VIOLATIONS: {len(bad)}   UNVERIFIED: {len(unverified)}")
    lines.append("")
    for r in report:
        if r["note"]:
            status = "?  "
        elif r["start_ok"] and r["end_ok"]:
            status = "OK "
        else:
            status = "FAIL"
        lines.append(
            f"[{status}] {r['theme_id']}  ep{r['episode']:03d}  "
            f"{r['start']:.1f}-{r['end']:.1f}s ({r['duration']:.0f}s)"
        )
        lines.append(f"       last line: {r['last_line'][:170]}")
        if r["note"]:
            lines.append(f"       note: {r['note']}")
    path = os.path.join(output_root, f"super_cut_verify_{label}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote boundary verification report: %s", path)
    return path


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _episode_span_str(episodes: List[int]) -> str:
    """Compact '1-5, 8, 19, 79' range string for a sorted episode list."""
    if not episodes:
        return "?"
    parts: List[str] = []
    start = prev = episodes[0]
    for ep in episodes[1:]:
        if ep == prev + 1:
            prev = ep
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = ep
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _themes_report_data(
    global_themes: List[GlobalTheme],
    selected_by_theme: Optional[Dict[str, int]] = None,
) -> dict:
    selected = selected_by_theme or {}
    for gt in global_themes:
        gt.selected_count = selected.get(gt.id, 0)
    return {
        "themes": [
            {
                "id": gt.id,
                "title": gt.title,
                "description": gt.description,
                "importance": gt.importance,
                "source_theme_ids": gt.source_theme_ids,
                "related_item_ids": gt.related_item_ids,
                "candidate_count": gt.candidate_count,
                "episode_count": gt.episode_count,
                "episode_numbers": gt.episode_numbers,
                "selected_count": gt.selected_count,
            }
            for gt in global_themes
        ],
    }


def print_theme_table(global_themes: List[GlobalTheme]) -> None:
    """Print the "top topics over all the universe" table."""
    print("=" * 60)
    print("TOP TOPICS OVER ALL EPISODES")
    print("=" * 60)
    for gt in sorted(global_themes, key=lambda g: -g.importance):
        sel = gt.selected_count if gt.selected_count else "-"
        span = _episode_span_str(gt.episode_numbers)
        print(
            f"  {gt.title}  (imp={gt.importance:.2f}, eps {span}, "
            f"{gt.candidate_count} candidates, {sel} selected)"
        )
        print(f"      {gt.description[:130]}")
    print("")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_super_cut(
    playlist_url: str,
    cfg: Config,
    *,
    start_episode: int = 1,
    end_episode: int = 144,
    chunk_size: int = 12,
    max_themes: int = 25,
    theme_ids: Optional[List[str]] = None,
    retry_empty_chunks: bool = False,
    output_dir: str = "",
    dry_run: bool = False,
    combine_name: Optional[str] = None,
    min_duration_floor: float = 0.0,
) -> dict:
    """Build per-theme MP3s for the top topics over the full episode range.

    Offline from disk: reads per-episode global_state.json, merges into a
    cumulative universe, chunked theme extraction → coalesce → episode-diverse
    candidate resolution → per-theme LLM selection (LLM-owned volume) → one
    MP3 per theme.

    ``dry_run`` runs merge → chunk → coalesce → resolve only and prints the
    theme table (with per-theme episode spread) — the "many episodes quoted"
    check before spending on selection.

    ``combine_name`` (with ``theme_ids``) assembles ONE MP3 instead of one per
    theme: selections are reordered to the ``--theme`` arg order (playback
    order) with triple beeps between themes and single beeps within — the
    "combined theme cut" for a single listening session. Every assembled
    selection is verified to land on sentence-block boundaries and the report
    is written to ``super_cut_verify_<label>.txt``.
    """
    output_root = cfg.output_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    result: dict = {
        "phases": [],
        "errors": [],
        "outputs": {},
        "dry_run": dry_run,
    }
    overall_t0 = time.time()

    # ── Phase 1: Cumulative universe merge (offline, 0 API calls) ────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Merge per-episode global_state into cumulative universe")
    logger.info("=" * 60)
    t1 = time.time()
    universe_path = os.path.join(
        output_root, f"universe_state_{start_episode:03d}_{end_episode:03d}.json"
    )
    universe_data = build_universe_from_global_states(
        output_root, start_episode, end_episode, state_path=universe_path,
    )
    in_state = {s.get("episode_number") for s in universe_data.get("episode_summaries", [])}
    n_eps = len(in_state)
    n_items = sum(len(universe_data.get(c, [])) for c in CATEGORIES)
    if start_episode <= 30:
        missing = [e for e in range(start_episode, min(end_episode, 30) + 1) if e not in in_state]
        if missing:
            logger.warning(
                "Missing global_state for %d early episode(s) %s — run "
                "build-universe --start 1 --end 30 first",
                len(missing), missing[:5],
            )
    result["phases"].append({
        "phase": "merge_universe",
        "elapsed_sec": round(time.time() - t1, 1),
        "episodes": n_eps,
        "items": n_items,
    })
    logger.info("Cumulative universe: %d episodes, %d items", n_eps, n_items)

    # ── Phase 2+3: Theme discovery (chunk extraction + coalesce) ──────────
    # Cached on disk so repeated --theme cuts reuse it (0 discovery calls).
    logger.info("=" * 60)
    logger.info("PHASE 2+3: Theme discovery (chunked extraction → coalesce)")
    logger.info("=" * 60)
    api_key = resolve_api_key()
    if not api_key:
        result["errors"].append("DeepSeek API key not set")
        return result
    ds_client = DeepSeekClient(api_key=api_key)

    discovery_path = _discovery_cache_path(output_root, start_episode, end_episode)
    cached = _load_discovery_cache(discovery_path)
    if cached is not None:
        chunk_episodes, all_records, global_themes = cached
        result["phases"].append({
            "phase": "discovery",
            "cached": True,
            "chunks": len(chunk_episodes),
            "chunk_theme_count": len(all_records),
            "global_theme_count": len(global_themes),
        })
        logger.info(
            "Discovery cache HIT: %d chunks, %d chunk themes, %d global themes "
            "(0 LLM calls)",
            len(chunk_episodes), len(all_records), len(global_themes),
        )

        # Targeted recovery: re-attempt ONLY chunks that produced 0 themes,
        # then re-coalesce. Avoids a full discovery re-run.
        if retry_empty_chunks:
            retried = _retry_empty_chunks(
                chunk_episodes, all_records, output_root, ds_client, cfg,
            )
            if retried:
                global_themes = coalesce_themes(
                    all_records, chunk_episodes, ds_client, cfg,
                    max_themes=max_themes,
                )
                _save_discovery_cache(
                    discovery_path, chunk_episodes, all_records, global_themes,
                )
                logger.info(
                    "Re-coalesced after retrying %d empty chunk(s): %d global themes",
                    retried, len(global_themes),
                )
            else:
                logger.info("No empty chunks to retry")
    else:
        t2 = time.time()
        chunks = build_chunks(output_root, start_episode, end_episode, chunk_size)
        if not chunks:
            result["errors"].append("No episodes with global_state.json in range")
            return result
        all_records = []
        chunk_episodes = {}
        for ci, ep_list in enumerate(chunks):
            records = extract_chunk_themes(ep_list, output_root, ds_client, cfg, ci)
            logger.info(
                "Chunk %d (eps %d-%d): %d themes",
                ci, ep_list[0], ep_list[-1], len(records),
            )
            all_records.extend(records)
            chunk_episodes[ci] = ep_list
        result["phases"].append({
            "phase": "chunk_themes",
            "elapsed_sec": round(time.time() - t2, 1),
            "chunks": len(chunks),
            "chunk_theme_count": len(all_records),
        })

        t3 = time.time()
        global_themes = coalesce_themes(
            all_records, chunk_episodes, ds_client, cfg, max_themes=max_themes,
        )
        if not global_themes:
            result["errors"].append("No global themes coalesced — cannot continue")
            return result
        result["phases"].append({
            "phase": "coalesce",
            "elapsed_sec": round(time.time() - t3, 1),
            "global_theme_count": len(global_themes),
            "theme_ids": [gt.id for gt in global_themes],
        })
        _save_discovery_cache(discovery_path, chunk_episodes, all_records, global_themes)

    # ── Phase 4: Resolve ALL themes' candidates + cap (0 API calls) ──────
    logger.info("=" * 60)
    logger.info("PHASE 4: Resolve ALL themes' candidate segments (0 API calls)")
    logger.info("=" * 60)
    t4 = time.time()
    manifests = _scan_existing_episodes(output_root, start_episode, end_episode)
    record_by_id = {r.theme_id: r for r in all_records}
    themes_with_segments = resolve_global_theme_candidates(
        global_themes, record_by_id, universe_data, manifests, output_root,
    )
    total_candidates = sum(len(tws.segments) for tws in themes_with_segments)
    result["phases"].append({
        "phase": "resolve_segments",
        "elapsed_sec": round(time.time() - t4, 1),
        "total_candidates": total_candidates,
    })
    logger.info("Segment resolution: %d total candidates across %d themes",
                total_candidates, len(global_themes))

    # Persist candidates cache (deterministic — bracket analysis reads this).
    candidates_path = _candidates_cache_path(output_root, start_episode, end_episode)
    _save_candidates_cache(candidates_path, global_themes, themes_with_segments)
    result["candidates_path"] = candidates_path

    # Load selections cache (persisted per cut — 0 calls for already-cut themes).
    selections_path = _selections_cache_path(output_root, start_episode, end_episode)
    selections_cache = _load_selections_cache(selections_path)
    selected_by_theme = {tid: len(sels) for tid, sels in selections_cache.items()}

    # Write the FULL themes report — always all themes, never the filtered subset.
    themes_path = os.path.join(
        output_root, f"super_cut_themes_{start_episode:03d}_{end_episode:03d}.json"
    )
    with open(themes_path, "w", encoding="utf-8") as f:
        json.dump(_themes_report_data(global_themes, selected_by_theme),
                  f, ensure_ascii=False, indent=2)
    result["themes_path"] = themes_path

    if dry_run:
        print_theme_table(global_themes)
        logger.info("Dry run complete — themes report: %s", themes_path)
        return result

    # ── Resolve requested themes (--theme filter, else all) ───────────────
    # ``wanted`` is an ORDERED list, not a set: it preserves the order the
    # user passed ``--theme`` args, which is the playback order for a
    # combined cut.
    if theme_ids:
        wanted: List[str] = []
        for f in theme_ids:
            fl = f.strip().lower()
            for gt in global_themes:
                if fl and (fl in gt.id.lower() or fl in gt.title.lower()):
                    if gt.id not in wanted:
                        wanted.append(gt.id)
        if not wanted:
            result["errors"].append(
                f"No global theme matches --theme {theme_ids} "
                f"(have {[gt.id for gt in global_themes][:10]})"
            )
            return result
        keep_idx = [i for i, gt in enumerate(global_themes) if gt.id in wanted]
        requested_global = [global_themes[i] for i in keep_idx]
        requested_tws = [themes_with_segments[i] for i in keep_idx]
        logger.info("Requested %d theme(s): %s",
                    len(requested_global), [gt.id for gt in requested_global])
    else:
        requested_global = global_themes
        requested_tws = themes_with_segments

    # ── Phase 5: Selection — skip cached themes (0 calls), select the rest ─
    logger.info("=" * 60)
    logger.info("PHASE 5: Per-theme LLM selection (cached themes skipped)")
    logger.info("=" * 60)
    t5 = time.time()
    all_selections: List[Selection] = []
    newly_selected: Dict[str, List[Selection]] = {}
    to_select: List[ThemeWithSegments] = []
    for gt, tws in zip(requested_global, requested_tws):
        cached = selections_cache.get(gt.id)
        if cached:
            cached_sels = _selections_from_dicts(cached)
            all_selections.extend(cached_sels)
            logger.info("Theme '%s': cache HIT (%d segments, 0 LLM calls)",
                        gt.id, len(cached_sels))
        elif tws.segments:
            to_select.append(tws)
        else:
            logger.info("Theme '%s': no candidates — skipping", gt.id)

    if to_select:
        plan = select_segments_for_master_cut(
            to_select,
            manifests=manifests,
            output_root=output_root,
            client=ds_client,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout,
            target_duration=float(cfg.master_cut_target_duration),
        )
        for s in plan.selections:
            newly_selected.setdefault(s.theme_id, []).append(s)
        all_selections.extend(plan.selections)
        if plan.selections:
            _merge_selections_cache(selections_path, newly_selected)
            selections_cache = _load_selections_cache(selections_path)
        logger.info("Newly selected: %d segments across %d theme(s)",
                    len(plan.selections), len(newly_selected))
    else:
        plan = MasterCutPlan(
            selections=all_selections,
            total_duration=sum(s.segment.duration for s in all_selections),
            theme_allocations={}, coverage={},
        )
        logger.info("All requested themes already selected — 0 LLM calls")

    result["phases"].append({
        "phase": "select_segments",
        "elapsed_sec": round(time.time() - t5, 1),
        "selected_count": len(all_selections),
        "total_duration_sec": plan.total_duration,
        "llm_calls": len(to_select),
    })

    if not all_selections:
        result["errors"].append("No segments selected — cannot continue")
        return result

    # ── Deterministic volume floor (opt-in) ─────────────────────────────
    # DeepSeek under-selects volume unreliably, so the LLM's kept set is the
    # quality ranking + narrative spine, and the floor fills remaining runtime
    # with the theme's best un-kept candidates (per-episode capped, later
    # episodes preferred). ``--min-duration-floor`` opts a curated cut into
    # guaranteed volume; 0 = pure LLM volume (archive default). Universal.
    if min_duration_floor and min_duration_floor > 0:
        all_selections = fill_selection_volume(
            all_selections,
            requested_tws,
            min_duration_floor=float(min_duration_floor),
        )
        plan.total_duration = round(
            sum(s.segment.duration for s in all_selections), 1
        )
        logger.info(
            "After volume floor: %d segments, %.0fs (%.1f min @1x)",
            len(all_selections), plan.total_duration, plan.total_duration / 60,
        )

    # ── Re-snap cached AND fresh selections to sentence blocks ──────────
    # A stale selections cache (boundaries written before/without the snap)
    # must never ship mid-sentence cuts. Re-snap everything deterministically
    # (0 LLM calls) and write the corrected boundaries back to the cache so it
    # self-heals. Keeps the LLM's editorial choice; only aligns boundaries.
    resnapped = _resnap_selections(all_selections, output_root)
    changed = sum(
        1 for a, b in zip(all_selections, resnapped)
        if abs(a.segment.start - b.segment.start) > 0.01
        or abs(a.segment.end - b.segment.end) > 0.01
    )
    if changed:
        logger.info(
            "Re-snapped %d/%d segment boundaries to sentence blocks "
            "(stale cache self-healed)",
            changed, len(all_selections),
        )
        all_selections = resnapped
        plan.total_duration = round(
            sum(s.segment.duration for s in all_selections), 1
        )
        by_theme: Dict[str, List[Selection]] = {}
        for s in all_selections:
            by_theme.setdefault(s.theme_id, []).append(s)
        _merge_selections_cache(selections_path, by_theme)

    warnings = _compute_selection_warnings(
        all_selections, [tws.theme for tws in requested_tws],
    )
    result["warnings"] = warnings

    # ── Boundary verification (post-snap, on what will actually be cut) ──
    # Hard "no mid-sentence cuts" proof: every selection's start/end must land
    # on a sentence-block boundary, with the literal last line recorded.
    boundary_report = verify_cut_boundaries(all_selections, output_root)

    # ── Phase 6: Assemble ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 6: Assemble audio")
    logger.info("=" * 60)
    if not output_dir:
        output_dir = output_root  # flat — MP3s go straight into output_root, no subfolder
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    t6 = time.time()

    # Authoritative final playback order: theme-conclusion segments LAST. This
    # runs on the assembled list (fresh, cache-HIT, or volume-filled alike) so
    # the LLM's "Concludes..." segment can never play mid-arc. The combine
    # reorder below preserves within-theme order, so placement survives combine.
    all_selections = _place_conclusion_selections(all_selections)

    combine_base = ""
    if combine_name:
        if not theme_ids:
            result["errors"].append("--combine requires at least one --theme")
            return result
        # Combined cut: reorder to the user's --theme playback order and set
        # triple beeps between themes, single within. ``wanted`` is already
        # the ordered list of resolved theme ids.
        combine_base = re.sub(r"[^a-z0-9-]+", "-", combine_name.lower()).strip("-")
        combine_base = combine_base or "combined"
        all_selections = order_combined_selections(all_selections, wanted)
        if not all_selections:
            result["errors"].append("No selections to assemble")
            return result
        plan.total_duration = round(sum(s.segment.duration for s in all_selections), 1)
        out_path = os.path.join(output_dir, combine_base + ".mp3")
        try:
            assemble_master_cut(
                selections=all_selections,
                output_path=out_path,
                sample_rate=cfg.audio_sample_rate,
                bitrate=cfg.audio_bitrate,
                speed=cfg.audio_speed,
                parallel_workers=4,
                keep_temp=cfg.keep_temp,
            )
            result["outputs"] = {combine_name: out_path}
        except Exception as e:
            logger.exception("Combined cut assembly failed: %s", e)
            result["errors"].append(f"Audio assembly failed: {e}")
    else:
        try:
            result["outputs"] = assemble_super_cut(all_selections, output_dir, cfg)
        except Exception as e:
            logger.exception("Super cut assembly failed: %s", e)
            result["errors"].append(f"Audio assembly failed: {e}")

    # Write the boundary report (label = combine name for combined runs).
    verify_label = f"{combine_base}_{start_episode:03d}_{end_episode:03d}" \
        if combine_name else f"{start_episode:03d}_{end_episode:03d}"
    result["verify_path"] = write_boundary_report(
        boundary_report, output_root, verify_label,
    )

    result["phases"].append({
        "phase": "assemble_audio",
        "elapsed_sec": round(time.time() - t6, 1),
        "output_dir": output_dir,
    })

    # ── Final reports (full themes report + stats) ───────────────────────
    selected_by_theme = {tid: len(sels) for tid, sels in selections_cache.items()}
    with open(themes_path, "w", encoding="utf-8") as f:
        json.dump(_themes_report_data(global_themes, selected_by_theme),
                  f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(
        output_root, f"super_cut_stats_{start_episode:03d}_{end_episode:03d}.json"
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_duration_sec": plan.total_duration,
            "themes": len(requested_global),
            "segments": len(all_selections),
            "warnings": warnings,
            "errors": result.get("errors", []),
            "phases": result.get("phases", []),
            "verify_path": result.get("verify_path", ""),
            "selections": [
                {
                    "theme_title": s.theme_title,
                    "theme_id": s.theme_id,
                    "episode": s.segment.episode_number,
                    "start": round(s.segment.start, 1),
                    "end": round(s.segment.end, 1),
                    "duration": round(s.segment.duration, 1),
                    "beep_before": s.beep_before,
                }
                for s in all_selections
            ],
        }, f, indent=2)
    result["stats_path"] = stats_path

    total_elapsed = time.time() - overall_t0
    print_theme_table(global_themes)
    logger.info(
        "SUPER CUT COMPLETE — %d themes cut, %d segments, %.0fs (%.1fh), "
        "outputs in %s (%.1fs total)",
        len(requested_global), len(all_selections), plan.total_duration,
        plan.total_duration / 3600, output_dir, total_elapsed,
    )
    return result
