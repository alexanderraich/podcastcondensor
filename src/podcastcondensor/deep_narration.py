"""Deep per-episode narration — transcript → arc → narration → TTS.

The current product (CLAUDE.md, 2026-09-22). Cutting the original audio failed
on *delivery* — the hosts' pauses and jokes lose the listener — and the digest
was built for *global* condensing, not for telling one episode. So this path
skips both: it reads the transcript directly, maps the episode's arc, and
writes a spoken narration rendered by a synthetic voice.

Two logical stages:

1. ``build_arc`` — one DeepSeek call per transcript chunk (~70 min of audio
   each, ``prompts/arc_identify.txt``). Each call returns the chunk's
   movements: what they establish, what merely restates earlier material *in
   the same episode*, what connects back to earlier *episodes*, and which Q&A
   earns its place. Each chunk also returns a bounded ``carry_forward`` note
   that seeds the next chunk's prompt, so restatement is recognised across
   chunk boundaries.
2. ``narrate_deep`` — one call: the assembled arc → the narration prose
   (``prompts/narrate_deep.txt``).

Both stages have hard input ceilings (``_MAX_CHUNK_WORDS``, ``_ARC_MAX_WORDS``):
the point is that each stage's input fits in a single prompt. Episode length is
absorbed by the *chunk count*, never by growing the prompt. The arc caps are
ceilings, not quotas — a chunk that is all restatement returns a short section
and nothing back-fills it.

Artifacts (per episode):
- ``ep-NNN_arc_deep.json``      — per-chunk arc, ✅ git-versioned (a prompt change is a diff)
- ``ep-NNN_narration_deep.txt`` — the narration prose, ✅ git-versioned
- ``ep-NNN_narration_deep.mp3`` — the edge-tts render, ❌ gitignored

Resumable at three levels: an episode with an MP3 is skipped entirely, an
episode with a narration text skips both LLM stages, and a partially-mined arc
reuses the chunks that already succeeded.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from podcastcondensor.llm.deepseek import DeepSeekClient, resolve_api_key
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.summary_doc import _default_output_root, _episode_title, _ts
from podcastcondensor.tts import synthesize_narration
from podcastcondensor.ultra_cut import _parse_json_object

logger = logging.getLogger(__name__)

_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
)
_ARC_PROMPT_PATH = os.path.join(_PROMPT_DIR, "arc_identify.txt")
_NARRATE_PROMPT_PATH = os.path.join(_PROMPT_DIR, "narrate_deep.txt")

# ── Prompt-fit limits — the whole point of the design ──────────────────
# Stage 1 input is one transcript chunk. Measured rate on CLEANED entries is
# ~2.52 words/s (ep-144: 5,632 s → 14,215 words), so 10,000 words ≈ 66 min of
# audio ≈ 13k tokens. With the prompt and the ≤2,000-word output that is ~18k
# of a 64K window — and, usefully, three chunks for a three-hour episode,
# which is the hosts' own "three halves" shape.
_MAX_CHUNK_WORDS = 10000
# Stage 1 output: a CEILING per chunk, never a quota. Set above the narration
# target on purpose — the arc is the only carrier of substance into stage 2
# (which never sees the transcript), so it must hold more than the narration
# is allowed to keep, or stage 2 would have nothing to select from.
_ARC_MAX_WORDS = 2000
_ARC_MAX_TOKENS = 4000
# The carry-forward note is the only thing that grows across chunks, so it is
# bounded twice: per chunk, and in how many chunks are worth keeping.
_CARRY_FORWARD_WORDS = 150
_CARRY_FORWARD_MAX_CHUNKS = 4

# Stage 2: the assembled arc (~4,500 words for a 3-chunk episode) → narration.
_NARRATION_MAX_TOKENS = 6000
_TARGET_MIN_WORDS = 2600
_TARGET_MAX_WORDS = 3600

# Coarse timestamps: one marker per paragraph, not one per cue. A 3h episode
# has ~4,000 cues; marking each would add ~5k tokens of pure timestamp.
_WORDS_PER_PARAGRAPH = 150

_ARC_FILENAME = "arc_deep.json"
_TEXT_FILENAME = "narration_deep.txt"
_MP3_FILENAME = "narration_deep.mp3"


def _ep_file(ep_num: int, name: str) -> str:
    """Episode-scoped artefact filename.

    The episode number belongs in the *filename*, not just the directory: the
    MP3 gets copied out to a phone, where ``output/ep-145/`` no longer exists
    to say which episode it is. (Renamed 2026-09-22 from the bare
    ``narration_deep.mp3`` for exactly that reason.)
    """
    return f"ep-{ep_num:03d}_{name}"


def _load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Transcript → chunks
# ---------------------------------------------------------------------------


def paragraphs(entries: List[dict], words_per_para: int = _WORDS_PER_PARAGRAPH) -> List[str]:
    """Group cleaned SRT cues into ~``words_per_para`` paragraphs, timestamped.

    Each paragraph is prefixed with its absolute episode timestamp (``[H:MM:SS]``)
    so the arc can report movement boundaries without a timestamp on every cue.
    """
    paras: List[str] = []
    buf: List[str] = []
    buf_words = 0
    buf_start = 0.0
    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        if not buf:
            buf_start = entry.get("start", 0.0)
        buf.append(text)
        buf_words += len(text.split())
        if buf_words >= words_per_para:
            paras.append(f"[{_ts(buf_start)}] " + " ".join(buf))
            buf, buf_words = [], 0
    if buf:
        paras.append(f"[{_ts(buf_start)}] " + " ".join(buf))
    return paras


def chunk_transcript(
    entries: List[dict], max_words: int = _MAX_CHUNK_WORDS
) -> List[List[dict]]:
    """Split cleaned SRT entries into chunks of at most ``max_words`` words.

    Boundaries fall wherever the word budget runs out, so a split may land
    mid-thought. That is harmless: chunks are only ever input to the arc call,
    never cut points in audio, and each chunk is handed the previous chunk's
    ``carry_forward`` note so restatement is still recognised across the seam.
    """
    chunks: List[List[dict]] = []
    current: List[dict] = []
    words = 0
    for entry in entries:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        current.append(entry)
        words += len(text.split())
        if words >= max_words:
            chunks.append(current)
            current, words = [], 0
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Stage 1 — arc identification
# ---------------------------------------------------------------------------


def _build_arc_prompt(
    chunk_text: str,
    ep_num: int,
    title: str,
    index: int,
    total: int,
    carry_forward: str,
) -> str:
    parts = [f"Episode {ep_num} — {title}", f"Part {index + 1} of {total}."]
    if carry_forward:
        parts.append(
            "Previously established in this episode:\n" + carry_forward
        )
    parts.append("TRANSCRIPT:\n" + chunk_text)
    return _load_prompt(_ARC_PROMPT_PATH) + "\n\n" + "\n\n".join(parts)


def identify_arc_chunk(
    chunk_text: str,
    ep_num: int,
    title: str,
    index: int,
    total: int,
    carry_forward: str,
    client: DeepSeekClient,
) -> Dict:
    """One DeepSeek call: a transcript chunk → movements + carry_forward.

    Retries once, matching ``narration.narrate_episode``. Raises if the
    response cannot be parsed into an object — a chunk that silently yields no
    movements would look like a thin episode, not a failed call.
    """
    prompt = _build_arc_prompt(chunk_text, ep_num, title, index, total, carry_forward)
    last_error = None
    for _attempt in range(2):
        try:
            raw = client.generate(
                prompt=prompt,
                timeout=900,
                temperature=0.1,
                max_tokens=_ARC_MAX_TOKENS,
                force_json=True,
            )
            data = _parse_json_object(raw)
            if data is None:
                raise RuntimeError("response was not parseable JSON")
            return data
        except Exception as e:  # noqa: BLE001 — mirror narration.py's retry
            last_error = e
            logger.warning(
                "Arc call failed for ep %d part %d/%d: %s — retrying",
                ep_num, index + 1, total, e,
            )
    raise RuntimeError(
        f"Arc identification failed for ep {ep_num} part {index + 1}/{total}: {last_error}"
    )


def _carry_forward_text(chunks: List[dict]) -> str:
    """The bounded carry-forward handed to the next chunk's prompt.

    Only the last ``_CARRY_FORWARD_MAX_CHUNKS`` notes are kept (each already
    capped at ``_CARRY_FORWARD_WORDS`` by the prompt), which bounds the growth:
    the prompt does not get longer as the episode does.
    """
    notes = [c.get("carry_forward", "").strip() for c in chunks]
    notes = [n for n in notes if n][-_CARRY_FORWARD_MAX_CHUNKS:]
    return "\n".join(notes)


def _arc_cache_path(output_root: str, ep_num: int) -> str:
    return os.path.join(
        output_root, f"ep-{ep_num:03d}", _ep_file(ep_num, _ARC_FILENAME)
    )


def _load_arc_cache(path: str, chunk_words: int) -> List[dict]:
    """Load cached per-chunk arcs, or [] when absent/stale/incompatible.

    ``chunk_words`` is stored in the cache: changing it re-splits the
    transcript, which shifts every chunk index, so the cache is discarded
    rather than silently mis-matched.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Unreadable arc cache %s (%s) — rebuilding", path, e)
        return []
    if data.get("chunk_words") != chunk_words:
        logger.info("Arc cache built with different chunk size — rebuilding")
        return []
    chunks = data.get("chunks")
    return chunks if isinstance(chunks, list) else []


def _save_arc_cache(path: str, ep_num: int, title: str, chunk_words: int,
                    chunks: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "episode": ep_num,
        "title": title,
        "chunk_words": chunk_words,
        "chunks": chunks,
        "movements": [m for c in chunks for m in c.get("movements", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_arc(
    output_root: str,
    ep_num: int,
    title: str,
    client: Optional[DeepSeekClient] = None,
    chunk_words: int = _MAX_CHUNK_WORDS,
    force: bool = False,
) -> Dict:
    """Stage 1: the episode's transcript → the assembled arc.

    One call per chunk; a chunk already in the cache is reused (0 calls) and
    still contributes its ``carry_forward`` to the next chunk's prompt.
    Returns the cache payload (``movements`` flattened across chunks).
    """
    ep_dir = os.path.join(output_root, f"ep-{ep_num:03d}")
    srt_path = os.path.join(ep_dir, "source_subtitles.srt")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"No transcript for ep {ep_num}: {srt_path}")

    cache_path = _arc_cache_path(output_root, ep_num)
    cached = [] if force else _load_arc_cache(cache_path, chunk_words)
    done = {c.get("index"): c for c in cached}

    entries = load_subtitles(srt_path)
    if not entries:
        raise RuntimeError(f"Empty transcript for ep {ep_num}: {srt_path}")
    raw_chunks = chunk_transcript(entries, max_words=chunk_words)
    total = len(raw_chunks)
    logger.info(
        "Ep %d: %d entries → %d chunk(s) (cache: %d done)",
        ep_num, len(entries), total, len(done),
    )

    if client is None:
        client = DeepSeekClient(api_key=resolve_api_key())

    chunks: List[dict] = []
    for i, raw in enumerate(raw_chunks):
        if i in done:
            logger.info("Ep %d part %d/%d: cached, reusing", ep_num, i + 1, total)
            chunks.append(done[i])
            continue
        chunk_text = "\n\n".join(paragraphs(raw))
        data = identify_arc_chunk(
            chunk_text=chunk_text,
            ep_num=ep_num,
            title=title,
            index=i,
            total=total,
            carry_forward=_carry_forward_text(chunks),
            client=client,
        )
        movements = data.get("movements") or []
        chunks.append({
            "index": i,
            "start": raw[0].get("start", 0.0),
            "end": raw[-1].get("end", 0.0),
            "words": sum(len((e.get("text") or "").split()) for e in raw),
            "movements": movements,
            "carry_forward": data.get("carry_forward", ""),
        })
        logger.info(
            "Ep %d part %d/%d: %d movements (substance %s)",
            ep_num, i + 1, total, len(movements),
            ",".join(str(m.get("substance", "?")) for m in movements) or "none",
        )
        # Persist after every chunk — a later failure must not re-spend these.
        _save_arc_cache(cache_path, ep_num, title, chunk_words, chunks)

    return {
        "episode": ep_num,
        "title": title,
        "chunk_words": chunk_words,
        "chunks": chunks,
        "movements": [m for c in chunks for m in c.get("movements", [])],
    }


# ---------------------------------------------------------------------------
# Stage 2 — the narration
# ---------------------------------------------------------------------------


def _render_arc(arc: Dict, ep_num: int, title: str) -> str:
    """Render the arc as readable text for the narration prompt.

    Text rather than raw JSON: the second call is writing prose, and a
    labelled structure reads as content where a JSON blob reads as data.
    """
    lines = [f"# Episode {ep_num} — {title}", ""]
    movements = arc.get("movements") or []
    if not movements:
        raise RuntimeError(f"Arc for ep {ep_num} has no movements")
    for i, mv in enumerate(movements, 1):
        lines.append(f"## Movement {i}: {mv.get('label', '(unnamed)')}")
        lines.append(f"Substance: {mv.get('substance', 'medium')}")
        if mv.get("develops"):
            lines.append(f"Establishes: {mv['develops']}")
        if mv.get("restates"):
            lines.append(f"Restates earlier in this episode: {mv['restates']}")
        if mv.get("backrefs"):
            lines.append(f"Connects back to earlier episodes: {mv['backrefs']}")
        if mv.get("qa"):
            merit = (mv.get("qa_merit") or "").strip()
            lines.append(f"Q&A — worth keeping: {merit}" if merit else "Q&A — redundant")
        key_points = mv.get("key_points") or []
        if key_points:
            lines.append("Key points:")
            lines.extend(f"- {p}" for p in key_points)
        scripture = mv.get("scripture") or []
        if scripture:
            lines.append("Scripture: " + "; ".join(str(s) for s in scripture))
        lines.append("")
    return "\n".join(lines)


def narrate_deep(
    arc: Dict,
    ep_num: int,
    title: str,
    client: DeepSeekClient,
    max_tokens: int = _NARRATION_MAX_TOKENS,
) -> str:
    """Stage 2: the arc → the narration prose. One call, retried once.

    The transcript is deliberately NOT sent — the arc already carries the
    substance, and handing over both would blow the prompt budget the design
    exists to protect.
    """
    prompt = _load_prompt(_NARRATE_PROMPT_PATH) + "\n\n" + _render_arc(arc, ep_num, title)
    last_error = None
    for _attempt in range(2):
        try:
            narration = client.generate(
                prompt=prompt,
                timeout=900,
                temperature=0.3,
                max_tokens=max_tokens,
            ).strip()
            if not narration:
                raise RuntimeError("empty narration response")
            return narration
        except Exception as e:  # noqa: BLE001 — mirror narration.py's retry
            last_error = e
            logger.warning("Narration call failed for ep %d: %s — retrying", ep_num, e)
    raise RuntimeError(f"Narration failed for ep {ep_num}: {last_error}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fetch_episodes(
    output_root: str,
    ep_nums: List[int],
    playlist_url: str,
    whisper_model: str = "base",
    beam_size: int = 1,
    vad: bool = False,
) -> List[int]:
    """Download + transcribe the requested episodes that aren't already on disk.

    One call per episode so that fetching {145, 147} does not also pull 146.
    Already-present transcripts skip YouTube entirely (the on-disk scan runs
    first inside ``ensure_all_episode_artifacts``). Fail-loud: raises if any
    requested episode still has no transcript afterwards.
    """
    from podcastcondensor.download_pool import ensure_all_episode_artifacts

    for ep in ep_nums:
        srt = os.path.join(output_root, f"ep-{ep:03d}", "source_subtitles.srt")
        if os.path.exists(srt):
            logger.info("Ep %d: transcript already on disk — no fetch needed", ep)
            continue
        logger.info("Ep %d: fetching + transcribing (whisper %s, beam %d, vad %s)",
                    ep, whisper_model, beam_size, vad)
        ensure_all_episode_artifacts(
            playlist_url=playlist_url,
            output_root=output_root,
            start_episode=ep,
            end_episode=ep,
            parallel=1,
            whisper_model=whisper_model,
            whisper_beam_size=beam_size,
            whisper_vad=vad,
        )

    missing = [
        ep for ep in ep_nums
        if not os.path.exists(
            os.path.join(output_root, f"ep-{ep:03d}", "source_subtitles.srt")
        )
    ]
    if missing:
        raise RuntimeError(f"Fetch incomplete — no transcript for: {missing}")
    return ep_nums


def build_deep_narration(
    output_root: str = "",
    ep_num: int = 1,
    client: Optional[DeepSeekClient] = None,
    skip_tts: bool = False,
    force: bool = False,
    chunk_words: int = _MAX_CHUNK_WORDS,
) -> Dict:
    """Run the whole path for one episode: arc → narration → MP3.

    Resumable: an existing ``ep-NNN_narration_deep.mp3`` short-circuits the
    episode entirely (``force`` overrides), an existing
    ``ep-NNN_narration_deep.txt`` skips both LLM stages, and a partial
    ``ep-NNN_arc_deep.json`` reuses its finished chunks. The DeepSeek client
    is created lazily so a pure TTS re-render needs no API key.
    """
    root = output_root or _default_output_root()
    ep_dir = os.path.join(root, f"ep-{ep_num:03d}")
    srt_path = os.path.join(ep_dir, "source_subtitles.srt")
    arc_path = os.path.join(ep_dir, _ep_file(ep_num, _ARC_FILENAME))
    text_path = os.path.join(ep_dir, _ep_file(ep_num, _TEXT_FILENAME))
    mp3_path = os.path.join(ep_dir, _ep_file(ep_num, _MP3_FILENAME))

    if not force and os.path.exists(mp3_path):
        logger.info("Ep %d: %s already on disk — skipping", ep_num,
                    os.path.basename(mp3_path))
        return {
            "episode": ep_num, "mp3": mp3_path, "arc": arc_path,
            "narration": text_path, "words": 0, "skipped": True,
        }

    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"No transcript for ep {ep_num}: {srt_path}")
    title = _episode_title(root, ep_num)

    # ── Stage 1 ────────────────────────────────────────────────────────
    arc = build_arc(
        output_root=root, ep_num=ep_num, title=title,
        client=client, chunk_words=chunk_words, force=force,
    )

    # ── Stage 2 ────────────────────────────────────────────────────────
    narration = ""
    if not force and os.path.exists(text_path):
        narration = open(text_path, encoding="utf-8").read().strip()
        if narration:
            logger.info("Ep %d: reusing %s (%d words)", ep_num,
                        os.path.basename(text_path), len(narration.split()))
    if not narration:
        if client is None:
            client = DeepSeekClient(api_key=resolve_api_key())
        narration = narrate_deep(arc, ep_num, title, client)
        os.makedirs(ep_dir, exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(narration + "\n")

    words = len(narration.split())
    logger.info("Ep %d: narration %d words (~%.0f min at 1x) — target zone %d-%d",
                ep_num, words, words / 175.0, _TARGET_MIN_WORDS, _TARGET_MAX_WORDS)
    if words < _TARGET_MIN_WORDS:
        logger.info("Ep %d: came in under the zone — accepted, volume is not padded",
                    ep_num)
    elif words > _TARGET_MAX_WORDS:
        logger.warning("Ep %d: %d words is over the %d-word ceiling",
                       ep_num, words, _TARGET_MAX_WORDS)

    mp3 = ""
    if skip_tts:
        logger.info("Ep %d: --skip-tts — no render", ep_num)
    else:
        mp3 = synthesize_narration(narration, mp3_path)

    return {
        "episode": ep_num, "mp3": mp3, "arc": arc_path,
        "narration": text_path, "words": words, "skipped": False,
        "chunks": len(arc.get("chunks", [])),
        "movements": len(arc.get("movements", [])),
    }
