"""Summary document builder — full per-episode digest of the corpus.

``build-summary-doc`` produces ``output/summaries_all.txt``: a per-episode
digest of the whole corpus from the structured knowledge already extracted in
each episode's ``global_state.json`` — the summary plus every concept, claim,
entity, scriptural link and glossary term — one ``# Episode NNN — <title>``
header per episode acting as a separation marker. Each structured item is
annotated with its ``segments`` timestamps (``(ep 1, 28:24–29:55)``) so any
claim/concept can be traced back to the audio/SRT. Deterministic, offline,
0 LLM calls.

Why this exists: the thematic cut was abandoned (mid-sentence cuts, confusing
segment adjacency), but the artefacts are reusable. The user feeds the corpus
to a downstream LLM for TTS narration of a condensed overview. The full digest
is ~110k words (~143k tokens) — sized for a large-context LLM, with no
re-running of the per-episode extraction (the summaries + structured knowledge
were already generated).

Inputs (all on disk, per the "on-disk scan first" convention):
- ``output/ep-NNN/global_state.json`` (per-episode extraction)
- ``output/ep-NNN/<video_id>.mp3`` (ID3 ``title`` tag = episode title)

The summaries are narrative paraphrase (per ``prompts/global_state.txt``), not
verbatim quotes — verified across the corpus, none contain quotation marks.
"""

import glob
import json
import logging
import os
import subprocess
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# Q&A / "Pantheon & Pandemonium Live Q&A" specials — deliberately without
# global_state.json per the universe-state convention (CLAUDE.md). Of these,
# only ep-18 has an SRT on disk (transcribed before Q&A skipping was
# standard); excluding the full documented set keeps this consistent with
# build-universe if other Q&A SRTs ever appear.
_QA_EPISODES = frozenset({
    18, 66, 67, 74, 78, 80, 89, 90, 98,
    104, 106, 111, 117, 121, 122, 126, 134, 135, 141,
})

_SUMMARY_FILENAME = "summaries_all.txt"

# (global_state key, ## heading, per-item line formatter) — empty categories
# are omitted entirely so the digest stays prose-clean.
_STRUCTURED_SECTIONS: Tuple[Tuple[str, str, Callable[[dict], str]], ...] = (
    ("concepts", "Concepts", lambda i: _pair(i.get("title"), i.get("summary"))),
    ("claims", "Claims", lambda i: i.get("text", "").strip()),
    ("entities", "Entities", lambda i: _pair(
        f"{i.get('title', '').strip()}"
        f" ({i.get('category', '').strip()})" if i.get("category") else i.get("title", ""),
        i.get("summary"))),
    ("scriptural_links", "Scripture", lambda i: _pair(i.get("reference"), i.get("summary"))),
    ("glossary", "Glossary", lambda i: _pair(i.get("term"), i.get("definition"))),
)


def _pair(head: str, rest: str) -> str:
    """``head — rest``, or ``head`` alone / ``rest`` alone when one is empty."""
    head = (head or "").strip()
    rest = (rest or "").strip()
    if head and rest:
        return f"{head} — {rest}"
    return head or rest


def _ts(seconds: float) -> str:
    """Seconds → ``M:SS`` (or ``H:MM:SS`` for episodes past an hour)."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _segments_suffix(segments) -> str:
    """``(ep 1, 28:24–29:55; ep 2, 1:02:03–1:05:00)`` from a segments array.

    Each structured item carries a ``segments`` list (episode, start, end —
    seconds) locating where it is discussed, enabling trace-back to the
    audio/SRT. Returns "" when there are no usable segments.
    """
    if not segments:
        return ""
    parts = []
    for seg in segments:
        ep, start, end = seg.get("episode"), seg.get("start"), seg.get("end")
        if ep is None or start is None or end is None:
            continue
        parts.append(f"ep {ep}, {_ts(start)}–{_ts(end)}")
    return f"({'; '.join(parts)})" if parts else ""


def _default_output_root() -> str:
    """Repo output dir, matching Config's default (parent of src/)."""
    return os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        "output",
    )


def _episode_dir_numbers(output_root: str) -> List[int]:
    """Sorted canonical episode numbers with an SRT on disk, excluding Q&A."""
    nums = []
    for srt in glob.glob(os.path.join(output_root, "ep-*/source_subtitles.srt")):
        ep_dir = os.path.basename(os.path.dirname(srt))
        try:
            num = int(ep_dir.split("-")[1])
        except (IndexError, ValueError):
            logger.warning("Skipping non-canonical episode dir: %s", ep_dir)
            continue
        if num in _QA_EPISODES:
            continue
        nums.append(num)
    return sorted(nums)


def _episode_title(output_root: str, ep_num: int) -> str:
    """Episode title from the MP3 ID3 ``title`` tag; 'Episode NNN' fallback.

    MP3 filenames are the video ID and the ID3 title is the YouTube title,
    e.g. "Lord of Spirits: Angels and Demons I - Introducing Lord of Spirits
    [Ep. 1]". ffprobe reads it offline — no YouTube call.
    """
    ep_dir = f"ep-{ep_num:03d}"
    mp3s = glob.glob(os.path.join(output_root, ep_dir, "*.mp3"))
    if mp3s:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format_tags=title",
             "-of", "default=nw=1:nk=1", mp3s[0]],
            capture_output=True, text=True,
        )
        title = out.stdout.strip()
        if title:
            return title
        logger.warning("No ID3 title for %s; using 'Episode %d'", ep_dir, ep_num)
    else:
        logger.warning("No MP3 for %s; using 'Episode %d'", ep_dir, ep_num)
    return f"Episode {ep_num}"


def _episode_digest(
    output_root: str, ep_num: int, title: str, with_segments: bool = True
) -> str:
    """Full per-episode digest: summary + all structured knowledge.

    ``with_segments=True`` (default) appends each item's trace-back timestamps
    ``(ep N, M:SS–M:SS)``. ``False`` strips them — used for the narration
    input so trace-back metadata never leaks into spoken audio.

    Raises FileNotFoundError / RuntimeError (fail-loud, never a partial
    document) if the global state is missing or its summary is empty.
    """
    path = os.path.join(output_root, f"ep-{ep_num:03d}", "global_state.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global state not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", "").strip()
    if not summary:
        raise RuntimeError(f"Empty summary for episode {ep_num}: {path}")

    block = [f"# Episode {ep_num} — {title}", "", summary]
    for key, heading, fmt in _STRUCTURED_SECTIONS:
        lines = []
        for item in data.get(key, []):
            base = fmt(item)
            if not base:
                continue
            suffix = _segments_suffix(item.get("segments")) if with_segments else ""
            lines.append(f"{base} {suffix}" if suffix else base)
        if lines:
            block.extend(("", f"## {heading}", *(f"- {l}" for l in lines)))
    return "\n".join(block)


def build_summary_doc(output_root: str = "", out_path: str = "") -> str:
    """Write the full per-episode digest to one text file.

    Each episode contributes ``# Episode N — <title>`` + summary + all
    structured knowledge (~110k words / ~143k tokens for the corpus — one
    large-context LLM pass). Deterministic, offline, 0 LLM calls.

    Returns the output path written.
    """
    root = output_root or _default_output_root()
    ep_nums = _episode_dir_numbers(root)
    if not ep_nums:
        raise RuntimeError(f"No episode SRTs found under {root!r}")

    parts = []
    for ep_num in ep_nums:
        logger.info("Episode %d: digest", ep_num)
        title = _episode_title(root, ep_num)
        parts.append(_episode_digest(root, ep_num, title))

    dest = os.path.abspath(out_path) if out_path else os.path.join(
        root, _SUMMARY_FILENAME
    )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(parts) + "\n")
    logger.info("Wrote %s (%d episode digests)", dest, len(ep_nums))
    return dest
