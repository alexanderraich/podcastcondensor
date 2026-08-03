"""Subtitle parsing — normalize .srt and .vtt to clean, deduplicated entries."""

import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)
SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s*\n"
    r"((?:(?!\n\n|\n$).+\n?)*)",
    re.MULTILINE,
)
TIMESTAMP_LEAK_RE = re.compile(
    r"\d+\s+\d{2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{1,3}"
)


def _ts_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to seconds."""
    m = TIMESTAMP_RE.match(ts_str.strip())
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts_str}")
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if len(str(ms)) == 2:
        ms = ms * 10
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _normalize_vtt(text: str) -> str:
    """Remove WEBVTT header and metadata lines, return clean SRT-like blocks."""
    lines = text.split("\n")
    clean = []
    found_first = False
    for line in lines:
        if line.strip() == "":
            if found_first:
                clean.append("")
            continue
        if line.strip() == "WEBVTT":
            found_first = True
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line.strip()):
            line = line.replace(".", ",")
        clean.append(line)
    return "\n".join(clean)


def parse_srt_text(text: str) -> List[dict]:
    """Parse SRT text into list of raw entry dicts.

    Each entry: {index, start, end, text}
    Indices are sequential from the SRT file.
    """
    if text.strip().startswith("WEBVTT"):
        text = _normalize_vtt(text)

    entries = []
    for m in SRT_BLOCK_RE.finditer(text):
        seq = int(m.group(1))
        start = _ts_to_seconds(m.group(2))
        end = _ts_to_seconds(m.group(3))
        raw_text = m.group(4).strip().replace("\n", " ").replace("  ", " ")
        clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
        if clean_text:
            entries.append({
                "index": seq,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clean_text,
            })

    logger.info("Parsed %d raw entries from SRT", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Cleanup pass (applied after raw parsing, before chunking/segmentation)
# ---------------------------------------------------------------------------

def _strip_timestamp_leak(text: str) -> str:
    """Remove residual SRT timestamp + sequence numbers that leak into text fields."""
    return TIMESTAMP_LEAK_RE.sub("", text).strip()


def _classify_entry(text: str) -> str:
    """Classify a subtitle entry as 'speech', 'music', or 'noise'."""
    t = text.strip().lower()
    if not t:
        return "noise"
    if re.match(r"^\[music\]", t) or re.match(r"^♪", t):
        return "music"
    if len(t) < 3 or re.match(r"^[\s\-_*#]+$", t):
        return "noise"
    return "speech"


def _is_echo(entry: dict, prev_entry: dict) -> bool:
    """True if this entry is a <=50ms echo of the prior entry (auto-caption carryover)."""
    duration = entry["end"] - entry["start"]
    if duration > 0.05:
        return False
    prev_text = prev_entry.get("text", "").lower().strip()
    this_text = entry["text"].lower().strip()
    if not this_text or not prev_text:
        return False
    return this_text in prev_text or prev_text.endswith(this_text)


def _partial_dedup(entries: List[dict]) -> List[dict]:
    """Remove near-duplicate consecutive speech entries using word-overlap."""
    result = []
    for entry in entries:
        if entry["type"] != "speech":
            result.append(entry)
            continue
        if not result:
            result.append(entry)
            continue
        prev = result[-1]
        if prev["type"] != "speech":
            result.append(entry)
            continue

        prev_words = set(prev["text"].lower().split())
        curr_words = set(entry["text"].lower().split())
        if not prev_words or not curr_words:
            result.append(entry)
            continue

        # Measure how much of the current entry's content is NEW
        # (not already present in the previous entry)
        new_words = curr_words - prev_words
        novelty = len(new_words) / len(curr_words) if curr_words else 0

        # If current entry adds almost nothing new, it's a duplicate
        if novelty < 0.25:
            prev_dur = prev["end"] - prev["start"]
            curr_dur = entry["end"] - entry["start"]
            if curr_dur <= prev_dur:
                continue  # drop current (shorter or equal, adds nothing)
            # Current is longer but mostly repeats — keep the longer one
            result[-1] = entry
            continue

        result.append(entry)
    return result


def clean_entries(entries: List[dict], reindex: bool = True) -> List[dict]:
    """Clean raw subtitle entries: strip artifacts, dedup, remove echoes.

    When *reindex* is True (default, backward-compatible) entries get
    sequential 1‑based indices.  When False, original SRT cue numbers
    are preserved (with gaps where noise/echo entries were removed),
    matching what the LLM sees in the raw SRT file.

    Each returned entry:
      {index, start, end, text, type}
    where type is "speech", "music", or "noise".
    """
    if not entries:
        return []

    cleaned = []
    echo_removed = 0
    noise_removed = 0
    leak_fixes = 0

    for entry in entries:
        # Step 1: Strip timestamp leaks
        text = _strip_timestamp_leak(entry["text"])
        if text != entry["text"]:
            leak_fixes += 1

        if not text:
            noise_removed += 1
            continue

        entry["text"] = text
        entry["type"] = _classify_entry(text)

        # Step 2: Skip noise (non-speech, non-music)
        if entry["type"] == "noise":
            noise_removed += 1
            continue

        # Step 3: Remove echo entries
        if cleaned and _is_echo(entry, cleaned[-1]):
            echo_removed += 1
            continue

        cleaned.append(dict(entry))

    # Step 4: Partial dedup
    deduped = _partial_dedup(cleaned)

    logger.info(
        "Cleanup: %d leaks fixed, %d echoes removed, %d noise removed, "
        "%d deduped -> %d entries",
        leak_fixes, echo_removed, noise_removed,
        len(cleaned) - len(deduped), len(deduped),
    )

    # Re-index (unless caller wants original SRT indices preserved)
    if reindex:
        for i, entry in enumerate(deduped):
            entry["index"] = i + 1

    return deduped


def load_subtitles(filepath: str, reindex: bool = True) -> List[dict]:
    """Load subtitles from .srt or .vtt file, return CLEANED entries.

    When *reindex* is True (default), entries get sequential 1-based indices.
    When False, original SRT cue numbers are preserved (with gaps where
    noise/echo entries were removed), matching what the LLM sees in the raw
    SRT.

    Phase-2 callers (global state) should use the default reindex=True.
    Phase-3/4 callers (raw classifier + audio cutting) should use
    reindex=False so LLM decisions by cue number map to the actual entries.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    raw = parse_srt_text(text)
    cleaned = clean_entries(raw, reindex=reindex)
    logger.info("Loaded %s: %d entries after cleaning%s", filepath, len(cleaned),
                " (original indices)" if not reindex else "")
    return cleaned


# Words/phrases whisper frequently follows with a sentence-final period even
# though the thought continues. If a block's last entry ends with one of these
# (plus punctuation), the period is a whisper artifact and the block must NOT
# close there — it keeps absorbing until a real sentence end. False positives
# only lengthen a block; false negatives are the costly mid-sentence cuts.
_CONTINUATION_WORDS = frozenset(
    """and or but yet so then that because when while if which who whom whose
    where how why until unless though although since as than whether whenever
    wherever""".split()
)

_CONTINUATION_PHRASES = frozenset(
    """at the same time on the other hand and yet and so so that in the same
    way in the meantime on the one hand even though as well in order in other
    words that is""".split()
)

# Dangling prepositions — a sentence rarely ends on one in podcast speech, so
# a trailing period after one is a whisper artifact. (Short high-frequency
# words like "on"/"off"/"up"/"for" are excluded — they legitimately end
# sentences too often.)
_DANGLING_PREPOSITIONS = frozenset(
    """about across against among around behind below beneath beside between
    beyond during except from inside like near of onto opposite outside over
    past through toward under underneath upon within without""".split()
)


def _is_continuation_end(text: str) -> bool:
    """True if ``text`` (already stripped, may end with .!?) reads as a
    mid-thought continuation despite trailing sentence punctuation.

    Whisper frequently inserts sentence-final periods mid-thought (e.g.
    "...and yet at the same time.", "...And so."). Detect those by checking
    the trailing word / trailing phrase against continuation markers.
    """
    t = text.rstrip().rstrip(".!?").strip()
    if not t:
        return False
    words = t.lower().split()
    if words[-1] in _CONTINUATION_WORDS:
        return True
    if words[-1] in _DANGLING_PREPOSITIONS:
        return True
    for n in range(1, min(4, len(words)) + 1):
        if " ".join(words[-n:]) in _CONTINUATION_PHRASES:
            return True
    return False


def _is_leading_continuation(text: str) -> bool:
    """True if ``text`` starts a fresh block with a dangling preposition —
    a fragment that continues the PREVIOUS sentence.

    Whisper sometimes closes a sentence with a spurious period and then
    starts the continuation phrase as its own block (e.g. "...something other
    than Christianity." followed by "of Western European origin or connection
    or cultural tradition."). Such a block has no verb and depends on the
    prior sentence; a cut landing on it is a mid-thought cut. Mirrors
    ``_is_continuation_end`` but on the leading edge.

    Same philosophy as the trailing check: false positives only lengthen a
    block; false negatives are the costly mid-sentence cuts, so the set leans
    aggressive (and is restricted to dangling prepositions — coordinating
    conjunctions like "And"/"But"/"So" legitimately begin sentences).
    """
    t = text.strip().lstrip("\"'(-")
    if not t:
        return False
    words = t.lower().split()
    return bool(words) and words[0] in _DANGLING_PREPOSITIONS


def build_sentence_blocks(entries: List[dict]) -> List[dict]:
    """Merge SRT entries into sentence-complete blocks.

    Greedily absorbs consecutive entries until the merged text ends with
    sentence-ending punctuation (. ! ?). Each block is a guaranteed complete
    thought with clean timestamp boundaries.

    Returns list of::

        {"block_index": int, "start": float, "end": float,
         "text": str, "entry_indices": List[int]}

    Entries that end without sentence punctuation are absorbed into the
    next block. A trailing block that never reaches sentence punctuation
    is still returned as-is (don't drop content).
    """
    if not entries:
        return []

    blocks = []
    current_texts: List[str] = []
    current_entries: List[dict] = []

    for e in entries:
        if e.get("type") not in ("speech", None):
            continue
        text = e.get("text", "").strip()
        if not text:
            continue

        # Leading-fragment detection: when we are about to START a fresh block
        # with an entry that begins with a dangling preposition, that entry is
        # a fragment of the PREVIOUS sentence (whisper closed the sentence with
        # a spurious period, then continued it as a new block). Reopen the
        # previous block so the fragment AND the following real sentence stay
        # in one block — a block must never START on a fragment (which would
        # let a mid-sentence cut land clean on its end).
        if not current_entries and blocks and _is_leading_continuation(text):
            prev = blocks.pop()
            current_entries = list(prev.get("entries", []))
            current_texts = [ce["text"].strip() for ce in current_entries]

        current_texts.append(text)
        current_entries.append(e)

        # Close only at a REAL sentence end: terminal punctuation that is
        # neither a trailing continuation ("...and yet at the same time.")
        # nor a leading continuation ("of Western European origin...", which
        # whisper ended with a period even though the thought continues).
        if (
            text[-1] in (".", "!", "?")
            and not _is_continuation_end(text)
            and not _is_leading_continuation(text)
        ):
            blocks.append({
                "block_index": len(blocks) + 1,
                "start": current_entries[0]["start"],
                "end": current_entries[-1]["end"],
                "text": " ".join(current_texts),
                "entry_indices": [ce["index"] for ce in current_entries],
                "entries": current_entries,
            })
            current_texts = []
            current_entries = []

    # Flush remaining (trailing text that never hit sentence punctuation)
    if current_texts:
        blocks.append({
            "block_index": len(blocks) + 1,
            "start": current_entries[0]["start"],
            "end": current_entries[-1]["end"],
            "text": " ".join(current_texts),
            "entry_indices": [ce["index"] for ce in current_entries],
            "entries": current_entries,
        })

    logger.debug(
        "Sentence blocks: %d entries → %d blocks",
        len(entries), len(blocks),
    )
    return blocks
