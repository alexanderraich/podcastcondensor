"""Semantic resegmentation — merge cleaned subtitle entries into editing segments.

Deterministic pass. No LLM calls.
Uses three signals: gap silence, discourse markers, hard word cap.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Discourse markers that indicate a new rhetorical move.
# Must be lowercase, checked at the start of an entry's text.
DISCOURSE_MARKERS = [
    "so ", "now ", "but ", "anyway ", "alright ", "okay ",
    "all right", "right,", "let's talk", "let's take",
    "the point is", "the question is", "moving on",
    "what about", "how about", "next ", "finally",
    "for example", "for instance", "in other words",
    "that is", "i mean", "specifically", "so the",
    "which brings us", "going back to", "remember that",
    "of course,", "now,", "well,", "and so", "so then",
    "so that's", "so we've", "so what", "so here's",
    "turning to", "consider this", "there's also",
    "the other thing", "another way", "in addition",
    "on the other hand", "having said that",
]


def _has_discourse_marker(text: str) -> bool:
    """True if text starts with a known discourse marker."""
    lower = text.lower().strip()
    if not lower:
        return False
    for marker in DISCOURSE_MARKERS:
        if lower.startswith(marker):
            return True
    return False


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


@dataclass
class Segment:
    """A semantic editing unit — the atomic unit for keep/drop decisions."""

    segment_id: str = ""
    block_id: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    word_count: int = 0
    source_indices: List[int] = field(default_factory=list)
    boundary_reason: str = ""  # "init" | "gap" | "marker" | "cap" | "end" | "merged_orphan"

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "block_id": self.block_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "word_count": self.word_count,
            "source_indices": self.source_indices,
            "boundary_reason": self.boundary_reason,
        }


def _dedup_merge_texts(texts: list) -> str:
    """Merge consecutive entry texts, removing carryover overlap at boundaries.

    Auto-captions produce entries where entry N+1 repeats the last ~5-15 words
    of entry N before continuing with new content. This function detects and
    removes those boundary overlaps so the final text reads as natural speech.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    result = texts[0]

    for i in range(1, len(texts)):
        current = texts[i].strip()
        if not current:
            continue

        # Get last ~20 words of accumulated text
        result_words = result.split()
        suffix = " ".join(result_words[-20:]).lower()

        # Find the longest overlap between suffix of result and prefix of current
        current_lower = current.lower()
        best_overlap_len = 0

        # Try overlap lengths from longest to shortest
        for ov in range(min(20, len(result_words), len(current.split())), 2, -1):
            # Candidate suffix from result
            cand = " ".join(result_words[-ov:]).lower()
            # Check if current text starts with this suffix
            if current_lower.startswith(cand):
                best_overlap_len = ov
                break

        if best_overlap_len > 0:
            # Strip the overlapped prefix from current text
            overlap_words_count = best_overlap_len
            overlap_str = " ".join(result_words[-overlap_words_count:])
            # The current text may start with slightly different casing/punctuation
            # Find where the overlap ends in the current text
            trimmed = current[len(overlap_str):].strip()
            if trimmed:
                result += " " + trimmed
            # If trimmed is empty, the current entry added nothing new — skip it
        else:
            result += " " + current

    return result


def _seal_segment(
    entries: list,
    reason: str,
) -> Segment:
    """Build a Segment from a list of cleaned entries with boundary dedup."""
    texts = [e["text"] for e in entries]
    full_text = _dedup_merge_texts(texts)
    return Segment(
        start=entries[0]["start"],
        end=entries[-1]["end"],
        text=full_text,
        word_count=_word_count(full_text),
        source_indices=[e["index"] for e in entries],
        boundary_reason=reason,
    )


def resegment(
    entries: List[dict],
    gap_threshold: float = 0.5,
    max_words: int = 300,
    min_words: int = 20,
) -> List[Segment]:
    """Split cleaned entries into semantic editing segments.

    Three signals for segment boundaries (checked in order):

    A. Gap silence > gap_threshold seconds between consecutive entries.
    B. Discourse marker at the start of a new entry (after segment has content).
    C. Hard cap at max_words — split even if no other signal fires.

    Post-pass: orphan segments < min_words get merged into the following segment.

    Args:
        entries: List of cleaned entries (from clean_entries).
        gap_threshold: Seconds of silence between entries to force a split.
        max_words: Maximum words per segment (hard cap).
        min_words: Segments below this word count get merged into the next.

    Returns:
        List of Segment objects.
    """
    if not entries:
        return []

    segments = []
    current_entries = []
    current_words = 0
    i = 0

    while i < len(entries):
        entry = entries[i]

        # Skip non-speech entries for segmentation boundaries,
        # but preserve their timing
        if entry.get("type") == "music":
            # Append music markers into current segment if there is one
            if current_entries:
                current_entries.append(entry)
                current_words += _word_count(entry["text"])
            # If no current segment, skip music entirely
            i += 1
            continue

        # Initialize first segment
        if not current_entries:
            current_entries = [entry]
            current_words = _word_count(entry["text"])
            i += 1
            continue

        # Signal A: Gap > threshold
        gap = entry["start"] - current_entries[-1]["end"]
        if gap > gap_threshold and current_words >= min_words:
            segments.append(_seal_segment(current_entries, "gap"))
            current_entries = [entry]
            current_words = _word_count(entry["text"])
            i += 1
            continue

        # Signal B: Discourse marker (only split if we have meaningful content)
        if current_words >= min_words and _has_discourse_marker(entry["text"]):
            segments.append(_seal_segment(current_entries, "marker"))
            current_entries = [entry]
            current_words = _word_count(entry["text"])
            i += 1
            continue

        # Signal C: Hard cap
        added = _word_count(entry["text"])
        if current_words + added > max_words and current_words >= min_words:
            segments.append(_seal_segment(current_entries, "cap"))
            current_entries = [entry]
            current_words = added
            i += 1
            continue

        # Default: append to current segment
        current_entries.append(entry)
        current_words += added
        i += 1

    # Flush last segment
    if current_entries:
        segments.append(_seal_segment(current_entries, "end"))

    # Post-pass: merge orphans
    segments = _merge_orphans(segments, min_words)

    # Assign IDs
    for idx, seg in enumerate(segments, 1):
        seg.segment_id = f"seg-{idx:04d}"

    logger.info(
        "Resegmented %d entries into %d segments "
        "(gap=%.2fs, max_words=%d, min_words=%d)",
        len(entries), len(segments),
        gap_threshold, max_words, min_words,
    )

    return segments


def _merge_orphans(
    segments: List[Segment],
    min_words: int,
) -> List[Segment]:
    """Merge segments below min_words into the following segment."""
    if len(segments) <= 1:
        return segments

    result = []
    i = 0
    while i < len(segments):
        if (
            i < len(segments) - 1
            and segments[i].word_count < min_words
        ):
            # Merge into next
            nxt = segments[i + 1]
            nxt.text = segments[i].text + " " + nxt.text
            nxt.start = segments[i].start
            nxt.source_indices = segments[i].source_indices + nxt.source_indices
            nxt.word_count += segments[i].word_count
            nxt.boundary_reason = "merged_orphan"
            i += 2
        else:
            result.append(segments[i])
            i += 1

    # Count merges
    merges = len(segments) - len(result)
    if merges:
        logger.info("Merged %d orphan segments", merges)

    return result


def resolve_block_ids(
    segments: List[Segment],
    old_chunks: List[dict],
    chunk_to_block: dict,
) -> List[Segment]:
    """Assign each segment a block_id from the existing chunk_to_block map.

    Uses temporal overlap weighting: the block_id with the most
    overlapping time wins.
    """
    for seg in segments:
        overlap_scores = {}
        for chunk in old_chunks:
            overlap_start = max(seg.start, chunk["start"])
            overlap_end = min(seg.end, chunk["end"])
            if overlap_end > overlap_start:
                block_id = chunk_to_block.get(chunk["uid"], 0)
                overlap_secs = overlap_end - overlap_start
                overlap_scores[block_id] = (
                    overlap_scores.get(block_id, 0) + overlap_secs
                )
        if overlap_scores:
            seg.block_id = max(overlap_scores, key=overlap_scores.get)
        else:
            seg.block_id = 0

    return segments
