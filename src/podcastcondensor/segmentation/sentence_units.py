"""Deterministic sentence-unit extraction from cleaned subtitle entries.

The core intermediate representation for LLM-based segmentation.  The
transcript is split into ``SentenceUnit`` objects — guaranteed sentence-
complete, preserving exact text order, and mapped back to the original
subtitle entry indices.

Concatenating all ``SentenceUnit.text`` values in order must exactly
reproduce ``transcript_text``.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SentenceUnit:
    """A single sentence from the deduplicated transcript.

    Maps back to the subtitle entry span that contains this sentence.
    """

    sentence_id: int
    text: str
    start_entry_index: int
    end_entry_index: int
    start_time: float
    end_time: float
    word_count: int = 0

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text.strip())


def _split_sentences(text: str) -> List[str]:
    """Split text into individual sentences at sentence-ending punctuation."""
    text = _normalize_ws(text)
    if not text:
        return []
    # Split at sentence boundaries, keeping the delimiter attached to the
    # preceding sentence.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_sentence_units(
    entries: List[dict],
    transcript_text: str,
) -> List[SentenceUnit]:
    """Build deterministic sentence units from the deduplicated transcript.

    Steps:
      1. Split ``transcript_text`` into individual sentences.
      2. For each sentence, find its character position in the transcript.
      3. Build a character-to-entry map from the original entries to
         determine which entry range each sentence falls within.
      4. Assemble ``SentenceUnit`` objects.

    Args:
        entries: Cleaned subtitle entry dicts, each with keys:
                 index, start, end, text, type.
        transcript_text: Merged deduplicated full transcript text.

    Returns:
        List of ``SentenceUnit``, covering the entire transcript with
        no gaps.  May be empty if the transcript is empty.

    Raises:
        ValueError: If sentence extraction cannot preserve exact fidelity
                    (the caller should fall back to deterministic segmentation).
    """
    if not transcript_text.strip():
        return []

    sentences = _split_sentences(transcript_text)
    if not sentences:
        # Single block of text with no sentence breaks — make one unit
        norm = _normalize_ws(transcript_text)
        if norm:
            return [_make_fallback_unit(entries, norm)]
        return []

    # Build a character-position map: for each character index in the
    # transcript, record which entry index it came from.
    norm_transcript = _normalize_ws(transcript_text)
    char_to_entry: List[int] = []  # char_position -> entry index

    for entry in entries:
        entry_text = _normalize_ws(entry.get("text", ""))
        if not entry_text:
            continue
        # Find this entry's text in the normalized transcript
        pos = norm_transcript.find(entry_text, len(char_to_entry) if char_to_entry else 0)
        if pos < 0:
            # Entry text was deduped away or not found — skip gracefully
            continue
        # Pad with previous entry index until we reach this position
        while len(char_to_entry) < pos:
            char_to_entry.append(char_to_entry[-1] if char_to_entry else entry["index"])
        # Map each character of this entry
        for _ in range(len(entry_text)):
            char_to_entry.append(entry["index"])
        # Account for the space added by _normalize_ws between entries
        # (the normalizer adds a space between joined texts)
        if len(char_to_entry) < len(norm_transcript):
            char_to_entry.append(entry["index"])

    # Handle any trailing characters
    while len(char_to_entry) < len(norm_transcript):
        char_to_entry.append(char_to_entry[-1] if char_to_entry else 0)

    # If the map is empty or doesn't cover the transcript, fall back
    if len(char_to_entry) < len(norm_transcript):
        raise ValueError(
            f"Character-to-entry map covers {len(char_to_entry)}/{len(norm_transcript)} "
            f"chars — round-trip fidelity not guaranteed"
        )

    # Build sentence units by scanning through sentences
    units: List[SentenceUnit] = []
    search_pos = 0
    entry_by_index = {e["index"]: e for e in entries}

    for sid, sentence in enumerate(sentences, 1):
        norm_sent = _normalize_ws(sentence)
        if not norm_sent:
            continue

        # Find this sentence in the transcript starting from search_pos
        pos = norm_transcript.find(norm_sent, search_pos)
        if pos < 0:
            # Sentence not found — fidelity broken
            raise ValueError(
                f"Sentence {sid} not found in transcript at position {search_pos}: "
                f"{norm_sent[:80]}..."
            )

        sent_end = pos + len(norm_sent)

        # Determine entry index range from character map
        if pos < len(char_to_entry):
            start_entry = char_to_entry[pos]
        else:
            start_entry = char_to_entry[-1] if char_to_entry else 0

        end_pos = min(sent_end, len(char_to_entry) - 1)
        end_entry = char_to_entry[end_pos] if end_pos >= 0 else start_entry

        # Map to time boundaries
        start_entry_obj = entry_by_index.get(start_entry)
        end_entry_obj = entry_by_index.get(end_entry)
        start_time = start_entry_obj["start"] if start_entry_obj else 0.0
        end_time = end_entry_obj["end"] if end_entry_obj else 0.0

        units.append(SentenceUnit(
            sentence_id=sid,
            text=sentence,
            start_entry_index=start_entry,
            end_entry_index=end_entry,
            start_time=start_time,
            end_time=end_time,
        ))

        search_pos = sent_end

    # Verify round-trip: concatenated texts must equal transcript
    reconstructed = " ".join(u.text for u in units)
    if _normalize_ws(reconstructed) != _normalize_ws(transcript_text):
        raise ValueError(
            "Sentence unit reconstruction failed round-trip fidelity check"
        )

    logger.info(
        "Built %d sentence units from %d entries (%d chars transcript)",
        len(units), len(entries), len(transcript_text),
    )

    return units


def _make_fallback_unit(entries: List[dict], text: str) -> SentenceUnit:
    """Create a single SentenceUnit when the text has no sentence breaks."""
    if entries:
        return SentenceUnit(
            sentence_id=1,
            text=text,
            start_entry_index=entries[0]["index"],
            end_entry_index=entries[-1]["index"],
            start_time=entries[0]["start"],
            end_time=entries[-1]["end"],
        )
    return SentenceUnit(
        sentence_id=1,
        text=text,
        start_entry_index=0,
        end_entry_index=0,
        start_time=0.0,
        end_time=0.0,
    )


def build_transcript_from_entries(entries: List[dict]) -> str:
    """Build the deduplicated full transcript text from subtitle entries.

    Uses the same carryover dedup logic as the existing segmentation
    pipeline so the result is identical to what ``_dedup_merge_texts``
    would produce for all entries.
    """
    from podcastcondensor.dedup import _dedup_merge_texts

    texts = [e["text"] for e in entries]
    return _dedup_merge_texts(texts)
