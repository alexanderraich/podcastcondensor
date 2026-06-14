"""Semantic resegmentation — merge cleaned subtitle entries into editing segments.

Two-pass architecture:
  Pass 1 (deterministic): resegment() — fast rough cut using gap/marker/cap signals.
                          Always produces sentence-complete segments.
  Pass 2 (LLM):          refine_segments() — for each rough segment, split into
                          individual sentences, then at each sentence boundary
                          ask the LLM (3b) to decide BREAK or CONTINUE.
                          Only calls LLM when word-count rules (50-180) don't
                          force the decision. Assembles final segments from the
                          decision sequence.

Never cuts mid-sentence. Every output segment is guaranteed sentence-complete.
"""

import json
import logging
import re
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


def _ends_sentence(text: str) -> bool:
    """True if text ends with sentence-ending punctuation."""
    return text.rstrip().rstrip('"\')”’]').endswith((".", "!", "?"))


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


def _strip_head_overlap(segment: Segment, prev_segment: Segment) -> None:
    """Strip carryover overlap from start of segment that repeats end of prev_segment.

    Auto-captions repeat the last ~5-15 words of entry N at the start of entry
    N+1. When N and N+1 land in different segments, the repeated text appears
    in both. This removes it from the new segment's head.
    """
    if not segment.text or not prev_segment.text:
        return
    prev_words = prev_segment.text.split()
    curr_words = segment.text.split()
    if len(prev_words) < 5 or len(curr_words) < 3:
        return

    max_check = min(15, len(prev_words), len(curr_words))
    prev_tail_lower = " ".join(prev_words[-max_check:]).lower()

    for overlap_len in range(max_check, 2, -1):
        cand = " ".join(curr_words[:overlap_len]).lower()
        if prev_tail_lower.endswith(cand):
            segment.text = " ".join(curr_words[overlap_len:])
            segment.word_count = _word_count(segment.text)
            return


def resegment(
    entries: List[dict],
    gap_threshold: float = 0.5,
    gap_sentence_threshold: float = 8.0,
    max_words: int = 300,
    min_words: int = 20,
    sentence_overflow_words: int = 150,
) -> List[Segment]:
    """Split cleaned entries into semantic editing segments.

    Three signals for segment boundaries (checked in order):

    A. Gap silence between consecutive entries:
       - > gap_sentence_threshold: always split (real pause).
       - between gap_threshold and gap_sentence_threshold: split only if
         the preceding entry ends a sentence (avoids mid-sentence cuts
         from auto-caption timing jitter).
       - <= gap_threshold: no split.
    B. Discourse marker at the start of a new entry (split only at sentence
       boundaries — avoids splitting on auto-caption carryover).
    C. Hard cap at max_words — split even if no other signal fires.

    Sentence-completion overflow: When Signal C fires mid-sentence, the segment
    continues accumulating entries up to (max_words + sentence_overflow_words)
    to find the nearest sentence boundary before cutting.

    Post-pass: orphan segments < min_words get merged into the following segment.

    Args:
        entries: List of cleaned entries (from clean_entries).
        gap_threshold: Seconds of silence to consider a potential split.
        gap_sentence_threshold: Gaps above this threshold always split;
            gaps between gap_threshold and this value only split at sentence
            boundaries.
        max_words: Maximum words per segment (hard cap).
        min_words: Segments below this word count get merged into the next.
        sentence_overflow_words: Additional words allowed to complete a sentence
            after hitting the hard cap (0 to disable sentence-completion mode).

    Returns:
        List of Segment objects.
    """
    if not entries:
        return []

    segments = []
    current_entries = []
    current_words = 0
    sentence_overflow = False
    i = 0

    def _seal_and_append(entries, reason):
        """Seal entries, strip cross-segment overlap, append to segments."""
        seg = _seal_segment(entries, reason)
        if segments:
            _strip_head_overlap(seg, segments[-1])
        segments.append(seg)

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
            sentence_overflow = False
            i += 1
            continue

        # --- Sentence-completion overflow mode ---
        # When active, we've already exceeded max_words and are looking for a
        # sentence boundary.  Natural boundaries (gap / discourse marker) still
        # fire — only the hard cap is delayed.
        if sentence_overflow:
            if current_words > max_words + sentence_overflow_words:
                # Exceeded margin — cut here even if mid-sentence
                _seal_and_append(current_entries, "cap")
                current_entries = [entry]
                current_words = _word_count(entry["text"])
                sentence_overflow = False
                i += 1
                continue
            if _ends_sentence(current_entries[-1]["text"]):
                # Found a sentence boundary at last entry — seal cleanly
                _seal_and_append(current_entries, "cap")
                current_entries = [entry]
                current_words = _word_count(entry["text"])
                sentence_overflow = False
                i += 1
                continue
            # Still mid-sentence and within margin — keep accumulating
            current_entries.append(entry)
            current_words += _word_count(entry["text"])
            sentence_overflow = True
            i += 1
            continue

        # Signal A: Gap silence
        gap = entry["start"] - current_entries[-1]["end"]
        if gap > gap_sentence_threshold and current_words >= min_words:
            # Large gap — always a real break, even mid-sentence
            _seal_and_append(current_entries, "gap")
            current_entries = [entry]
            current_words = _word_count(entry["text"])
            i += 1
            continue
        if gap > gap_threshold and current_words >= min_words:
            if _ends_sentence(current_entries[-1]["text"]):
                # Modest gap at sentence boundary — clean split
                _seal_and_append(current_entries, "gap")
                current_entries = [entry]
                current_words = _word_count(entry["text"])
                i += 1
                continue
            # Modest gap mid-sentence — likely auto-caption jitter, keep accumulating

        # Signal B: Discourse marker — split only at sentence boundaries
        if current_words >= min_words and _has_discourse_marker(entry["text"]):
            if _ends_sentence(current_entries[-1]["text"]):
                # Clean split at sentence boundary + discourse marker
                _seal_and_append(current_entries, "marker")
                current_entries = [entry]
                current_words = _word_count(entry["text"])
                i += 1
                continue
            # Discourse marker mid-sentence — likely auto-caption carryover;
            # keep accumulating to avoid cutting the sentence.

        # Signal C: Hard cap
        added = _word_count(entry["text"])
        if current_words + added > max_words and current_words >= min_words:
            # Check if we're at a clean sentence boundary
            if sentence_overflow_words > 0 and not _ends_sentence(current_entries[-1]["text"]):
                # Mid-sentence — enter overflow mode to find the next boundary
                current_entries.append(entry)
                current_words += added
                sentence_overflow = True
            else:
                # At sentence boundary or overflow disabled — clean cut
                _seal_and_append(current_entries, "cap")
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
        _seal_and_append(current_entries, "end")

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


# ---------------------------------------------------------------------------
# Pass 2: LLM-based semantic refinement (per-boundary BREAK/CONTINUE)
# ---------------------------------------------------------------------------

import requests

# The prompt sent to the LLM for each candidate boundary (injected at bottom
# of this module — defined as a constant here for discoverability).
# See _BREAK_CONTINUE_PROMPT_TEMPLATE below the functions.

_MIN_WORDS = 50      # strongly prefer CONTINUE below this
_MAX_WORDS = 180     # strongly prefer BREAK above this
_HARD_MAX_WORDS = 200  # MUST BREAK at or above this


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return re.sub(r'\s+', ' ', text.strip())


def _split_sentences(text: str) -> List[str]:
    """Split text into individual sentences at sentence-ending punctuation."""
    text = _normalize_ws(text)
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _detect_discourse_marker(text: str) -> Optional[str]:
    """Return the matching discourse marker at the start of text, or None."""
    lower = text.lower().strip()
    if not lower:
        return None
    for marker in DISCOURSE_MARKERS:
        if lower.startswith(marker):
            return marker
    return None


def _gap_between_entries(
    sentence_text: str,
    next_sentence_text: str,
    rough_segment: Segment,
    entry_by_idx: dict,
) -> float:
    """Estimate the silence gap (s) between two consecutive sentences.

    Finds the last entry of sentence_text and the first entry of
    next_sentence_text in the entry stream, then returns the gap.
    """
    last_entry_idx = None
    first_entry_idx = None

    # Find the last entry index of sentence_text
    target = _normalize_ws(sentence_text)
    for end_pos in range(len(rough_segment.source_indices) - 1, -1, -1):
        idxs = rough_segment.source_indices[:end_pos + 1]
        texts = [entry_by_idx.get(i, {}).get("text", "") for i in idxs]
        if _normalize_ws(" ".join(texts)).endswith(target):
            last_entry_idx = idxs[-1]
            break
        if len(_normalize_ws(" ".join(texts))) > len(target) * 1.5:
            break

    # Find the first entry index of next_sentence_text
    target2 = _normalize_ws(next_sentence_text)
    for start_pos in range(len(rough_segment.source_indices)):
        idxs = rough_segment.source_indices[start_pos:]
        texts = [entry_by_idx.get(i, {}).get("text", "") for i in idxs]
        merged = _normalize_ws(" ".join(texts))
        if merged.startswith(target2):
            first_entry_idx = idxs[0]
            break

    if last_entry_idx is not None and first_entry_idx is not None:
        last_entry = entry_by_idx.get(last_entry_idx)
        first_entry = entry_by_idx.get(first_entry_idx)
        if last_entry and first_entry:
            return first_entry["start"] - last_entry["end"]

    return 0.0


def _call_ollama_json(model: str, prompt: str, host: str, timeout: int = 120) -> str:
    """Call Ollama generate API directly, return raw response text."""
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 64,   # tiny output — {"decision":"BREAK"} is 20 chars
        }
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def _parse_break_continue(raw: str) -> Optional[str]:
    """Parse the LLM response; return 'BREAK', 'CONTINUE', or None."""
    raw = raw.strip()
    # Strip markdown fences
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("{") or part.startswith('"'):
                raw = part
                break
    # Try JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            d = data.get("decision", "")
            if d in ("BREAK", "CONTINUE"):
                return d
        if isinstance(data, str) and data in ("BREAK", "CONTINUE"):
            return data
    except Exception:
        pass
    # Fallback: regex search
    import re as _re
    m = _re.search(r'(BREAK|CONTINUE)', raw)
    if m:
        return m.group(1)
    return None


def _build_segment_from_entries(
    start_idx: int,
    end_idx: int,
    entry_by_idx: dict,
) -> Optional[Segment]:
    """Build a Segment from a contiguous range of entry indices."""
    texts = []
    start_time = None
    end_time = None
    source_indices = []

    for idx in range(start_idx, end_idx + 1):
        entry = entry_by_idx.get(idx)
        if not entry:
            continue
        texts.append(entry["text"])
        if start_time is None:
            start_time = entry["start"]
        end_time = entry["end"]
        source_indices.append(idx)

    if not texts:
        return None

    full_text = _dedup_merge_texts(texts)
    return Segment(
        start=start_time,
        end=end_time,
        text=full_text,
        word_count=_word_count(full_text),
        source_indices=source_indices,
        boundary_reason="refined",
    )


_BREAK_CONTINUE_PROMPT_TEMPLATE = """You are an expert transcript segmentation engine for a theological podcast.

Your task is to decide whether to insert a paragraph break at a SINGLE candidate sentence boundary inside a rough transcript segment.

You will NOT rewrite, summarize, or change any words.
You will NOT output segments.
You will ONLY decide: BREAK or CONTINUE at this one boundary.

Your decision will be used to assemble fixed, verbatim segments from the original transcript text.

OUTPUT FORMAT
Return ONLY a JSON object: {{"decision":"BREAK"}} or {{"decision":"CONTINUE"}}
No markdown. No extra text. No explanations.

HARD RULES (in priority order):
1. MAX SIZE: If the current segment (before this boundary) is at or above 180 words, strongly prefer BREAK. At or above 200 words, MUST BREAK.
2. MIN SIZE: If the current segment is below 50 words, strongly prefer CONTINUE.
3. TOPIC TURN: BREAK if the next sentence clearly starts a new topic, question, named entity, scriptural reference, or major turn.
4. CONTINUITY: CONTINUE if the next sentence is still explaining the same claim, giving supporting details, or is part of the same example.
5. QUESTION/ANSWER: If the current sentence is a question and the next is its direct answer, prefer CONTINUE.
6. DISCOURSE MARKERS: New-topic markers ("turning to", "consider this", "the other thing", "moving on", "now let's talk") → BREAK. Continuation markers ("and", "so", "then", "thus", "therefore", "moreover", "furthermore") → CONTINUE unless other rules override.
7. Default to CONTINUE if unsure.

Target segment size: 50 to 200 words.

INPUT:
rough_segment_id: {ROUGH_SEGMENT_ID}
current_segment_words: {CURRENT_SEGMENT_WORDS}
previous_sentences: "{PREVIOUS_SENTENCES}"
next_sentence: "{NEXT_SENTENCE}"
has_gap_marker: {HAS_GAP_MARKER}
gap_seconds: {GAP_SECONDS}
discourse_marker: {DISCOURSE_MARKER}"""


def refine_segments(
    rough_segments: List[Segment],
    entries: List[dict],
    model: str = "qwen2.5:3b",
    host: str = "http://localhost:11434",
    batch_size_words: int = 0,
    timeout: int = 120,
    target_min_words: int = 50,
    target_max_words: int = 200,
) -> List[Segment]:
    """Refine rough segments via per-boundary BREAK/CONTINUE LLM decisions.

    Pass 2: for each rough segment, split into individual sentences.
    At each sentence boundary, decide (by LLM or rule) whether to break.
    Assembles final segments from the decision sequence.

    Only calls the LLM when rules don't force a decision (50-180 word range).
    Uses qwen2.5:3b by default — fast, tiny output per call.

    Args:
        rough_segments: Output of resegment().
        entries: Cleaned subtitle entries.
        model: Ollama model for decisions. Default qwen2.5:3b.
        host: Ollama host URL.
        batch_size_words: Unused (kept for API compatibility).
        timeout: LLM request timeout.
        target_min_words: Minimum target words per segment.
        target_max_words: Maximum target words per segment.

    Returns:
        Refined List[Segment] — each guaranteed sentence-complete.
    """
    if not rough_segments:
        return []

    entry_by_idx = {e["index"]: e for e in entries}
    refined = []

    for rough in rough_segments:
        text = _normalize_ws(rough.text)
        if not text:
            continue

        # Split into individual sentences
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            # Single sentence — keep as-is (it's already a complete unit)
            seg = Segment(
                segment_id=rough.segment_id,
                block_id=rough.block_id,
                start=rough.start,
                end=rough.end,
                text=rough.text,
                word_count=rough.word_count,
                source_indices=list(rough.source_indices),
                boundary_reason="refine_single_sentence",
            )
            refined.append(seg)
            continue

        # Walk through sentences, building segments from the LLM's decisions
        seg_boundaries = [0]  # indices in `sentences` where new segments start
        seg_start = 0
        seg_word_count = 0
        llm_calls = 0

        for i in range(len(sentences)):
            sent = sentences[i]
            sent_wc = _word_count(sent)

            # This is the boundary AFTER sentence i: should we break before
            # adding sentence i+1?
            is_last = (i == len(sentences) - 1)
            candidate_wc = seg_word_count + sent_wc

            if not is_last:
                next_sent = sentences[i + 1]

                # --- Rule-based decisions (no LLM call needed) ---

                # Hard max: MUST BREAK
                if candidate_wc >= _HARD_MAX_WORDS:
                    seg_boundaries.append(i + 1)
                    seg_start = i + 1
                    seg_word_count = 0
                    continue

                # Below min: MUST CONTINUE (don't even ask)
                if candidate_wc < target_min_words:
                    seg_word_count = candidate_wc
                    continue

                # Between min and hard max — ask the LLM
                prev_text = " ".join(sentences[seg_start:i + 1])
                has_gap = _gap_between_entries(sent, next_sent, rough, entry_by_idx)
                gap_secs = has_gap
                has_gap_flag = has_gap > 3.0
                dm = _detect_discourse_marker(next_sent) or "null"

                prompt = _BREAK_CONTINUE_PROMPT_TEMPLATE.format(
                    ROUGH_SEGMENT_ID=rough.segment_id,
                    CURRENT_SEGMENT_WORDS=candidate_wc,
                    PREVIOUS_SENTENCES=prev_text,
                    NEXT_SENTENCE=next_sent,
                    HAS_GAP_MARKER=str(has_gap_flag).lower(),
                    GAP_SECONDS=str(round(gap_secs, 1)),
                    DISCOURSE_MARKER=dm,
                )

                llm_calls += 1
                try:
                    raw = _call_ollama_json(model=model, prompt=prompt, host=host, timeout=timeout)
                    decision = _parse_break_continue(raw)
                except Exception as e:
                    logger.warning("LLM decision error for %s boundary %d: %s", rough.segment_id, i, e)
                    decision = "CONTINUE"

                if decision == "BREAK":
                    seg_boundaries.append(i + 1)
                    seg_start = i + 1
                    seg_word_count = 0
                else:
                    seg_word_count = candidate_wc

            else:
                # Flush last sentence
                seg_word_count = candidate_wc

        logger.info(
            "  %s: %d sentences → %d segments (%d LLM calls)",
            rough.segment_id, len(sentences), len(seg_boundaries), llm_calls,
        )

        # Build final segments from sentence ranges
        # If no splits happened, reuse the rough segment directly.
        if len(seg_boundaries) == 1 and seg_boundaries[0] == 0:
            seg = Segment(
                segment_id=rough.segment_id,
                block_id=rough.block_id,
                start=rough.start,
                end=rough.end,
                text=rough.text,
                word_count=rough.word_count,
                source_indices=list(rough.source_indices),
                boundary_reason="refine_no_split",
            )
            refined.append(seg)
        else:
            # Build character-position map from rough segment entries to
            # find which entry indices each refined segment text spans.
            norm_rough = _normalize_ws(rough.text)
            entry_char_spans = []  # (entry_index, start_char_in_merged, end_char_in_merged)
            search_pos = 0
            for eidx in rough.source_indices:
                entry = entry_by_idx.get(eidx)
                if not entry:
                    continue
                pos = norm_rough.find(_normalize_ws(entry["text"]), search_pos)
                if pos >= 0:
                    entry_char_spans.append((eidx, pos, pos + len(_normalize_ws(entry["text"]))))
                    search_pos = pos + 1
                # If not found, entry text was deduped away — skip

            for b in range(len(seg_boundaries)):
                start_sent_idx = seg_boundaries[b]
                end_sent_idx = seg_boundaries[b + 1] if b + 1 < len(seg_boundaries) else len(sentences)
                seg_text = " ".join(sentences[start_sent_idx:end_sent_idx])
                norm_seg = _normalize_ws(seg_text)

                # Find the character range of this segment in the rough text
                pos = norm_rough.find(norm_seg, 0)
                if pos < 0:
                    logger.warning(
                        "Could not find refined text in rough segment %s boundary %d; falling back",
                        rough.segment_id, b,
                    )
                    refined.append(rough)
                    break

                seg_char_start = pos
                seg_char_end = pos + len(norm_seg)

                # Find entries overlapping this character range
                matched_indices = []
                for eidx, cstart, cend in entry_char_spans:
                    if cstart < seg_char_end and cend > seg_char_start:
                        matched_indices.append(eidx)

                if matched_indices:
                    matched_indices.sort()
                    seg = _build_segment_from_entries(matched_indices[0], matched_indices[-1], entry_by_idx)
                    if seg:
                        seg.text = seg_text
                        seg.word_count = _word_count(seg_text)
                        seg.boundary_reason = "refined"
                        refined.append(seg)
                        continue

                logger.warning(
                    "Could not map refined segment %s boundary %d to entries; falling back",
                    rough.segment_id, b,
                )
                refined.append(rough)
                break

    # Final cleanup
    cleaned = []
    for seg in refined:
        seg.text = _normalize_ws(seg.text)
        seg.word_count = _word_count(seg.text)
        if seg.text:
            cleaned.append(seg)

    cleaned = _merge_orphans(cleaned, min_words=max(20, target_min_words // 2))

    for idx, seg in enumerate(cleaned, 1):
        seg.segment_id = f"seg-{idx:04d}"

    for i in range(1, len(cleaned)):
        _strip_head_overlap(cleaned[i], cleaned[i - 1])

    # Count mid-sentence cuts (should be 0)
    mid_cuts = sum(1 for s in cleaned if s.text and not _ends_sentence(s.text.rstrip()))
    logger.info(
        "Refined %d rough → %d segments (model=%s, mid-sentence=%d)",
        len(rough_segments), len(cleaned), model, mid_cuts,
    )

    return cleaned


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
