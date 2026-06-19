"""DeepSeek cloud segmentation — two-pass (punctuate then segment)."""

import json
import logging
import os
import re
from typing import List, Optional

from podcastcondensor.segmentation.validation import (
    SegmentationValidator,
    validate_plan_coverage,
    SegmentationValidationError,
)
from podcastcondensor.segmentation.schemas import SegmentationPlan, SegmentationPlanItem
from podcastcondensor.dedup import _dedup_merge_texts

logger = logging.getLogger(__name__)

_PUNCTUATE_PROMPT = """You are a transcript cleanup engine.
Return json only.
Add sentence-ending punctuation (periods, question marks, exclamation marks)
and capitalisation to the given transcript text.
Rules:
- Add periods at sentence boundaries.
- Capitalise the first word of each sentence.
- Add question marks where it sounds like a question.
- Do NOT change, reword, reorder, or omit any words.
- Output the FULL punctuated text in the "text" field.
Return exactly: {"schema_version":1,"text":"Punctuated text here..."}"""

_SEGMENT_PROMPT = """You are a transcript segmentation engine.
Return json only.
You receive transcript units with unit_id/text pairs.
Group CONTIGUOUS unit ID ranges into segments where each segment covers
one complete thought, topic, or argumentative unit.
Rules:
- Every segment: contiguous unit ID range, cover full transcript, no gaps.
- Split at EVERY natural break: topic shift, completed thought, new example, question.
- If a long discussion covers multiple sub-topics, split it.
Return exactly:
{"schema_version":1,"segments":[{"segment_id":"seg-0001","start_unit_id":1,"end_unit_id":8,"boundary_reason":"completed thought"}]}"""


class DeepSeekSegmentation:
    """Segments a podcast transcript using DeepSeek.

    If entries have sentence punctuation: fast path (one call for entry ranges).
    If no punctuation: punctuation pass first, then split into sentences,
    then segment call on sentence ID ranges.
    """

    _PUNC_SAMPLE = 200
    _PUNC_THRESHOLD = 0.05

    def __init__(self, client, model="deepseek-chat", timeout=300, max_tokens=12000, retries=1, checkpoint_dir=None):
        self._client = client
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._retries = retries
        self._validator = SegmentationValidator(check_sentence_complete=False, check_round_trip=False)
        self._checkpoint_dir = checkpoint_dir

    def segment(self, entries: List[dict], transcript_text: str) -> List[dict]:
        if not entries:
            raise RuntimeError("No entries to segment")

        has_punc = self._detect_punctuation(entries)
        if has_punc:
            return self._segment_entries(entries)
        else:
            return self._segment_no_punc(entries, transcript_text)

    def _detect_punctuation(self, entries: List[dict]) -> bool:
        sample = entries[:self._PUNC_SAMPLE]
        n = sum(1 for e in sample if e.get("text", "").rstrip()[-1:] in (".", "!", "?"))
        ratio = n / len(sample) if sample else 0
        logger.info("Punctuation: %d/%d entries (%.0f%%)", n, len(sample), ratio * 100)
        return ratio >= self._PUNC_THRESHOLD

    # ---- Fast path: entries have punctuation ----

    def _segment_entries(self, entries: List[dict]) -> List[dict]:
        plan = self._call_segment([
            {"unit_id": e["index"], "text": e["text"]} for e in entries
        ], len(entries))
        return self._reconstruct_entry_plan(plan, entries)

    # ---- Slow path: no punctuation ----

    _PUNCTUATED_CHECKPOINT = "punctuated_text.json"

    def _segment_no_punc(self, entries: List[dict], transcript_text: str) -> List[dict]:
        logger.info("No punctuation — adding via DeepSeek (%d chars)...", len(transcript_text))

        # Checkpoint: save/load the expensive DeepSeek punctuation output
        checkpoint_path = None
        if self._checkpoint_dir:
            checkpoint_path = os.path.join(
                self._checkpoint_dir, self._PUNCTUATED_CHECKPOINT,
            )

        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            punctuated = ckpt["text"]
            logger.info("Loaded punctuated text from checkpoint (%d chars)", len(punctuated))
        else:
            punctuated = self._call_punctuate(transcript_text)
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"text": punctuated, "schema_version": 1}, f, ensure_ascii=False)
                logger.info("Saved punctuated text to checkpoint: %s", checkpoint_path)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', punctuated) if s.strip()]
        if len(sentences) <= 1:
            raise RuntimeError(f"Punctuation produced only {len(sentences)} sentence(s)")

        # Map sentences to entry ranges using text matching
        entry_ranges = self._map_sentences_to_entries_accurate(
            sentences, entries, transcript_text,
        )

        # Segment by sentence IDs
        sent_entries = [
            {"unit_id": i + 1, "text": sentences[i]}
            for i in range(len(sentences))
        ]
        plan = self._call_segment(sent_entries, len(sentences), sentence_ids=True)
        return self._reconstruct_sentence_plan(plan, sentences, entry_ranges, entries)

    # ---- API calls ----

    def _call_with_retry(self, prompt: str, max_tok: int) -> str:
        for attempt in range(self._retries + 1):
            try:
                return self._client.generate(
                    prompt=prompt, model=self._model,
                    timeout=self._timeout, temperature=0.1,
                    max_tokens=max_tok, force_json=True,
                )
            except Exception as e:
                if attempt < self._retries:
                    continue
                raise RuntimeError(f"DeepSeek failed after {self._retries+1} attempts: {e}")

    def _call_punctuate(self, text: str) -> str:
        payload = json.dumps({"transcript": text}, ensure_ascii=False)
        prompt = _PUNCTUATE_PROMPT.strip() + "\n\n" + payload
        max_tok = max(self._max_tokens, len(text))  # output ≈ input size in chars
        raw = self._call_with_retry(prompt, max_tok)

        text = raw.strip()
        if "```" in text:
            lines = text.split("\n")
            clean = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    clean.append(line)
            if clean:
                text = "\n".join(clean).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Punctuation: no JSON")
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            # DeepSeek sometimes drops the closing quote — try to repair
            last_brace = candidate.rstrip().rfind('}')
            if last_brace > 0 and candidate[last_brace - 1].strip() != '"':
                candidate = candidate[:last_brace] + '"' + candidate[last_brace:]
            try:
                data = json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                raise RuntimeError(f"Punctuation: JSON parse failed ({len(candidate)} chars)")
        result = data.get("text", "")
        if not result:
            raise RuntimeError("Punctuation: empty text")
        return result

    def _call_segment(self, units: list, total: int, sentence_ids=False) -> SegmentationPlan:
        id_key = "unit_id"
        payload = json.dumps({
            "sentences" if sentence_ids else "entries": units,
            "total_sentences" if sentence_ids else "total_entries": total,
        }, ensure_ascii=False, indent=2)
        prompt = _SEGMENT_PROMPT.strip() + "\n\n" + payload

        max_tok = max(self._max_tokens, total // 10)
        logger.info("Segmentation: %d units, %d chars", total, len(prompt))
        raw = self._call_with_retry(prompt, max_tok)

        text = raw.strip()
        if "```" in text:
            lines = text.split("\n")
            clean = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    clean.append(line)
            if clean:
                text = "\n".join(clean).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Segmentation: no JSON")
        data = json.loads(text[start:end + 1], strict=False)
        raw_segs = data.get("segments")
        if not isinstance(raw_segs, list) or not raw_segs:
            raise RuntimeError("Segmentation: no segments")

        items = []
        for i, item in enumerate(raw_segs):
            if not isinstance(item, dict):
                raise RuntimeError(f"Segment {i} not a dict")
            sid = str(item.get("segment_id", f"seg-{i+1:04d}"))
            start_id = item.get("start_unit_id") or item.get("start_entry_id") or item.get("start_sentence_id")
            end_id = item.get("end_unit_id") or item.get("end_entry_id") or item.get("end_sentence_id")
            if start_id is None or end_id is None:
                raise RuntimeError(f"Segment {i} missing IDs")
            items.append(SegmentationPlanItem(
                segment_id=sid,
                start_sentence_id=int(start_id),
                end_sentence_id=int(end_id),
                boundary_reason=str(item.get("boundary_reason", "")),
            ))

        plan = SegmentationPlan(schema_version=1, segments=items)
        validate_plan_coverage(plan, total)
        return plan

    # ---- Reconstruction helpers ----

    @staticmethod
    def _reconstruct_entry_plan(plan: SegmentationPlan, entries: List[dict]) -> List[dict]:
        entry_by_index = {e["index"]: e for e in entries}
        segments = []
        for pi in plan.segments:
            covered = []
            for idx in range(pi.start_sentence_id, pi.end_sentence_id + 1):
                e = entry_by_index.get(idx)
                if e:
                    covered.append(e)
            if not covered:
                continue
            text = _dedup_merge_texts([e["text"] for e in covered])
            text = text.rstrip().rstrip('"\')”’]')
            if text and text[-1] not in (".", "!", "?"):
                text += "."
            segments.append({
                "segment_id": pi.segment_id, "block_id": 0,
                "start": round(covered[0]["start"], 3), "end": round(covered[-1]["end"], 3),
                "text": text, "word_count": len(text.split()),
                "source_indices": [e["index"] for e in covered],
                "boundary_reason": pi.boundary_reason or "refined",
            })
        return segments

    @staticmethod
    def _map_sentences_to_entries_accurate(
        sentences: List[str],
        entries: List[dict],
        transcript_text: str,
    ) -> List[dict]:
        """Map punctuated sentences to real entry timestamps.

        For each punctuated sentence, find its first words as an exact
        substring in the continuous deduped transcript text, scanning
        forward sequentially.  Map the character position to an SRT
        entry via a character-level ownership map built by replaying
        the dedup merge logic — each character in transcript_text knows
        which SRT entry index contributed it, so timestamps are always
        real.

        Never falls back to proportional timestamps.
        """
        import re

        if not entries:
            raise ValueError("No entries to map against")

        # ── Build char-to-entry map by replaying dedup ───────────────
        # Each character in transcript_text is owned by the SRT entry
        # that contributed it (after overlap removal).  Track it here.
        char_to_entry: list[int] = []
        raw_texts = [e["text"] for e in entries]
        result = raw_texts[0]
        for _ in range(len(result)):
            char_to_entry.append(entries[0]["index"])
        result_words = result.split()

        for i in range(1, len(raw_texts)):
            current = raw_texts[i].strip()
            if not current:
                continue
            result_words = result.split()
            suffix = " ".join(result_words[-20:]).lower()
            current_lower = current.lower()
            best_ov = 0
            for ov in range(min(20, len(result_words), len(current.split())), 2, -1):
                if current_lower.startswith(" ".join(result_words[-ov:]).lower()):
                    best_ov = ov
                    break
            if best_ov > 0:
                overlap_str = " ".join(result_words[-best_ov:])
                trimmed = current[len(overlap_str):].strip()
                if trimmed:
                    result += " " + trimmed
                    char_to_entry.append(entries[i]["index"])  # the space
                    for _ in range(len(trimmed)):
                        char_to_entry.append(entries[i]["index"])
                # else: entry added nothing new — skip
            else:
                result += " " + current
                char_to_entry.append(entries[i]["index"])  # the space
                for _ in range(len(current)):
                    char_to_entry.append(entries[i]["index"])

        norm_transcript = re.sub(r"\s+", " ", result.strip().lower())

        # If char_to_entry grew differently from norm_transcript length,
        # trim or pad to match
        if len(char_to_entry) > len(norm_transcript):
            char_to_entry = char_to_entry[:len(norm_transcript)]
        while len(char_to_entry) < len(norm_transcript):
            char_to_entry.append(char_to_entry[-1] if char_to_entry else 0)

        entry_by_index = {e["index"]: e for e in entries}

        def _norm_words(text: str, n: int = 0) -> str:
            clean = re.sub(r"[^\w'\s]", " ", text)
            words = clean.split()
            return " ".join(words[:n]) if n else " ".join(words)

        def _find_prefix(prefix: str, start_pos: int) -> int:
            """Try progressively shorter prefixes, return match position or -1."""
            words = prefix.split()
            for n in range(min(len(words), 8), 1, -1):
                pos = norm_transcript.find(" ".join(words[:n]), start_pos)
                if pos >= 0:
                    return pos
            for w in words:
                if len(w) > 2:
                    pos = norm_transcript.find(w, start_pos)
                    if pos >= 0:
                        return pos
            return -1

        result_maps: list[dict] = []
        search_pos = 0
        zero_overlap_count = 0

        for sid, sentence in enumerate(sentences, 1):
            prefix = _norm_words(sentence, 8)
            if not prefix:
                if result_maps:
                    last = result_maps[-1]
                    st = et_ = last["end_time"]
                    se = ee = last["end_entry"]
                else:
                    st = et_ = entries[0]["start"]
                    se = ee = entries[0]["index"]
                result_maps.append({
                    "sentence_id": sid, "start_entry": se, "end_entry": ee,
                    "start_time": st, "end_time": et_,
                })
                continue

            match_pos = _find_prefix(prefix, search_pos)

            if match_pos < 0:
                logger.warning(
                    "Sentence %d zero-overlap (invented?): %r",
                    sid, sentence[:80],
                )
                zero_overlap_count += 1
                if result_maps:
                    last = result_maps[-1]
                    st = et_ = last["end_time"]
                    se = ee = last["end_entry"]
                else:
                    st = et_ = entries[0]["start"]
                    se = ee = entries[0]["index"]
                result_maps.append({
                    "sentence_id": sid, "start_entry": se, "end_entry": ee,
                    "start_time": st, "end_time": et_,
                })
                continue

            entry_idx = char_to_entry[match_pos] if match_pos < len(char_to_entry) else char_to_entry[-1]
            entry_obj = entry_by_index.get(entry_idx, entries[0])

            result_maps.append({
                "sentence_id": sid,
                "start_entry": entry_idx,
                "end_entry": entry_idx,
                "start_time": entry_obj["start"],
                "end_time": entry_obj["end"],
            })

            logger.debug(
                "Sentence %d: search_pos=%d → match=%d, entry %d (t=%.0f), prefix=%r",
                sid, search_pos, match_pos, entry_idx, entry_obj["start"], prefix[:60],
            )

            search_pos = match_pos + 1

        matched = len(sentences) - zero_overlap_count
        logger.info(
            "Sentence-to-entry mapping: %d sentences, %d matched, "
            "%d carry-forward (%.0f%%), last search_pos=%d/%d",
            len(sentences), matched, zero_overlap_count,
            zero_overlap_count / max(len(sentences), 1) * 100,
            search_pos, len(norm_transcript),
        )

        if zero_overlap_count > 0:
            logger.warning(
                "%d sentences had no word overlap — carried forward "
                "last known timestamp instead",
                zero_overlap_count,
            )

        return result_maps

    @staticmethod
    def _reconstruct_sentence_plan(plan, sentences, entry_ranges, entries):
        er_by_id = {er["sentence_id"]: er for er in entry_ranges}
        segments = []
        for pi in plan.segments:
            seg_sents = sentences[pi.start_sentence_id - 1:pi.end_sentence_id]
            if not seg_sents:
                continue
            text = " ".join(seg_sents)
            first_er = er_by_id.get(pi.start_sentence_id)
            last_er = er_by_id.get(pi.end_sentence_id)

            segments.append({
                "segment_id": pi.segment_id, "block_id": 0,
                "start": round(first_er["start_time"], 3) if first_er else 0.0,
                "end": round(last_er["end_time"], 3) if last_er else 0.0,
                "text": text, "word_count": len(text.split()),
                "source_indices": [],
                "boundary_reason": pi.boundary_reason or "refined",
            })

        # Force monotonic (proportional is already monotonic, but float rounding...)
        for i in range(1, len(segments)):
            if segments[i]["start"] < segments[i - 1]["end"]:
                segments[i]["start"] = segments[i - 1]["end"]
        return segments
