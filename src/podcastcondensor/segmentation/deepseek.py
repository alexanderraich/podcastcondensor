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
        """Map punctuated sentences to real entry timestamps via word overlap.

        For each sentence (from the DeepSeek-punctuated output), we scan
        forward through the cleaned SRT entries (which have real timestamps)
        and find the range of entries with the highest word-overlap score.

        This handles DeepSeek's occasional word insertions/deletions gracefully
        because it only cares about overlapping words, not exact positions.
        Forward-only scanning prevents backwards timestamp jumps.

        Never falls back to proportional timestamps.
        """
        def _words(text: str) -> set:
            """Lowercased word set, stripped of punctuation."""
            return set(re.sub(r"[^\w'\s]", "", text).lower().split())

        if not entries:
            raise ValueError("No entries to map against")

        # Pre-compute word sets for each entry (lowercase, no punctuation)
        entry_words = [
            (e, _words(e.get("text", "")))
            for e in entries
        ]

        result: List[dict] = []
        entry_idx = 0  # forward scan cursor
        hallucinated_count = 0
        total_overlap_score = 0
        total_entries_scanned = 0

        for sid, sentence in enumerate(sentences, 1):
            sw = _words(sentence)
            if not sw:
                # Empty sentence — carry forward
                if result:
                    last = result[-1]
                    st = et_ = last["end_time"]
                    se = ee = last["end_entry"]
                else:
                    st = et_ = entries[0]["start"]
                    se = ee = entries[0]["index"]
                result.append({
                    "sentence_id": sid, "start_entry": se, "end_entry": ee,
                    "start_time": st, "end_time": et_,
                })
                continue

            # Scan forward from entry_idx, find the entry range with
            # best word overlap.  Stop when overlap drops off for
            # several consecutive entries.
            best_start = entry_idx
            best_end = entry_idx
            best_score = 0
            scan_limit = min(entry_idx + 200, len(entry_words))
            running_score = 0
            for i in range(entry_idx, scan_limit):
                ew = entry_words[i][1]
                if not ew:
                    if running_score > 0:
                        running_score *= 0.9  # decay through empty entries
                    continue
                overlap = len(sw & ew)
                if overlap > 0:
                    running_score += overlap
                else:
                    running_score *= 0.5  # decay
                if running_score > best_score:
                    best_score = running_score
                    best_end = i
                    # Reset start if this is the first entry in a new cluster
                    if best_start == best_end or running_score == overlap:
                        best_start = i

            if best_score < 1:
                # No overlap at all — hallucinated sentence
                logger.debug(
                    "Sentence %d (cursor=%d, scanned=%d-%d) no overlap: %r",
                    sid, entry_idx, entry_idx, scan_limit, sentence[:80],
                )
                logger.warning(
                    "Sentence %d completely unmatchable (hallucinated): %r",
                    sid, sentence[:80],
                )
                hallucinated_count += 1
                if result:
                    last = result[-1]
                    st = et_ = last["end_time"]
                    se = ee = last["end_entry"]
                else:
                    st = et_ = entries[0]["start"]
                    se = ee = entries[0]["index"]
                result.append({
                    "sentence_id": sid, "start_entry": se, "end_entry": ee,
                    "start_time": st, "end_time": et_,
                })
                continue

            # Use the first and last overlapping entry for timestamps
            start_e = entry_words[best_start][0]
            end_e = entry_words[best_end][0]

            result.append({
                "sentence_id": sid,
                "start_entry": start_e["index"],
                "end_entry": end_e["index"],
                "start_time": start_e["start"],
                "end_time": end_e["end"],
            })

            total_overlap_score += best_score
            total_entries_scanned += scan_limit - entry_idx

            logger.debug(
                "Sentence %d mapped: cursor %d→%d, entries %d–%d "
                "(t=%.0f–%.0f), score=%d",
                sid, entry_idx, best_end,
                start_e["index"], end_e["index"],
                start_e["start"], end_e["end"],
                best_score,
            )

            # Advance cursor past the matched entries
            entry_idx = max(entry_idx, best_end)

        # ── Summary debug ─────────────────────────────────────────────
        matched = len(sentences) - hallucinated_count
        avg_scan = total_entries_scanned / max(len(sentences), 1)
        logger.info(
            "Sentence-to-entry mapping: %d sentences, %d matched, "
            "%d hallucinated/skipped (%.0f%%), avg scan window=%.0f entries, "
            "cursor reached entry %d/%d",
            len(sentences), matched, hallucinated_count,
            hallucinated_count / max(len(sentences), 1) * 100,
            avg_scan, entry_idx, len(entry_words),
        )

        # Sanity check: too many hallucinated → fail
        if len(sentences) > 50 and hallucinated_count / len(sentences) > 0.5:
            raise ValueError(
                f"{hallucinated_count}/{len(sentences)} sentences "
                f"({hallucinated_count/len(sentences):.0%}) fully unmatchable "
                f"— punctuated output is too divergent"
            )

        return result

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
