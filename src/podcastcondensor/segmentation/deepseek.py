"""DeepSeek cloud segmentation — two-pass (punctuate then segment)."""

import json
import logging
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

    def __init__(self, client, model="deepseek-chat", timeout=300, max_tokens=12000, retries=1):
        self._client = client
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._retries = retries
        self._validator = SegmentationValidator(check_sentence_complete=False, check_round_trip=False)

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

    def _segment_no_punc(self, entries: List[dict], transcript_text: str) -> List[dict]:
        logger.info("No punctuation — adding via DeepSeek (%d chars)...", len(transcript_text))
        punctuated = self._call_punctuate(transcript_text)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', punctuated) if s.strip()]
        if len(sentences) <= 1:
            raise RuntimeError(f"Punctuation produced only {len(sentences)} sentence(s)")

        # Map sentences to entry ranges
        entry_ranges = self._map_sentences_to_entries(sentences, entries)

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
    def _map_sentences_to_entries(sentences: List[str], entries: List[dict]) -> List[dict]:
        """Map each sentence to proportional timing.

        Auto-caption entries are short overlapping fragments that get
        deduped away, making text-based mapping unreliable. Instead,
        allocate timing proportionally: sentence N gets its position
        based on N/total_sentences through the episode duration.
        """
        total_sec = entries[-1]["end"] - entries[0]["start"] if entries else 0
        total = len(sentences)
        result = []
        for sid, sentence in enumerate(sentences, 1):
            frac = sid / total
            start_t = entries[0]["start"] + frac * total_sec
            end_t = entries[0]["start"] + ((sid + 0.5) / total) * total_sec
            result.append({
                "sentence_id": sid,
                "start_entry": 1,
                "end_entry": 1,
                "start_time": start_t,
                "end_time": end_t,
            })
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
