"""Validation for segmentation plans and final segments.

Two validators:
  1. ``validate_plan_coverage`` — checks that a ``SegmentationPlan``
     partitions all sentence IDs contiguously.
  2. ``SegmentationValidator`` — checks final ``Segment`` dict invariants
     (round-trip fidelity, sentence completeness, monotonic times, etc.).

Any failure raises ``SegmentationValidationError``, which triggers
fallback to deterministic segmentation.
"""

import logging
import re
from typing import List

from podcastcondensor.segmentation.schemas import SegmentationPlan

logger = logging.getLogger(__name__)


class SegmentationValidationError(ValueError):
    """Raised when segmentation validation fails (triggers fallback)."""
    pass


# ---------------------------------------------------------------------------
# Plan-level validation (before reconstruction)
# ---------------------------------------------------------------------------


def validate_plan_coverage(
    plan: SegmentationPlan,
    total_sentences: int,
) -> None:
    """Validate that the plan covers all sentences with no gaps/overlaps.

    Args:
        plan: The segmentation plan from the LLM.
        total_sentences: Total number of sentence units.

    Raises:
        SegmentationValidationError: On any coverage violation.
    """
    if not plan.segments:
        raise SegmentationValidationError("Plan has zero segments")

    if plan.schema_version != 1:
        raise SegmentationValidationError(
            f"Unsupported schema_version: {plan.schema_version} (expected 1)"
        )

    # Check first segment starts at sentence 1
    first = plan.segments[0]
    if first.start_sentence_id != 1:
        raise SegmentationValidationError(
            f"First segment starts at sentence {first.start_sentence_id}, "
            f"expected 1"
        )

    # Check last segment ends at total_sentences
    last = plan.segments[-1]
    if last.end_sentence_id != total_sentences:
        raise SegmentationValidationError(
            f"Last segment ends at sentence {last.end_sentence_id}, "
            f"expected {total_sentences}"
        )

    # Check contiguity: each segment starts where the previous ended + 1
    prev_end = 0
    for i, item in enumerate(plan.segments):
        if item.start_sentence_id != prev_end + 1:
            raise SegmentationValidationError(
                f"Segment {i} ({item.segment_id}): starts at sentence "
                f"{item.start_sentence_id}, expected {prev_end + 1}"
                f" (gap or overlap at position {prev_end + 1})"
            )
        if item.end_sentence_id < item.start_sentence_id:
            raise SegmentationValidationError(
                f"Segment {i} ({item.segment_id}): end_sentence_id "
                f"{item.end_sentence_id} < start_sentence_id "
                f"{item.start_sentence_id}"
            )
        prev_end = item.end_sentence_id


# ---------------------------------------------------------------------------
# Final segment validation (after reconstruction)
# ---------------------------------------------------------------------------


class SegmentationValidator:
    """Validates final ``Segment`` dict invariants.

    Args:
        check_sentence_complete: Whether to require each segment to end
            with sentence punctuation.  Set to ``False`` for auto-caption
            transcripts where punctuation is unreliable.
    """

    def __init__(
        self,
        check_sentence_complete: bool = True,
        check_round_trip: bool = True,
    ):
        self._check_sentence_complete = check_sentence_complete
        self._check_round_trip = check_round_trip

    def validate_all(
        self,
        segments: List[dict],
        transcript_text: str,
    ) -> None:
        """Run all validation checks.

        Args:
            segments: Reconstructed segment dicts.
            transcript_text: Original deduplicated transcript.

        Raises:
            SegmentationValidationError: On any check failure.
        """
        self._validate_non_empty(segments)
        self._validate_monotonic_times(segments)
        if self._check_sentence_complete:
            self._validate_sentence_complete(segments)
        if self._check_round_trip:
            self._validate_round_trip(segments, transcript_text)
        self._validate_no_empty_text(segments)

        logger.info(
            "Segmentation validation passed: %d segments, %d chars",
            len(segments), len(transcript_text),
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _validate_non_empty(self, segments: List[dict]) -> None:
        if not segments:
            raise SegmentationValidationError("Zero segments after reconstruction")

    def _validate_monotonic_times(self, segments: List[dict]) -> None:
        for i, seg in enumerate(segments):
            if seg["start"] > seg["end"]:
                raise SegmentationValidationError(
                    f"Segment {i} ({seg['segment_id']}): start time "
                    f"{seg['start']} > end time {seg['end']}"
                )
        for i in range(1, len(segments)):
            if segments[i]["start"] < segments[i - 1]["end"]:
                raise SegmentationValidationError(
                    f"Segment {i} ({segments[i]['segment_id']}): start time "
                    f"{segments[i]['start']} < previous end time "
                    f"{segments[i-1]['end']}"
                )

    def _validate_sentence_complete(self, segments: List[dict]) -> None:
        for seg in segments:
            text = seg.get("text", "").rstrip()
            if not text:
                continue
            # Strip trailing quotes/parens before checking
            stripped = text.rstrip('"\')”’]')
            if not stripped.endswith((".", "!", "?")):
                raise SegmentationValidationError(
                    f"Segment {seg['segment_id']} does not end with "
                    f"sentence-ending punctuation: ...{stripped[-60:]}"
                )

    def _validate_round_trip(
        self,
        segments: List[dict],
        transcript_text: str,
    ) -> None:
        reconstructed = " ".join(s["text"] for s in segments)
        if self._normalize(reconstructed) != self._normalize(transcript_text):
            # Log diff for debugging
            norm_rec = self._normalize(reconstructed)
            norm_orig = self._normalize(transcript_text)
            diff_pos = 0
            while (
                diff_pos < len(norm_rec)
                and diff_pos < len(norm_orig)
                and norm_rec[diff_pos] == norm_orig[diff_pos]
            ):
                diff_pos += 1
            raise SegmentationValidationError(
                f"Round-trip fidelity check failed at character {diff_pos}: "
                f"reconstructed={len(norm_rec)} chars vs "
                f"original={len(norm_orig)} chars"
            )

    def _validate_no_empty_text(self, segments: List[dict]) -> None:
        for seg in segments:
            if not seg.get("text", "").strip():
                raise SegmentationValidationError(
                    f"Segment {seg['segment_id']} has empty text"
                )
            if seg.get("word_count", 0) <= 0:
                raise SegmentationValidationError(
                    f"Segment {seg['segment_id']} has non-positive word count"
                )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())
