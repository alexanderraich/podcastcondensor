"""Typed schema objects for LLM segmentation plans.

The contract between the LLM and the reconstruction code: a
``SegmentationPlan`` is a partition of all sentence IDs into contiguous
segments.  Each segment references the sentence ID range it covers.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SegmentationPlanItem:
    """One segment in the LLM's output plan."""

    segment_id: str
    start_sentence_id: int
    end_sentence_id: int
    boundary_reason: str = ""


@dataclass
class SegmentationPlan:
    """The full plan: a partition of sentence IDs into contiguous segments.

    Must cover the entire transcript (first segment starts at sentence 1,
    last segment ends at total_sentences, no gaps, no overlaps).
    """

    schema_version: int
    segments: List[SegmentationPlanItem] = field(default_factory=list)
