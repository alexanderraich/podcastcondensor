"""Strategy interfaces for LLM-dependent pipeline phases.

Each strategy encapsulates the orchestration logic for one pipeline phase
(classify, knowledge extract), leaving transport details to the LLM client.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class ClassificationFailedError(Exception):
    """Raised when the classification provider fails completely.

    The pipeline must NOT treat this as a valid decision pass.  It should
    fall back to an alternative provider or abort — never let failure
    masquerade as a ``{"label": "maybe"}`` result.
    """
    pass


class ClassifierStrategy(ABC):
    """Strategy for Phase B/C: segment classification + maybe resolution.

    Implementations wrap a specific LLM client + prompt combination and
    handle batching, retry, JSON parsing, and fallback internally.
    """

    @abstractmethod
    def classify_segments(
        self,
        segments: List[dict],
        global_outline: str,
        block_summaries: List[dict],
        max_segments_per_batch: int = 3,
        output_path: Optional[str] = None,
        universe_state_context: str = "",
        kept_claims_so_far: Optional[List[str]] = None,
    ) -> List[dict]:
        """Classify all segments, returning ``[{id, label, reason}, ...]``.

        Must be resumable when ``output_path`` is provided (save progress
        incrementally and skip already-classified segments on re-run).
        """
        ...

    @abstractmethod
    def resolve_maybe(
        self,
        maybe_segments: List[dict],
        all_segments: List[dict],
        all_decisions: List[dict],
    ) -> List[dict]:
        """Re-evaluate ``maybe`` segments, returning updated decisions list.

        Non-maybe segments are returned unchanged.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Provider name used in log messages and cache keys."""
        ...


class KnowledgeExtractionStrategy(ABC):
    """Strategy for Phase D: extract structured knowledge from episode data.

    Implementations call an LLM with block summaries + outline and return
    the 8-category extraction dict that ``UniverseState.add_episode_knowledge``
    expects.
    """

    @abstractmethod
    def extract(
        self,
        block_summaries: List[dict],
        global_outline: str,
        episode_title: str = "",
        episode_number: Optional[int] = None,
    ) -> dict:
        """Extract structured knowledge.

        Returns a dict with keys matching ``DEFAULT_STATE``, or an empty dict
        on failure.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Provider name used in log messages and cache fingerprints."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier used by this strategy (for cache keys)."""
        ...
