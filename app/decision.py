"""Phase 8 - decision engine.

Purely rule-based: the genuine/counterfeit signatures are already known
(G->N->V vs N->G->V), so this module only needs to compare the compressed
sequence string against those two patterns and report a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.sequence import SequenceRun
from app.utils import get_logger

logger = get_logger(__name__)

GENUINE = "GENUINE"
COUNTERFEIT = "COUNTERFEIT"
UNKNOWN = "UNKNOWN"


@dataclass
class DecisionResult:
    label: str
    sequence: str
    confidence: float
    reasoning: str


class DecisionEngine:
    def __init__(self, settings: dict):
        cfg = settings.get("decision", {})
        self.genuine_sequence = cfg.get("genuine_sequence", "GNV")
        self.counterfeit_sequence = cfg.get("counterfeit_sequence", "NGV")
        self.min_overall_confidence = cfg.get("min_overall_confidence", 0.4)
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _weighted_confidence(runs: list[SequenceRun]) -> float:
        if not runs:
            return 0.0
        total_frames = sum(r.frame_count for r in runs)
        if total_frames == 0:
            return 0.0
        return sum(r.mean_confidence * r.frame_count for r in runs) / total_frames

    def decide(self, sequence: str, runs: list[SequenceRun]) -> DecisionResult:
        confidence = self._weighted_confidence(runs)

        if sequence == self.genuine_sequence:
            label = GENUINE
            reasoning = (
                f"Detected color transition '{sequence}' matches the genuine "
                f"signature '{self.genuine_sequence}' (Green -> Neutral -> Violet)."
            )
        elif sequence == self.counterfeit_sequence:
            label = COUNTERFEIT
            reasoning = (
                f"Detected color transition '{sequence}' matches the counterfeit "
                f"signature '{self.counterfeit_sequence}' (Neutral -> Green -> Violet)."
            )
        elif sequence == "":
            label = UNKNOWN
            confidence = 0.0
            reasoning = (
                "No stable color transitions survived filtering; unable to "
                "determine a sequence. Check ROI alignment and lighting."
            )
        else:
            label = UNKNOWN
            reasoning = (
                f"Detected sequence '{sequence}' matches neither the genuine "
                f"('{self.genuine_sequence}') nor counterfeit "
                f"('{self.counterfeit_sequence}') signature."
            )

        if label != UNKNOWN and confidence < self.min_overall_confidence:
            reasoning += (
                f" Overall confidence {confidence:.2f} is below the minimum "
                f"threshold {self.min_overall_confidence:.2f}; downgrading to UNKNOWN."
            )
            label = UNKNOWN

        self.logger.info(
            "Decision: %s (sequence=%s, confidence=%.3f)", label, sequence, confidence
        )
        return DecisionResult(
            label=label, sequence=sequence, confidence=round(confidence, 4), reasoning=reasoning
        )
