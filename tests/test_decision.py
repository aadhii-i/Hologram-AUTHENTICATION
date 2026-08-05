"""Unit tests for app.decision: the GNV/NGV rule-based decision engine."""

from __future__ import annotations

from app.decision import COUNTERFEIT, GENUINE, UNKNOWN, DecisionEngine
from app.sequence import SequenceRun

SETTINGS = {
    "decision": {
        "genuine_sequence": "GNV",
        "counterfeit_sequence": "NGV",
        "min_overall_confidence": 0.4,
    }
}


def _runs(symbols: str, confidence: float = 0.8) -> list[SequenceRun]:
    labels = {"G": "green", "N": "neutral", "V": "violet"}
    return [
        SequenceRun(
            label=labels[s], symbol=s, start_frame=i * 4, end_frame=i * 4 + 3,
            frame_count=4, mean_confidence=confidence,
        )
        for i, s in enumerate(symbols)
    ]


def test_decide_genuine():
    engine = DecisionEngine(SETTINGS)
    result = engine.decide("GNV", _runs("GNV"))
    assert result.label == GENUINE
    assert result.confidence > 0.4


def test_decide_counterfeit():
    engine = DecisionEngine(SETTINGS)
    result = engine.decide("NGV", _runs("NGV"))
    assert result.label == COUNTERFEIT


def test_decide_unknown_for_unrecognized_sequence():
    engine = DecisionEngine(SETTINGS)
    result = engine.decide("GVN", _runs("GVN"))
    assert result.label == UNKNOWN


def test_decide_unknown_for_empty_sequence():
    engine = DecisionEngine(SETTINGS)
    result = engine.decide("", [])
    assert result.label == UNKNOWN
    assert result.confidence == 0.0


def test_decide_downgrades_low_confidence_to_unknown():
    engine = DecisionEngine(SETTINGS)
    result = engine.decide("GNV", _runs("GNV", confidence=0.1))
    assert result.label == UNKNOWN
