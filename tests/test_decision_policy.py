"""
Tests for decision_policy.py — the confidence-gating layer.

Pure logic tests: no audio, no model. Validates that the decision thresholds
behave correctly across all combinations of audio quality and confidence level.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.audio_quality import QualityReport
from app.decision_policy import decide, AUTO_USE_CONFIDENCE, FLAG_CONFIDENCE


def _quality(q: str) -> QualityReport:
    """Helper: build a QualityReport with the given quality label."""
    return QualityReport(quality=q, snr_db=20.0, speech_ratio=0.8, clipping_ratio=0.0)


# ---------------------------------------------------------------------------
# Insufficient audio — always discard regardless of confidence
# ---------------------------------------------------------------------------

def test_insufficient_always_discards():
    q = _quality("insufficient")
    for confidence in [0.0, 0.5, 0.85, 0.99, 1.0]:
        result = decide(confidence, q)
        assert result.decision == "discard", (
            f"insufficient audio with confidence={confidence} should always "
            f"discard, got {result.decision}"
        )


# ---------------------------------------------------------------------------
# Good audio quality
# ---------------------------------------------------------------------------

def test_good_quality_high_confidence_auto_use():
    q = _quality("good")
    result = decide(AUTO_USE_CONFIDENCE["good"], q)
    assert result.decision == "auto_use"


def test_good_quality_medium_confidence_flag():
    q = _quality("good")
    conf = (FLAG_CONFIDENCE["good"] + AUTO_USE_CONFIDENCE["good"]) / 2
    result = decide(conf, q)
    assert result.decision == "flag_for_review"


def test_good_quality_low_confidence_discard():
    q = _quality("good")
    result = decide(FLAG_CONFIDENCE["good"] - 0.01, q)
    assert result.decision == "discard"


def test_good_quality_zero_confidence_discard():
    q = _quality("good")
    result = decide(0.0, q)
    assert result.decision == "discard"


# ---------------------------------------------------------------------------
# Degraded audio quality — requires higher confidence to auto_use
# ---------------------------------------------------------------------------

def test_degraded_requires_higher_bar_than_good():
    """
    The same confidence that gets 'auto_use' on good audio should get
    'flag_for_review' or 'discard' on degraded audio — this is the key
    design invariant of the decision policy.
    """
    good_bar = AUTO_USE_CONFIDENCE["good"]
    degraded_bar = AUTO_USE_CONFIDENCE["degraded"]
    assert degraded_bar > good_bar, (
        "Degraded audio should require HIGHER confidence to auto_use"
    )

    # At the 'good' auto_use threshold, degraded should NOT auto_use
    result_on_degraded = decide(good_bar, _quality("degraded"))
    assert result_on_degraded.decision != "auto_use", (
        f"Confidence {good_bar} on degraded audio should not auto_use"
    )


def test_degraded_high_confidence_auto_use():
    q = _quality("degraded")
    result = decide(AUTO_USE_CONFIDENCE["degraded"], q)
    assert result.decision == "auto_use"


def test_degraded_medium_confidence_flag():
    q = _quality("degraded")
    conf = (FLAG_CONFIDENCE["degraded"] + AUTO_USE_CONFIDENCE["degraded"]) / 2
    result = decide(conf, q)
    assert result.decision == "flag_for_review"


# ---------------------------------------------------------------------------
# Decision field values are correct strings
# ---------------------------------------------------------------------------

def test_decision_values_are_valid_strings():
    valid = {"auto_use", "flag_for_review", "discard"}
    for quality in ("good", "degraded", "insufficient"):
        for conf in [0.0, 0.3, 0.6, 0.75, 0.9, 1.0]:
            result = decide(conf, _quality(quality))
            assert result.decision in valid, (
                f"Unknown decision '{result.decision}' for quality={quality} "
                f"confidence={conf}"
            )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def test_exactly_at_auto_use_threshold_good():
    conf = AUTO_USE_CONFIDENCE["good"]
    result = decide(conf, _quality("good"))
    assert result.decision == "auto_use"


def test_just_below_auto_use_threshold_good():
    conf = AUTO_USE_CONFIDENCE["good"] - 0.001
    result = decide(conf, _quality("good"))
    assert result.decision in ("flag_for_review", "discard")


def test_exactly_at_flag_threshold_good():
    conf = FLAG_CONFIDENCE["good"]
    result = decide(conf, _quality("good"))
    assert result.decision in ("flag_for_review", "auto_use")
