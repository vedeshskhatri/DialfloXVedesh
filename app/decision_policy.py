"""
This module is the actual differentiator. A raw model prediction + confidence
number is not, by itself, a decision about what to do with that prediction.
Dialflo's own product philosophy (stated publicly) is tier-one automation with
instant escalation on uncertainty — this module is where that philosophy gets
encoded for this specific service, instead of left implicit.

Downstream (the calling voice-agent system) is expected to branch on the
`decision` field:
  - auto_use: safe to use the prediction to personalize the conversation
  - flag_for_review: prediction returned, but system should not act on it
    silently — surface to a human/QA queue, or fall back to neutral phrasing
  - discard: don't use gender/age at all this call; treat as unknown
"""
from dataclasses import dataclass

from app.audio_quality import QualityReport

# Confidence thresholds — independent per audio_quality tier, since a
# "degraded" clip needs a higher bar to be trusted than a "good" one.
# See DESIGN_DECISIONS.md: combining these two signals properly needs a
# calibration dataset; this is a documented starting point, not a final answer.
AUTO_USE_CONFIDENCE = {
    "good": 0.75,
    "degraded": 0.90,
    "insufficient": 1.01,  # never auto-use on insufficient audio
}
FLAG_CONFIDENCE = {
    "good": 0.5,
    "degraded": 0.6,
    "insufficient": 0.0,
}


@dataclass
class Decision:
    decision: str  # "auto_use" | "flag_for_review" | "discard"


def decide(min_confidence: float, quality: QualityReport) -> Decision:
    q = quality.quality
    if q == "insufficient":
        return Decision(decision="discard")
    if min_confidence >= AUTO_USE_CONFIDENCE[q]:
        return Decision(decision="auto_use")
    if min_confidence >= FLAG_CONFIDENCE[q]:
        return Decision(decision="flag_for_review")
    return Decision(decision="discard")
