from typing import Literal, Optional
from pydantic import BaseModel, Field

GenderLabel = Literal["male", "female", "unknown"]
AgeBracketLabel = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQuality = Literal["good", "degraded", "insufficient"]
Decision = Literal["auto_use", "flag_for_review", "discard"]


class AttributeEstimate(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class GenderEstimate(AttributeEstimate):
    prediction: GenderLabel


class AgeBracketEstimate(AttributeEstimate):
    prediction: AgeBracketLabel


class LanguageEstimate(BaseModel):
    prediction: Optional[str] = None  # e.g. "en-IN", "hi-IN"; None if not run
    confidence: Optional[float] = None


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderEstimate
    age_bracket: AgeBracketEstimate
    language: Optional[LanguageEstimate] = None
    processing_ms: int
    audio_quality: AudioQuality
    decision: Decision  # extension beyond the base contract — see decision_policy.py
