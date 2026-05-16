from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid


class PredictionResult(BaseModel):
    """A single prediction with confidence score."""

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    """Successful inference response."""

    contact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    gender: PredictionResult
    age_bracket: PredictionResult
    processing_ms: int
    audio_quality: Literal["good", "degraded", "insufficient"]


class ErrorResponse(BaseModel):
    """Error response for failed requests."""

    error: str
    detail: Optional[str] = None
    contact_id: Optional[str] = None
