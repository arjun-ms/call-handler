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


class WebSocketStartEvent(BaseModel):
    """Event sent by client to initiate streaming."""
    
    type: Literal["start"]
    call_id: str
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"


class WebSocketInferenceResult(BaseModel):
    """Event sent by server to client with real-time inference predictions."""
    
    type: Literal["inference_result"]
    call_id: str
    speaker_id: str
    role: Literal["Agent", "Customer"]
    gender: str
    age_bracket: str
    confidence_gender: float = Field(ge=0.0, le=1.0)
    confidence_age: float = Field(ge=0.0, le=1.0)
