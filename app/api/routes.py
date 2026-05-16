import time
import os
import logging
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.audio.preprocess import normalize_audio, load_audio, validate_duration, cleanup_temp
from app.audio.quality import assess_quality
from app.audio.vad import extract_speech
from app.inference.pipeline import predict
from app.schemas.response import InferenceResponse, ErrorResponse
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".opus", ".m4a", ".ogg", ".flac"}


@router.post(
    "/analyze",
    response_model=InferenceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Unprocessable audio"},
    },
    summary="Infer voice attributes",
    description="Upload an audio file to predict gender and age bracket from voice.",
)
async def infer(file: UploadFile = File(...)):
    """Main inference endpoint.

    Accepts multipart audio upload, preprocesses, runs inference,
    and returns structured predictions with confidence scores.
    """
    start = time.perf_counter()
    tmp_input = None
    tmp_normalized = None

    try:
        # Validate file extension
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # Read and validate file size
        content = await file.read()
        if len(content) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum: {settings.max_upload_size_mb}MB",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Save to temp file for ffmpeg processing
        tmp_input = tempfile.mktemp(suffix=ext)
        with open(tmp_input, "wb") as f:
            f.write(content)

        # Normalize audio (convert to 16kHz mono PCM WAV)
        tmp_normalized = normalize_audio(tmp_input)

        # Load normalized audio
        audio, sr = load_audio(tmp_normalized)

        # Validate duration
        duration = validate_duration(audio, sr)
        logger.info("Audio loaded: %.1fs at %dHz", duration, sr)

        # Assess audio quality
        quality = assess_quality(audio, sr)
        if quality == "insufficient":
            raise HTTPException(
                status_code=422,
                detail="insufficient_audio: audio quality too poor for inference",
            )

        # Extract speech segments via VAD
        speech = extract_speech(audio, sr)

        # Run inference
        results = predict(speech, sr)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return InferenceResponse(
            gender=results["gender"],
            age_bracket=results["age_bracket"],
            processing_ms=elapsed_ms,
            audio_quality=quality,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Runtime error during inference: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Privacy: always clean up temp audio files
        cleanup_temp(tmp_input)
        cleanup_temp(tmp_normalized)
