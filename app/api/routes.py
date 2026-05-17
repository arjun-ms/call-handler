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

        # Use VadTrigger and Diarizer to separate speakers
        import numpy as np
        from app.audio.vad import VadTrigger
        from app.inference.diarizer import Diarizer

        trigger = VadTrigger(sample_rate=sr)
        diarizer = Diarizer()

        # Get bursts
        bursts = trigger.process_audio(audio)
        if trigger.current_burst:
            bursts.append(np.concatenate(trigger.current_burst))

        if not bursts:
            logger.warning("No speech bursts detected by VadTrigger, falling back to full audio")
            customer_speech = audio
        else:
            # Diarize bursts
            speaker_segments = {}
            first_speaker = None
            
            for burst in bursts:
                spk_id = diarizer.process_burst(burst)
                if spk_id not in speaker_segments:
                    speaker_segments[spk_id] = []
                    if first_speaker is None:
                        first_speaker = spk_id
                speaker_segments[spk_id].append(burst)
                
            # Heuristic: Assume first speaker is Agent. The other speaker is Customer.
            customer_speaker = None
            if len(speaker_segments) == 1:
                customer_speaker = list(speaker_segments.keys())[0]
            else:
                for spk_id in speaker_segments.keys():
                    if spk_id != first_speaker:
                        customer_speaker = spk_id
                        break
                        
            if not customer_speaker:
                customer_speaker = first_speaker
                
            logger.info("Diarization: selected %s as customer (total speakers: %d)", 
                        customer_speaker, len(speaker_segments))
                        
            customer_speech = np.concatenate(speaker_segments[customer_speaker])
            
        # Cap inference to max_inference_speech_sec to prevent latency/OOM spikes
        max_samples = int(settings.max_inference_speech_sec * sr)
        if len(customer_speech) > max_samples:
            logger.info("Capping customer speech from %.1fs to %.1fs", len(customer_speech)/sr, settings.max_inference_speech_sec)
            customer_speech = customer_speech[:max_samples]

        # Run inference on customer speech
        results = predict(customer_speech, sr)

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
