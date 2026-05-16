import logging
import time
import tempfile
import os
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.audio.preprocess import normalize_audio, load_audio, validate_duration, cleanup_temp
from app.audio.quality import assess_quality
from app.audio.vad import extract_speech
from app.inference.pipeline import predict
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/analyze")
async def websocket_infer(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    
    try:
        while True:
            # Receive message
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
                await websocket.send_json({"status": "buffering", "bytes_received": len(audio_buffer)})
                
            elif "text" in message:
                if message["text"].upper() == "PROCESS":
                    if len(audio_buffer) == 0:
                        await websocket.send_json({"error": "No audio buffered"})
                        continue
                        
                    start = time.perf_counter()
                    tmp_input = None
                    tmp_normalized = None
                    
                    try:
                        tmp_input = tempfile.mktemp(suffix=".webm")
                        with open(tmp_input, "wb") as f:
                            f.write(audio_buffer)
                            
                        tmp_normalized = normalize_audio(tmp_input)
                        audio, sr = load_audio(tmp_normalized)
                        duration = validate_duration(audio, sr)
                        
                        quality = assess_quality(audio, sr)
                        if quality == "insufficient":
                            await websocket.send_json({"error": "insufficient_audio"})
                            continue
                            
                        speech = extract_speech(audio, sr)
                        results = predict(speech, sr)
                        
                        elapsed_ms = int((time.perf_counter() - start) * 1000)
                        
                        await websocket.send_json({
                            "status": "success",
                            "gender": results["gender"],
                            "age_bracket": results["age_bracket"],
                            "processing_ms": elapsed_ms,
                            "audio_quality": quality
                        })
                        
                    except Exception as e:
                        logger.error("Error processing websocket audio: %s", e)
                        await websocket.send_json({"error": str(e)})
                    finally:
                        cleanup_temp(tmp_input)
                        cleanup_temp(tmp_normalized)
                        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected, final buffer size %d bytes", len(audio_buffer))
