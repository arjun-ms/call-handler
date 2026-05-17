import logging
import asyncio
import json
from pydantic import ValidationError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.response import WebSocketStartEvent
from app.audio.stream_session import StreamSession
from app.audio.codec import decode_burst

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/analyze")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    
    session = None
    call_id = None
    
    try:
        # Wait for the "start" event
        message = await websocket.receive()
        if "text" not in message:
            await websocket.close(code=1003, reason="Expected JSON 'start' event as text")
            return
            
        try:
            start_data = json.loads(message["text"])
            start_event = WebSocketStartEvent(**start_data)
        except (json.JSONDecodeError, ValidationError) as e:
            await websocket.close(code=1003, reason=f"Invalid start event: {e}")
            return
            
        call_id = start_event.call_id
        sample_rate = start_event.sample_rate
        encoding = start_event.encoding
        
        # Initialize the session
        session = StreamSession(sample_rate=sample_rate)
        logger.info(f"Started stream session for call_id={call_id} at {sample_rate}Hz")
        
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_bytes = message["bytes"]
                # Decode to float32
                audio_chunk = decode_burst(audio_bytes, sample_rate, encoding)
                
                # Run inference in a threadpool so we don't block the async event loop
                events = await asyncio.to_thread(session.process_chunk, audio_chunk)
                
                for event in events:
                    event_payload = {
                        "type": "inference_result",
                        "call_id": call_id,
                        **event
                    }
                    await websocket.send_json(event_payload)
                    
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        break
                except json.JSONDecodeError:
                    pass
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call_id={call_id}")
    except RuntimeError as e:
        # Starlette raises RuntimeError when trying to receive after disconnect
        if "disconnect" in str(e).lower():
            logger.info(f"WebSocket disconnected (runtime) for call_id={call_id}")
        else:
            logger.error(f"WebSocket runtime error for call_id={call_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"WebSocket error for call_id={call_id}: {e}", exc_info=True)
        # Attempt to close cleanly if still connected
        try:
            await websocket.close(code=1011, reason="Internal Server Error")
        except:
            pass
    finally:
        # Memory cleanup
        if session:
            del session
        logger.info(f"Session cleaned up for call_id={call_id}")
