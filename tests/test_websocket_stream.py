import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def mock_stream_session(monkeypatch):
    class MockSession:
        def __init__(self, sample_rate=16000):
            self.sample_rate = sample_rate
            
        def process_chunk(self, audio_chunk):
            # Return a fake event if we get enough bytes, else empty
            # Just return an event unconditionally for testing
            if len(audio_chunk) > 0:
                return [{
                    "speaker_id": "speaker_0",
                    "role": "Agent",
                    "gender": "female",
                    "age_bracket": "18-30",
                    "confidence_gender": 0.9,
                    "confidence_age": 0.8
                }]
            return []

    monkeypatch.setattr("app.api.websocket.StreamSession", MockSession)

def test_websocket_stream_lifecycle(mock_stream_session):
    client = TestClient(app)
    
    with client.websocket_connect("/ws/stream") as websocket:
        # 1. Initial handshake
        websocket.send_json({
            "type": "start",
            "call_id": "test_123",
            "sample_rate": 16000
        })
        
        # 2. Send binary chunk
        fake_audio = np.zeros(1600, dtype=np.int16).tobytes()
        websocket.send_bytes(fake_audio)
        
        # 3. Receive result
        response = websocket.receive_json()
        assert response["type"] == "inference_result"
        assert response["call_id"] == "test_123"
        assert response["speaker_id"] == "speaker_0"
        assert response["role"] == "Agent"
        
        # 4. Stop
        websocket.send_json({"type": "stop"})
        
def test_websocket_stream_no_start():
    client = TestClient(app)
    # If we don't send a start event, it should close
    with pytest.raises(Exception): # starlette.websockets.WebSocketDisconnect or similar
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_bytes(b"123")
            websocket.receive_json()
