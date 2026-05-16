import numpy as np
from app.audio.stream_session import StreamSession

def test_stream_session_initialization():
    session = StreamSession(sample_rate=16000)
    assert session.sample_rate == 16000
    assert len(session.speaker_role_map) == 0

def test_greeting_heuristic(monkeypatch):
    # Mock VAD to immediately return bursts
    class MockVad:
        def __init__(self, *args, **kwargs):
            pass
        def process_audio(self, chunk):
            return [chunk] # treat every chunk as a full burst
            
    # Mock Diarizer to alternate speakers based on length just for testing
    class MockDiarizer:
        def process_burst(self, burst):
            # If length is even, speaker 1. If odd, speaker 2.
            return "speaker_1" if len(burst) % 2 == 0 else "speaker_2"
            
    # Mock predict
    def mock_predict(*args, **kwargs):
        return {
            "gender": {"prediction": "female", "confidence": 0.9},
            "age_bracket": {"prediction": "18-30", "confidence": 0.8}
        }

    monkeypatch.setattr("app.audio.stream_session.VadTrigger", MockVad)
    monkeypatch.setattr("app.audio.stream_session.Diarizer", MockDiarizer)
    monkeypatch.setattr("app.audio.stream_session.predict", mock_predict)

    session = StreamSession()
    
    # First burst (even length -> speaker_1) — Agent, skip inference
    chunk1 = np.zeros(16000, dtype=np.float32)
    events1 = session.process_chunk(chunk1)
    assert len(events1) == 1
    assert events1[0]["speaker_id"] == "speaker_1"
    assert events1[0]["role"] == "Agent" # First speaker -> Agent
    assert events1[0]["status"] == "speaking"
    
    # Second burst (odd length -> speaker_2) — Customer, full inference
    chunk2 = np.zeros(16001, dtype=np.float32)
    events2 = session.process_chunk(chunk2)
    assert len(events2) == 1
    assert events2[0]["speaker_id"] == "speaker_2"
    assert events2[0]["role"] == "Customer" # Subsequent speaker -> Customer
    assert events2[0]["gender"] == "female"

    # Third burst (even length -> speaker_1 again) — Still Agent
    chunk3 = np.zeros(16000, dtype=np.float32)
    events3 = session.process_chunk(chunk3)
    assert len(events3) == 1
    assert events3[0]["speaker_id"] == "speaker_1"
    assert events3[0]["role"] == "Agent" # Should remain Agent


def test_agent_bursts_skip_inference(monkeypatch):
    """Agent bursts should NOT run the heavy predict() model.
    They should return a minimal event with role=Agent and status=speaking."""

    class MockVad:
        def __init__(self, *args, **kwargs):
            pass
        def process_audio(self, chunk):
            return [chunk]

    class MockDiarizer:
        def process_burst(self, burst):
            return "speaker_0"  # Always same speaker (the Agent)

    predict_call_count = 0
    def mock_predict(*args, **kwargs):
        nonlocal predict_call_count
        predict_call_count += 1
        return {
            "gender": {"prediction": "male", "confidence": 0.9},
            "age_bracket": {"prediction": "31-45", "confidence": 0.8}
        }

    monkeypatch.setattr("app.audio.stream_session.VadTrigger", MockVad)
    monkeypatch.setattr("app.audio.stream_session.Diarizer", MockDiarizer)
    monkeypatch.setattr("app.audio.stream_session.predict", mock_predict)

    session = StreamSession()

    # Send a burst — first speaker should become Agent
    chunk = np.zeros(16000, dtype=np.float32)
    events = session.process_chunk(chunk)

    assert len(events) == 1
    event = events[0]

    # Should have role and status but NOT gender/age predictions
    assert event["role"] == "Agent"
    assert event["speaker_id"] == "speaker_0"
    assert event.get("status") == "speaking"
    assert "gender" not in event
    assert "age_bracket" not in event

    # predict() should NOT have been called
    assert predict_call_count == 0


def test_customer_bursts_run_inference(monkeypatch):
    """Customer bursts should run predict() and return full predictions."""

    call_order = []

    class MockVad:
        def __init__(self, *args, **kwargs):
            pass
        def process_audio(self, chunk):
            return [chunk]

    class MockDiarizer:
        def __init__(self):
            self._call_count = 0
        def process_burst(self, burst):
            self._call_count += 1
            # First call -> Agent, second call -> Customer
            return "agent_spk" if self._call_count == 1 else "customer_spk"

    predict_call_count = 0
    def mock_predict(*args, **kwargs):
        nonlocal predict_call_count
        predict_call_count += 1
        return {
            "gender": {"prediction": "female", "confidence": 0.92},
            "age_bracket": {"prediction": "18-30", "confidence": 0.78}
        }

    monkeypatch.setattr("app.audio.stream_session.VadTrigger", MockVad)
    monkeypatch.setattr("app.audio.stream_session.Diarizer", MockDiarizer)
    monkeypatch.setattr("app.audio.stream_session.predict", mock_predict)

    session = StreamSession()

    # First burst -> Agent (should skip inference)
    chunk1 = np.zeros(16000, dtype=np.float32)
    events1 = session.process_chunk(chunk1)
    assert events1[0]["role"] == "Agent"
    assert predict_call_count == 0  # Still not called

    # Second burst -> Customer (should run inference)
    chunk2 = np.zeros(16000, dtype=np.float32)
    events2 = session.process_chunk(chunk2)

    assert len(events2) == 1
    event = events2[0]

    assert event["role"] == "Customer"
    assert event["speaker_id"] == "customer_spk"
    assert event["gender"] == "female"
    assert event["age_bracket"] == "18-30"
    assert event["confidence_gender"] == 0.92
    assert event["confidence_age"] == 0.78

    # predict() should have been called exactly once (for the Customer)
    assert predict_call_count == 1

