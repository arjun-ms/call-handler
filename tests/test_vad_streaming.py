import pytest
import numpy as np
from app.audio.vad import VadTrigger

class MockVadTrigger(VadTrigger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_speech = False
        
    def _is_speech(self, frame: np.ndarray) -> bool:
        return self.force_speech

def test_vad_trigger_basic():
    sample_rate = 16000
    trigger = MockVadTrigger(sample_rate=sample_rate, frame_duration_ms=30, silence_duration_ms=500, aggressiveness=2)
    
    # Generate 1.0s audio
    audio = np.zeros(sample_rate, dtype=np.float32)
    
    # Send silence
    trigger.force_speech = False
    bursts = trigger.process_audio(audio)
    assert len(bursts) == 0
    
    # Send speech
    trigger.force_speech = True
    bursts = trigger.process_audio(audio)
    assert len(bursts) == 0
    
    # Send silence again - should emit after 500ms (17 frames of 30ms)
    trigger.force_speech = False
    bursts = trigger.process_audio(audio)
    
    assert len(bursts) == 1
    # 1s audio = 33 frames (15840 samples) + 160 remainder in buffer.
    # 33 speech frames + 16 trailing silence frames = 49 frames.
    expected_frames = 33 + 16
    assert len(bursts[0]) == expected_frames * trigger.frame_len

def test_vad_trigger_small_chunks():
    sample_rate = 16000
    trigger = MockVadTrigger(sample_rate=sample_rate, frame_duration_ms=30, silence_duration_ms=500, aggressiveness=2)
    
    audio = np.zeros(sample_rate, dtype=np.float32)
    
    # Process in tiny 100-sample chunks
    bursts = []
    trigger.force_speech = True
    for i in range(0, len(audio), 100):
        chunk = audio[i:i+100]
        bursts.extend(trigger.process_audio(chunk))
        
    trigger.force_speech = False
    for i in range(0, len(audio), 100):
        chunk = audio[i:i+100]
        bursts.extend(trigger.process_audio(chunk))
        
    assert len(bursts) == 1
    expected_frames = 33 + 16
    assert len(bursts[0]) == expected_frames * trigger.frame_len

