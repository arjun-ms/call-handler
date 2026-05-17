import pytest
import numpy as np
from app.inference.diarizer import Diarizer

def test_diarizer_mock_behavior():
    # If we force mock by passing something, or we can just test the logic 
    # of the mock behavior if speechbrain is disabled, but here we can just 
    # instantiate it and see what it does.
    diarizer = Diarizer()
    
    # Generate some dummy audio
    audio1 = np.random.randn(16000).astype(np.float32)
    audio2 = np.random.randn(16000).astype(np.float32)
    
    # Process first burst
    spk1 = diarizer.process_burst(audio1)
    assert spk1 == "speaker_0"
    
    # If we pass same audio, it might cluster to same speaker or diff based on threshold.
    # Since audio1 is random noise, the embedding is deterministic.
    # It should cluster to a speaker ID
    spk1_again = diarizer.process_burst(audio1)
    assert spk1_again in ["speaker_0", "speaker_1"]
    
    # Check that it handles basic processing without crashing.
    spk2 = diarizer.process_burst(audio2)
    assert spk2 in ["speaker_0", "speaker_1", "speaker_2"]

def test_diarizer_empty_burst():
    diarizer = Diarizer()
    assert diarizer.process_burst(np.array([], dtype=np.float32)) == "unknown"
