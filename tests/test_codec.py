import pytest
import numpy as np
from app.audio.codec import decode_burst

def test_decode_burst_valid_pcm_16khz():
    # Generate 1 second of a 440Hz sine wave at 16000Hz sample rate
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    # Amplitude in int16 range
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    raw_bytes = audio_data.tobytes()
    
    decoded = decode_burst(raw_bytes, sample_rate=sample_rate, encoding="pcm_s16le")
    
    # Check shape, dtype and values
    assert decoded.dtype == np.float32
    assert decoded.shape == (sample_rate,)
    
    # Int16 -> float32 scales by dividing by 32768.0
    expected_max = 10000 / 32768.0
    assert np.isclose(np.max(decoded), expected_max, atol=1e-3)

def test_decode_burst_valid_pcm_8khz_resampling():
    sample_rate = 8000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    raw_bytes = audio_data.tobytes()
    
    # decode_burst should resample to 16000Hz
    decoded = decode_burst(raw_bytes, sample_rate=sample_rate, encoding="pcm_s16le")
    
    assert decoded.dtype == np.float32
    # 1 second of audio should be 16000 samples after resampling
    assert decoded.shape == (16000,)
    
def test_decode_burst_unsupported_encoding():
    with pytest.raises(ValueError, match="Unsupported encoding"):
        decode_burst(b"dummy", sample_rate=16000, encoding="opus")

def test_decode_burst_invalid_bytes():
    # Odd number of bytes cannot be parsed as int16
    invalid_bytes = b"123" 
    with pytest.raises(ValueError, match="Invalid byte stream"):
        decode_burst(invalid_bytes, sample_rate=16000, encoding="pcm_s16le")

def test_decode_burst_empty_bytes():
    decoded = decode_burst(b"", sample_rate=16000, encoding="pcm_s16le")
    assert decoded.shape == (0,)
    assert decoded.dtype == np.float32
