import pytest
import numpy as np
import soundfile as sf


@pytest.fixture
def sample_wav(tmp_path):
    """Generate a 3-second sine wave WAV at 16kHz (simulates voice)."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix of frequencies to simulate voice-like audio
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 600 * t)
    ).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture
def silent_wav(tmp_path):
    """Generate a 3-second silent WAV file."""
    sr = 16000
    audio = np.zeros(int(sr * 3), dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture
def short_wav(tmp_path):
    """Generate a 0.1-second WAV file (too short)."""
    sr = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "short.wav"
    sf.write(str(path), audio, sr)
    return str(path)


@pytest.fixture
def noisy_wav(tmp_path):
    """Generate a noisy audio file (simulates degraded conditions)."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.1 * np.sin(2 * np.pi * 200 * t)
    noise = 0.4 * np.random.randn(len(t))
    audio = (signal + noise).astype(np.float32)
    path = tmp_path / "noisy.wav"
    sf.write(str(path), audio, sr)
    return str(path)
