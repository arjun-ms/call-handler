import subprocess
import tempfile
import os
import logging

import numpy as np
import soundfile as sf

from app.core.config import settings

logger = logging.getLogger(__name__)


def normalize_audio(input_path: str) -> str:
    """Convert any audio format to 16kHz mono PCM WAV using ffmpeg.

    Returns path to the normalized temporary WAV file.
    Caller is responsible for cleanup via cleanup_temp().
    """
    output_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        str(settings.sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-loglevel",
        "error",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=10)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Audio conversion failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — install ffmpeg to process audio")

    return output_path


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file and return (waveform, sample_rate).

    Ensures mono output by averaging channels if stereo.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def validate_duration(audio: np.ndarray, sr: int) -> float:
    """Check audio duration is within configured bounds.

    Returns duration in seconds.
    Raises ValueError if out of bounds.
    """
    duration = len(audio) / sr
    if duration < settings.min_audio_duration_sec:
        raise ValueError(
            f"Audio too short: {duration:.1f}s (minimum {settings.min_audio_duration_sec}s)"
        )
    if duration > settings.max_audio_duration_sec:
        raise ValueError(
            f"Audio too long: {duration:.1f}s (maximum {settings.max_audio_duration_sec}s)"
        )
    return duration


def cleanup_temp(path: str | None):
    """Safely delete a temporary file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
