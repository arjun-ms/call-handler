"""Tests for audio preprocessing, quality assessment, and VAD."""

import numpy as np
import soundfile as sf

from app.audio.preprocess import normalize_audio, load_audio, validate_duration
from app.audio.quality import assess_quality, estimate_snr
from app.audio.vad import extract_speech


class TestPreprocessing:
    """Audio preprocessing tests."""

    def test_normalize_and_load(self, sample_wav):
        """Normalized audio should be 16kHz mono."""
        normalized = normalize_audio(sample_wav)
        audio, sr = load_audio(normalized)
        assert sr == 16000
        assert len(audio) > 0
        assert audio.ndim == 1

    def test_validate_duration_valid(self, sample_wav):
        """3-second audio should pass validation."""
        audio, sr = load_audio(sample_wav)
        duration = validate_duration(audio, sr)
        assert 2.5 < duration < 3.5

    def test_validate_duration_too_short(self, short_wav):
        """0.1-second audio should fail validation."""
        audio, sr = load_audio(short_wav)
        try:
            validate_duration(audio, sr)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too short" in str(e)


class TestQualityAssessment:
    """Audio quality assessment tests."""

    def test_good_quality(self, sample_wav):
        """Clean sine wave should be rated 'good'."""
        audio, sr = load_audio(sample_wav)
        quality = assess_quality(audio, sr)
        assert quality in ("good", "degraded")

    def test_silent_audio(self, silent_wav):
        """Silent audio should be rated 'insufficient'."""
        audio, sr = load_audio(silent_wav)
        quality = assess_quality(audio, sr)
        assert quality == "insufficient"

    def test_noisy_audio(self, noisy_wav):
        """Noisy audio should be rated 'degraded'."""
        audio, sr = load_audio(noisy_wav)
        quality = assess_quality(audio, sr)
        assert quality in ("degraded", "insufficient")

    def test_snr_estimation(self, sample_wav):
        """Clean audio should have positive SNR."""
        audio, sr = load_audio(sample_wav)
        snr = estimate_snr(audio, sr)
        assert snr > 0


class TestVAD:
    """Voice activity detection tests."""

    def test_extract_speech_from_signal(self, sample_wav):
        """VAD should extract non-silent segments."""
        audio, sr = load_audio(sample_wav)
        speech = extract_speech(audio, sr)
        assert len(speech) > 0

    def test_extract_speech_from_silence(self, silent_wav):
        """VAD on silence should return original audio (fallback)."""
        audio, sr = load_audio(silent_wav)
        speech = extract_speech(audio, sr)
        # Should return the full audio as fallback
        assert len(speech) == len(audio)
