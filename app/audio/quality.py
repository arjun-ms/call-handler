import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def assess_quality(audio: np.ndarray, sr: int) -> Literal["good", "degraded", "insufficient"]:
    """Assess audio quality based on volume, clipping, silence, and SNR.

    Returns:
        "good" — clean audio suitable for inference
        "degraded" — noisy but usable, results may be less reliable
        "insufficient" — too poor for meaningful inference
    """
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    silence_ratio = float(np.mean(np.abs(audio) < 0.01))

    # Completely silent or nearly silent audio
    if rms < 0.005 or silence_ratio > 0.95:
        logger.warning("Audio quality: insufficient (rms=%.4f, silence=%.2f)", rms, silence_ratio)
        return "insufficient"

    issues = []

    if peak > 0.99:
        issues.append("clipping")
    if rms < 0.02:
        issues.append("low_volume")
    if silence_ratio > 0.7:
        issues.append("mostly_silent")

    snr = estimate_snr(audio, sr)
    if snr < 5:
        issues.append("low_snr")

    if issues:
        logger.info("Audio quality: degraded (%s)", ", ".join(issues))
        return "degraded"

    return "good"


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate signal-to-noise ratio in dB using frame energy analysis.

    Compares energy of the loudest vs quietest frames as a proxy for SNR.
    """
    frame_len = int(0.025 * sr)  # 25ms frames
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]

    if not frames:
        return 0.0

    energies = [float(np.mean(f**2)) for f in frames]
    energies.sort()

    n = max(1, len(energies) // 4)
    noise_energy = np.mean(energies[:n]) + 1e-10
    signal_energy = np.mean(energies[-n:]) + 1e-10

    return float(10 * np.log10(signal_energy / noise_energy))
