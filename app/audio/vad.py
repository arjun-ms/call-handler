import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_speech(audio: np.ndarray, sr: int, aggressiveness: int = 2, max_speech_sec: float | None = None) -> np.ndarray:
    """Extract speech segments from audio using Voice Activity Detection.

    Tries webrtcvad first (more accurate), falls back to energy-based VAD.
    Returns only the voiced portions concatenated together, up to max_speech_sec limit.
    """
    from app.core.config import settings
    if max_speech_sec is None:
        max_speech_sec = settings.max_inference_speech_sec

    try:
        import webrtcvad

        return _webrtcvad_extract(audio, sr, aggressiveness, max_speech_sec)
    except (ImportError, Exception) as e:
        logger.warning("webrtcvad unavailable (%s), using energy-based VAD", e)
        return _energy_vad(audio, sr, max_speech_sec=max_speech_sec)


def _energy_vad(audio: np.ndarray, sr: int, threshold: float = 0.02, max_speech_sec: float = 7.0) -> np.ndarray:
    """Simple energy-based voice activity detection."""
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    max_frames = int((max_speech_sec * 1000) / frame_ms)
    voiced = []

    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i : i + frame_len]
        if np.sqrt(np.mean(frame**2)) > threshold:
            voiced.append(frame)
            if len(voiced) >= max_frames:
                logger.info("Reached maximum inference speech limit (%.1fs)", max_speech_sec)
                break

    if not voiced:
        logger.warning("No speech detected by energy VAD, returning full audio")
        return audio

    result = np.concatenate(voiced)
    logger.info(
        "Energy VAD: kept %.1fs of %.1fs (%.0f%%)",
        len(result) / sr,
        len(audio) / sr,
        100 * len(result) / len(audio),
    )
    return result


def _webrtcvad_extract(audio: np.ndarray, sr: int, aggressiveness: int, max_speech_sec: float = 7.0) -> np.ndarray:
    """WebRTC-based voice activity detection (more accurate)."""
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)

    # webrtcvad requires 16-bit PCM at 8/16/32/48 kHz
    pcm = (audio * 32767).astype(np.int16).tobytes()
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    frame_bytes = frame_len * 2  # 16-bit = 2 bytes per sample
    max_frames = int((max_speech_sec * 1000) / frame_ms)

    voiced = []
    for i in range(0, len(pcm) - frame_bytes, frame_bytes):
        frame = pcm[i : i + frame_bytes]
        if vad.is_speech(frame, sr):
            start = i // 2
            voiced.append(audio[start : start + frame_len])
            if len(voiced) >= max_frames:
                logger.info("Reached maximum inference speech limit (%.1fs)", max_speech_sec)
                break

    if not voiced:
        logger.warning("No speech detected by webrtcvad, returning full audio")
        return audio

    result = np.concatenate(voiced)
    logger.info(
        "WebRTC VAD: kept %.1fs of %.1fs (%.0f%%)",
        len(result) / sr,
        len(audio) / sr,
        100 * len(result) / len(audio),
    )
    return result
