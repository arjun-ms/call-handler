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


class VadTrigger:
    """
    Stateful Voice Activity Detection for processing streaming audio.
    Accumulates frames and emits a complete burst when a period of silence is detected.
    """
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, silence_duration_ms: int = 500, aggressiveness: int = 2, max_burst_sec: float = 7.0):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_duration_ms = silence_duration_ms
        
        self.frame_len = int(sample_rate * frame_duration_ms / 1000)
        self.silence_frames_threshold = int(silence_duration_ms / frame_duration_ms)
        self.max_burst_frames = int((max_burst_sec * 1000) / frame_duration_ms) if max_burst_sec else None
        
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self._use_webrtc = True
        except ImportError:
            self.vad = None
            self._use_webrtc = False
            logger.warning("webrtcvad unavailable, using energy-based VAD for VadTrigger")
            
        self.buffer = np.array([], dtype=np.float32)
        self.in_speech = False
        self.silence_counter = 0
        self.current_burst = []

    def _is_speech(self, frame: np.ndarray) -> bool:
        if self._use_webrtc:
            pcm = (frame * 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm, self.sample_rate)
        else:
            return np.sqrt(np.mean(frame**2)) > 0.02
            
    def process_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Processes streaming audio and returns a list of completed speech bursts.
        
        Args:
            audio: np.ndarray of float32 samples.
            
        Returns:
            list[np.ndarray]: A list of completed speech bursts.
        """
        if len(audio) > 0:
            self.buffer = np.concatenate([self.buffer, audio])
            
        bursts = []
        
        while len(self.buffer) >= self.frame_len:
            frame = self.buffer[:self.frame_len]
            self.buffer = self.buffer[self.frame_len:]
            
            is_speech = self._is_speech(frame)
            
            if is_speech:
                self.in_speech = True
                self.silence_counter = 0
                self.current_burst.append(frame)
                
                # Enforce max duration
                if self.max_burst_frames and len(self.current_burst) >= self.max_burst_frames:
                    bursts.append(np.concatenate(self.current_burst))
                    self.current_burst = []
                    self.in_speech = False
                    self.silence_counter = 0
            else:
                if self.in_speech:
                    self.current_burst.append(frame)  # Keep silence in the tail for natural sound
                    self.silence_counter += 1
                    
                    if self.silence_counter >= self.silence_frames_threshold:
                        # Burst complete
                        bursts.append(np.concatenate(self.current_burst))
                        self.current_burst = []
                        self.in_speech = False
                        self.silence_counter = 0

        return bursts

    def flush(self) -> list[np.ndarray]:
        """Flush any remaining speech burst in the buffer.
        
        Call this when the audio stream ends to emit the final burst
        that hasn't been terminated by a silence gap.
        """
        bursts = []
        if self.current_burst:
            bursts.append(np.concatenate(self.current_burst))
            self.current_burst = []
            self.in_speech = False
            self.silence_counter = 0
        return bursts
