import numpy as np
import librosa

def decode_burst(raw: bytes, sample_rate: int, encoding: str) -> np.ndarray:
    """
    Decodes a burst of raw audio bytes into a normalized float32 numpy array.
    Currently only supports 'pcm_s16le' encoding.
    Resamples the audio to 16,000 Hz if the input sample rate is different.
    
    Args:
        raw: Raw bytes of the audio stream.
        sample_rate: The sample rate of the input bytes.
        encoding: The encoding format, e.g., 'pcm_s16le'.
        
    Returns:
        np.ndarray: Audio data as float32 in the range [-1.0, 1.0], 16kHz.
    """
    if encoding != "pcm_s16le":
        raise ValueError(f"Unsupported encoding: {encoding}. Only 'pcm_s16le' is supported.")
    
    if len(raw) == 0:
        return np.array([], dtype=np.float32)

    # Convert raw bytes (16-bit PCM little-endian) to int16 numpy array
    try:
        audio_int16 = np.frombuffer(raw, dtype=np.int16)
    except ValueError as e:
        raise ValueError(f"Invalid byte stream for {encoding}: {e}")
        
    # Normalize to float32 between -1.0 and 1.0
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    # Resample to 16kHz if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        # librosa.resample expects float32
        audio_float32 = librosa.resample(audio_float32, orig_sr=sample_rate, target_sr=target_sample_rate)
        
    return audio_float32
