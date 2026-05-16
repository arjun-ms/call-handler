from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_prefix="VOICE_",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Voice Attribute Inference Service"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    max_audio_duration_sec: float = 600.0  # Allow up to 10 minute files
    min_audio_duration_sec: float = 0.5
    # Inference settings
    min_inference_speech_sec: float = 2.0
    max_inference_speech_sec: float = 15.0  # Increased for better baseline accuracy without diarization
    batch_size: int = 1
    max_upload_size_mb: int = 50
    confidence_threshold: float = 0.4
    model_name: str = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    sample_rate: int = 16000


settings = Settings()
