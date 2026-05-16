# Voice Attribute Inference Service Architecture

This document outlines the detailed architecture and request lifecycle of the Voice Attribute Inference Service.

## Request Lifecycle

```mermaid
flowchart TD
    %% Styling
    classDef external fill:#f9f,stroke:#333,stroke-width:2px;
    classDef api fill:#bbf,stroke:#333,stroke-width:2px;
    classDef processing fill:#bfb,stroke:#333,stroke-width:2px;
    classDef ml fill:#fbf,stroke:#333,stroke-width:2px;
    classDef error fill:#fbb,stroke:#333,stroke-width:2px;

    %% Actors / Externals
    Client([Client Application]):::external
    HFHub[(Hugging Face Hub)]:::external

    %% API Layer
    subgraph API["API Layer (FastAPI)"]
        Middleware[Request Timing & Logging Middleware]:::api
        RouterHTTP[POST /v1/infer]:::api
        RouterWS[WS /ws/infer]:::api
        
        Middleware --> RouterHTTP
        Middleware --> RouterWS
    end

    %% Audio Processing Pipeline
    subgraph AudioPipeline["Audio Processing Pipeline"]
        Normalize["Preprocessing (ffmpeg)<br/>Convert to 16kHz Mono PCM"]:::processing
        Validate["Duration Validation<br/>(Min/15s Max bounds)"]:::processing
        Quality["Quality Assessment<br/>(SNR, Clipping, Volume)"]:::processing
        VAD["Voice Activity Detection<br/>(webrtcvad / Energy Fallback)"]:::processing
    end

    %% ML Inference Engine
    subgraph InferenceEngine["ML Inference Engine"]
        Pipeline["Inference Coordinator"]:::ml
        Wav2Vec["Wav2Vec2 Processor & Model<br/>(audeering/wav2vec2-large-robust-6-ft-age-gender)"]:::ml
        Heads["Gender & Age Parsers"]:::ml
    end

    %% Flow
    Client -- "Multipart Audio Upload" --> Middleware
    Client -- "WebSocket Byte Stream + 'PROCESS'" --> Middleware

    RouterHTTP --> Normalize
    RouterWS -- "Buffered Bytes" --> Normalize

    Normalize --> Validate
    Validate -- "Valid" --> Quality
    Quality -- "Sufficient/Degraded" --> VAD
    
    Validate -- "Too Short/Long" --> ErrorResponse["Return 400 Bad Request"]:::error
    Quality -- "Insufficient" --> ErrorResponse422["Return 422 Unprocessable Entity"]:::error

    VAD -- "Extracted Speech Segments" --> Pipeline
    Pipeline --> Wav2Vec
    Wav2Vec -. "Download Weights (First Run)" .-> HFHub
    Wav2Vec -- "Hidden States / Logits" --> Heads
    
    Heads -- "Parsed Predictions" --> ResponseFormatter["Format JSON Response"]:::api
    
    ResponseFormatter -- "JSON Result" --> Client
    ErrorResponse --> Client
    ErrorResponse422 --> Client
```

## Component Details

1. **FastAPI Server (`app/main.py` & `app/api/`)**: Handles incoming HTTP and WebSocket requests. Responsible for request-level validation, logging, and performance timing (via middleware).
2. **Audio Preprocessing (`app/audio/preprocess.py`)**: Accepts various audio formats (MP3, WAV, OGG, WebM) and uses `ffmpeg` to securely and efficiently standardize them to the 16kHz Mono PCM format required by the ML model.
3. **Audio Quality (`app/audio/quality.py`)**: Calculates Signal-to-Noise Ratio (SNR), clipping occurrences, and silence ratios. It tags the request with metadata ("good", "degraded", "insufficient") allowing the API to reject completely unusable audio early.
4. **Voice Activity Detection (`app/audio/vad.py`)**: Employs `webrtcvad` to trim non-speech segments from the normalized audio, ensuring the ML model only processes segments containing actual human voice. It includes a graceful fallback to a simpler energy-based VAD if webrtc compilation fails in certain environments.
5. **Inference Engine (`app/inference/pipeline.py`)**: Loads the heavy `wav2vec2` transformer models into memory once at startup (or on the first request). It processes the speech segments and interprets the resultant output logits into structured human-readable JSON formats containing `prediction` strings and `confidence` scores (0.0 to 1.0) for both Gender and Age bracket.

## Future Roadmap
- [ ] GPU-accelerated inference support
- [ ] Multilingual/Accent detection support
- [ ] Speaker Diarization (Agent vs Customer separation)
- [ ] Real-time progressive inference for WebSockets
