# Voice Attribute Inference - Task Checklist

This document tracks the progress of the Voice Attribute Inference service implementation.

## 🏗️ Core Infrastructure
- [x] Project scaffolding and directory structure
- [x] Configuration management via `pydantic-settings`
- [x] Structured logging setup
- [x] Request-ID and Timing middleware for observability
- [x] Health check endpoint (`/health`)
- [x] Pydantic response models and schemas

## 🔊 Audio Pipeline
- [x] FFmpeg normalization (16kHz, mono PCM)
- [x] Audio quality assessment (Clipping, Silence, Low Volume)
- [x] Signal-to-Noise Ratio (SNR) estimation
- [x] Voice Activity Detection (VAD) integration (`webrtcvad`)
- [x] Energy-based VAD fallback
- [x] **Early Stopping:** VAD-based speech extraction limit (`max_inference_speech_sec`)

## 🧠 ML Inference
- [x] ONNX-based model loading with `audonnx` (Faster inference)
- [x] Transformers-based fallback for model loading
- [x] Gender classification (Male, Female, Child)
- [x] Age bracket classification (18-30, 31-45, 46-60, 60+)
- [x] Confidence score aggregation and fallback to `unknown`

## 🌐 API & Integration
- [x] HTTP POST `/analyze` endpoint for file uploads/raw stream
- [x] WebSocket `/ws/analyze` endpoint for streaming audio
- [x] Error handling for corrupted files and unsupported formats
- [x] Multi-format support (MP3, WAV, WebM, Opus, etc.)

## 🧪 Testing & Evaluation
- [x] Unit tests for audio preprocessing and quality checks
- [x] Integration tests for API endpoints
- [x] Mozilla CommonVoice evaluation script (`scripts/eval_commonvoice.py`)
- [x] CREMA-D dataset evaluation harness (`tests/test_crema_d.py`)
- [x] CommonPhone dataset evaluation (`tests/test_eval_datasets.py`) — 100% gender accuracy
- [x] AppTek Call-Center dataset download + evaluation — OOM on long audio (fixed via audio capping)
- [x] Confidence calibration and accuracy metrics in eval script
- [x] Sample audio file and instructions for smoke testing

## 🐳 Deployment & Documentation
- [x] Dockerfile with FFmpeg and model dependencies
- [x] Docker Compose setup
- [x] README with setup instructions and API docs
- [x] README section: Design decisions, model choice rationale, and known limitations
- [x] README section: Privacy handling documentation (treating audio as PII)
- [x] Design write-up (200 words on approach, improvements, scaling to 1,000 concurrent calls)
- [x] Maintain project changelog in `changelog/` folder
- [x] Initialize and maintain `CLAUDE.md` for project rules
- [ ] Deployment to staging/production environment
- [x] API versioning strategy (moved to assignment-mandated top-level endpoints /analyze)

## 🚀 Future Improvements
- [ ] GPU-accelerated inference support
- [ ] Multilingual/Accent detection support
- [x] Speaker Diarization & Role Labeling (Agent vs Customer separation)
- [x] Real-time progressive inference for WebSockets
- [ ] Model fine-tuning on noisy logistics-call datasets
