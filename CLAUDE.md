# CLAUDE.md - Development Guide

This document provides project-specific context and rules for development.

## 🛠️ Environment & Build
- **Tech Stack:** Python 3.11, FastAPI, FFmpeg, ONNX/audonnx.
- **Setup:** `pip install -r requirements.txt`
- **Run Server:** `python -m uvicorn app.main:app --reload`
- **Docker:** `docker compose up --build`

## 🧪 Test Commands
- **Run all tests:** `pytest`
- **Run audio tests:** `pytest tests/test_audio.py`
- **Run API tests:** `pytest tests/test_api.py`
- **Run CREMA-D evaluation:** `pytest tests/test_crema_d.py -s`
- **Run CommonVoice eval:** `python scripts/eval_commonvoice.py --dir <path> --tsv <path>`

## 📋 Project Tracking & Rules
- **Task List:** Always refer to and update [wiki/TASKS.md](file:///e:/Work/call-handler-classification/wiki/TASKS.md).
- **History:** Keep track of progress in the [changelog/](file:///e:/Work/call-handler-classification/changelog/) directory.
- **Rule:** Every significant change or feature implementation MUST be:
    1. Marked as completed in [TASKS.md](file:///e:/Work/call-handler-classification/wiki/TASKS.md).
    2. Documented in a new timestamped file in `changelog/` (e.g., `YYYY-MM-DD_HH-MM-SS.md`) explaining what worked, what didn't, and why.

## 📐 Architecture
- **Standardization:** All audio is normalized to 16kHz mono PCM.
- **VAD:** Mandatory for all inference to improve latency (max 7s speech extraction).
- **Inference:** Primary is ONNX (`audonnx`), fallback is `transformers`.
- **Docs:** See [wiki/ARCHITECTURE.md](file:///e:/Work/call-handler-classification/wiki/ARCHITECTURE.md) for details.
