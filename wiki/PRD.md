# Backend Engineering Assignment Plan
## Voice Attribute Inference Service

---

# Goal

Build a low-latency backend service that:
- accepts audio input (HTTP + optional WebSocket streaming)
- estimates:
  - gender
  - age bracket
- returns:
  - confidence scores
  - audio quality assessment
- works reliably under noisy logistics-call conditions
- runs fully in Docker with no external runtime dependencies

Target latency:
- under 500ms inference for ~5 second audio chunks

---

# High-Level Approach

The system will prioritize:
- reliability
- graceful degradation
- low latency
- explainability
- production-minded architecture

Instead of attempting highly accurate demographic prediction, the service will:
- provide broad classifications
- surface confidence scores
- return `unknown` when confidence is insufficient
- explicitly detect degraded audio conditions

This aligns with real-world logistics call environments where audio quality is often inconsistent.

---

# Tech Stack

## Backend
- FastAPI
- Uvicorn
- Python 3.11

## Audio Processing
- ffmpeg
- librosa
- numpy
- scipy
- webrtcvad

## ML / Inference
- SpeechBrain pretrained models
- HuggingFace audio classifiers (if needed)

## Observability
- Python logging
- timing middleware
- request IDs

## Deployment
- Docker
- Docker Compose

---

# Core Features

---

# 1. Audio Ingestion

## HTTP Endpoint (Primary MVP)

### Endpoint
```http
POST /analyze
```

### Input

Multipart audio upload.

Supported formats:

* wav
* mp3
* webm
* opus
* m4a

### Internal Standardization

All audio will be converted to:

```text
16kHz mono PCM WAV
```

Reason:

* simplifies downstream processing
* improves model consistency
* codec-independent pipeline

---

# 2. Audio Preprocessing

## Steps

### a. Audio normalization

* sample rate conversion
* mono conversion
* volume normalization

### b. Voice Activity Detection (VAD)

Using:

* webrtcvad

Purpose:

* remove silence
* reduce inference noise
* improve latency

### c. Quality Assessment

Detect:

* excessive silence
* clipping
* low volume
* poor signal-to-noise ratio

Output:

```json
"audio_quality": "good"
```

or:

```json
"audio_quality": "degraded"
```

or:

```json
"audio_quality": "insufficient"
```

This is important because logistics calls may contain:

* truck noise
* warehouse noise
* road interference
* compressed telephony codecs

---

# 3. Attribute Inference

---

# Gender Prediction

## Strategy

Use pretrained speech embeddings and classifier.

Output:

```json
{
  "prediction": "male",
  "confidence": 0.91
}
```

Fallback:

```json
{
  "prediction": "unknown",
  "confidence": 0.22
}
```

---

# Age Bracket Prediction

## Strategy

Predict broad age ranges instead of exact age.

Supported brackets:

* 18-30
* 31-45
* 46-60
* 60+
* unknown

Reason:

* age prediction from voice is inherently noisy
* broad buckets are more realistic and stable

Confidence-aware fallback will be used when prediction certainty is low.

---

# 4. API Response Format

## Example Response

```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "male",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.63
  },
  "processing_ms": 142,
  "audio_quality": "good"
}
```

---

# 5. WebSocket Streaming (Secondary / Bonus)

## Endpoint

```text
/ws/analyze
```

## Behavior

Client sends audio chunks incrementally.

Server:

* buffers chunks
* preprocesses incrementally
* optionally emits progressive predictions

Example partial response:

```json
{
  "partial": true,
  "gender": {
    "prediction": "male",
    "confidence": 0.74
  }
}
```

Final response:

```json
{
  "final": true,
  ...
}
```

---

# Architecture

```text
Client
   |
   | HTTP / WebSocket
   v
FastAPI API Layer
   |
   +--> Audio Validation
   |
   +--> ffmpeg Normalization
   |
   +--> Voice Activity Detection
   |
   +--> Audio Quality Analyzer
   |
   +--> Inference Pipeline
   |        |
   |        +--> Gender Predictor
   |        +--> Age Predictor
   |
   +--> Confidence Aggregation
   |
   +--> JSON Response
```

---

# Project Structure

```text
backend-assignment/
│
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── websocket.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── middleware.py
│   │
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── quality.py
│   │   ├── vad.py
│   │   └── codecs.py
│   │
│   ├── inference/
│   │   ├── gender.py
│   │   ├── age.py
│   │   ├── pipeline.py
│   │   └── models/
│   │
│   ├── schemas/
│   │   └── response.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_api.py
│   └── test_audio.py
│
├── scripts/
│   └── eval_commonvoice.py
│
├── sample_audio/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── PLAN.md
```

---

# Reliability & Error Handling

## Error Cases

Handle:

* unsupported codec
* corrupted audio
* insufficient speech
* silent audio
* invalid uploads

Example response:

```json
{
  "error": "insufficient_audio"
}
```

---

# Observability

## Logging

Log:

* request_id
* processing_ms
* audio_duration
* quality_flag
* confidence scores

Do NOT log:

* raw audio
* PII content

---

# Timing Middleware

Measure:

* preprocessing time
* inference time
* total request latency

---

# Privacy & Security

## Privacy Principles

Audio is treated as PII.

Implementation:

* no persistent audio storage
* temporary files deleted immediately
* no raw audio logging
* in-memory processing where possible

Production assumptions:

* HTTPS/TLS enabled
* container isolation
* restricted logging

---

# Dockerization

## Requirements

The service must run fully via Docker.

### Dockerfile includes:

* Python runtime
* ffmpeg
* model dependencies
* preloaded weights (if possible)

Command:

```bash
docker compose up --build
```

---

# Testing

## Minimum Test Coverage

### API Test

* upload sample audio
* validate response schema

### Audio Validation Test

* invalid format handling
* silence detection

---

# Evaluation Harness

## Optional Script

```bash
python scripts/eval_commonvoice.py
```

Purpose:

* run inference on Mozilla Common Voice samples
* print:

  * accuracy
  * confidence distribution
  * latency

---

# Scaling Strategy

---

# Phase 1

Single FastAPI instance.

CPU inference.

---

# Phase 2

Horizontal scaling behind load balancer.

Stateless architecture:

* no session storage
* no persistent audio

---

# Phase 3

Separate:

* API gateway
* streaming ingestion
* inference workers

Potential technologies:

* Redis Streams
* Kafka
* NATS

---

# Phase 4

GPU batching for transformer inference.

Batch multiple concurrent chunks together to improve throughput.

---

# Known Limitations

* age prediction from voice is approximate
* performance may degrade under extreme noise
* demographic prediction bias may exist
* English-centric pretrained models may generalize unevenly
* gender misclassification on minor/teenage speakers (under 18) due to acoustic overlap between young male and female voices
* gender misclassification on minor/teenage speakers (under ~18) -- younger female voices overlap acoustically with younger male voices, causing incorrect "male" predictions. Age bracket prediction remains accurate.
* gender misclassification on minor/teenage speakers (under ~18) — younger female voices often overlap acoustically with younger male voices, causing the model to predict "male" incorrectly. Age bracket prediction remains accurate in these cases.

---

# Future Improvements

* better calibration
* multilingual support
* accent/language detection
* speaker diarization
* GPU inference
* adaptive streaming inference
* model fine-tuning on telephony datasets

---

# 2-Day Execution Plan

---

# Day 1

## Morning

* FastAPI setup
* Docker setup
* HTTP upload endpoint

## Afternoon

* ffmpeg normalization
* VAD integration
* audio quality checks

## Evening

* integrate inference pipeline
* structured JSON response

---

# Day 2

## Morning

* WebSocket endpoint
* optional progressive inference

## Afternoon

* tests
* evaluation script
* logging/timing middleware

## Evening

* README
* architecture cleanup
* latency measurements
* final polish

---

# Key Engineering Principles

The system prioritizes:

* graceful degradation
* confidence-aware inference
* low operational complexity
* stateless scalability
* privacy-first processing

The focus is on building a production-minded backend service rather than a research-grade demographic classifier.