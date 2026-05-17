# /analyze Endpoint Flow Documentation

This document describes the detailed processing flow of the `POST /analyze` endpoint in the Voice Attribute Inference Service.

## Overview
The `/analyze` endpoint accepts an audio file containing a conversation (potentially with multiple speakers, e.g., an Assistant and a User), isolates the speech of the customer, and infers their age and gender.

## Detailed Processing Steps

### 1. Ingestion & Validation
- **Input:** Multipart form-data with the audio file.
- **Validation:** 
  - Checks if the file extension is supported (WAV, MP3, WebM, Opus, etc.).
  - Verifies that the file size is within the allowed limit (default 10MB).
- **Storage:** Saves the raw uploaded file to a temporary location on disk.

### 2. Audio Normalization (FFmpeg)
- Calls `ffmpeg` to convert the input file into a standardized format required by the inference pipeline:
  - **Sample Rate:** 16,000 Hz
  - **Channels:** 1 (Mono) - *Downmixes stereo files to mono.*
  - **Codec:** PCM 16-bit WAV

### 3. Loading & Duration Verification
- Loads the normalized WAV file into memory as a NumPy array.
- Verifies that the audio duration falls within the acceptable range (e.g., 0.5s to 30s).

### 4. Audio Quality Assessment
- Analyzes the audio array for potential issues such as low volume, clipping, and low Signal-to-Noise Ratio (SNR).
- Assigns a quality flag: `"good"`, `"degraded"`, or `"insufficient"`.
- If the quality is `"insufficient"`, the request is aborted with a `422 Unprocessable Entity` error.

### 5. VAD Segmentation
- Uses `VadTrigger` (based on `webrtcvad` or energy fallback) to split the audio into discrete **speech bursts** separated by silence.
- This step ensures that we have separate segments to analyze for speaker identification.

### 6. Speaker Diarization
- Iterates through each detected speech burst.
- Passes each burst to the `Diarizer` (using SpeechBrain ECAPA-TDNN embeddings or mock clustering).
- Assigns a speaker ID (e.g., `"speaker_0"`, `"speaker_1"`) to each burst based on voice similarity.

### 7. Role Labeling (Agent-First Heuristic)
- Applies a heuristic to differentiate the Assistant from the Customer:
  - The **first speaker** detected in the call is assumed to be the **Assistant/Agent** (typical for logistics calls where the agent introduces themselves).
  - Any **other speaker** is assumed to be the **Customer/User**.
  - If only one speaker is detected in the entire file, the system assumes that person is the Customer.

### 8. Targeted Inference
- Concatenates only the speech bursts belonging to the identified **Customer**.
- Passes this "Customer-only" audio array to the ML model (`Wav2Vec2`) for gender and age inference.
- This prevents the Assistant's voice from skewing the results.

### 9. Response Generation
- Calculates the total processing time.
- Returns the structured JSON response containing the predictions for the Customer, confidence scores, and the audio quality flag.

## API Contract Reference
```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "male" | "female" | "unknown",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "18-30" | "31-45" | "46-60" | "60+" | "unknown",
    "confidence": 0.63
  },
  "processing_ms": 142,
  "audio_quality": "good" | "degraded" | "insufficient"
}
```
