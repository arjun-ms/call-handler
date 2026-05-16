# Backend Engineering Assignment

## Background

We build voice AI agents that handle inbound and outbound calls for logistics companies, coordinating deliveries, resolving exceptions, and communicating with drivers, dispatchers, and customers.

To personalize these interactions, our agents need to quickly infer caller attributes from audio without relying on any prior data about the contact. Your task is to design and implement a backend service that accepts an audio stream and returns estimated attributes for the contact person, specifically their **gender** and **age bracket**. 

## What You'll Build

**Task 1 - Audio ingestion**
Accept a streaming or chunked audio input over HTTP or WebSocket. Handle common logistics-world conditions such as noisy environments and compressed codecs.

**Task 2 - Attribute inference**
Estimate gender and age bracket from the audio. Use any approach: pretrained models, acoustic feature extraction, or a hybrid pipeline.

**Task 3 - REST / WebSocket API**
Expose a clean API that returns structured results with confidence scores. Must be low-latency and suitable for real-time voice calls.

**Task 4 - Reliability and ops**
Handle errors gracefully. Include basic observability hooks (logging, timing). Containerize the service with a working Dockerfile.

## Expected API Contract

`POST /analyze` - multipart audio upload or raw stream

```
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

## Constraints and Considerations

**Logistics context**
Calls often have background noise from trucks, warehouses, and road noise. Your pipeline should degrade gracefully and surface an `audio_quality` flag rather than silently returning bad predictions.

**Latency**
Target end-to-end inference under 500ms on a 5-second audio chunk.

**Privacy**
No audio should be stored beyond the duration of a request. Treat caller audio as PII and document how your service ensures this.

**Portability**
The service must run via `docker compose up` with no external dependencies other than publicly available model weights.

## Bonus Tasks

- **Real-time streaming** - WebSocket endpoint that emits progressive predictions as audio chunks arrive
- **Language / accent detection** - best-effort language or accent field in the response
- **Eval harness** - script that runs your model against a public dataset (e.g. Mozilla Common Voice) and prints accuracy and confidence calibration metrics

## Submission Requirements

**Repository**
Share a private GitHub repo. Include a `README.md` with setup instructions, design decisions, model choice rationale, and known limitations.

**Design write-up**
200 words on your approach: why you chose the model or library, how you would improve it with more time, and how you would scale this to 1,000 concurrent calls.

**Sample test**
At least one working test (unit or integration) and a sample audio file or instructions for sourcing one to run a smoke test.

## Suggested Tools

- pyannote.audio · SpeechBrain · openSMILE · librosa · Whisper · FastAPI · ffmpeg · Mozilla Common Voice · VoxCeleb

**You are free to use any library or approach. You are also free to use Claude or ChatGPT. We care more about your reasoning than which specific model or tool you pick.** 

**Make sure you have enough understanding of the concepts so that there can be an insightful discussion around your approach.**

Questions? Reach out before you start. We would rather clarify than have you lose time on an ambiguity.

**Submission deadline:** Within 2 days of receiving this