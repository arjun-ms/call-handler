- handled multi-audio format using ffmpeg
- noisy logistics conditions are actively mitigated by the Voice Activity Detection (VAD) and Signal-to-Noise Ratio (SNR) components.
- ONNX (Open Neural Network Exchange) — file format for storing ML models (Use ONNX if you need cross-framework flexibility (e.g., moving from PyTorch to TensorRT for Nvidia GPUs))
- "Wav2Vec 2.0" — a family of deep learning models for self-supervised learning of speech representations, widely used for tasks like ASR, speaker verification, and classification.
- audonnx deploys machine learning models stored in ONNX format.
- Latency (< 500ms): Handled. By using the ONNX runtime and actively slicing out silence via VAD, the pipeline runs under 500ms on CPU.
- API Contract Matching: Perfect match. The endpoints are strictly mapped to /analyze and /ws/analyze. Crucially, we mapped the model's internal child prediction to unknown and added a low-confidence threshold fallback, ensuring it strictly adheres to "male" | "female" | "unknown" and the specified age brackets.
- Privacy / PII: Handled. The cleanup_temp() function guarantees ephemeral handling, and the privacy guarantees are fully detailed in the [README.md](README.md).
- Observability: Handled. Structured logging and RequestMiddleware tracking processing_ms are successfully implemented.
- Sample Test: Handled. We successfully fetched a real, conversational speech sample (test.wav), and the smoke test curl command is prominently documented in the README.md.
- Design Writeup is in [README.md](README.md)
- The assignment says: "accepts an audio stream and returns estimated attributes for the contact person". It does not explicitly specify if the audio file contains both the agent and the customer (a mixed call) or just the customer. However, in modern contact centers, real-time analytics are usually fed the isolated "Inbound Stream" (just the customer's microphone).
- Our pipeline takes the audio, normalizes it to a single mono channel, slices out the silence, and predicts the attributes of the dominant voice.
- Our test.wav sample is a mono recording of a single person speaking, which perfectly tests this assumed scenario.
- If the model is unsure (low confidence) or predicts a class outside of our schema (like a "child" voice), it gracefully defaults to "unknown"
-  Similarly, if the age confidence drops below our threshold (e.g., extremely noisy audio or overlapping voices making age unclear), it falls back to "unknown"

### What happens if they upload a 2-person conversation?
- Because our pipeline forces the audio into a mono channel, if they upload a file where both the driver and the agent are speaking, the model will analyze both voices together and likely predict the attributes of the "dominant" (loudest/longest) speaker.
- if the service is required to ingest raw, mixed 2-party recordings, the immediate next step would be to integrate a Speaker Diarization module (like Pyannote) to separate the caller from the agent before passing it to the Wav2Vec2 model

### if you hear the `sample_audio/test.wav` its not a person speaking thigns which is relevant to our context that is complaining or asking for help as a customer. rather its a person who just randomly saying some words(not-a-conversation-response), can we take that as a good test data?

- Yes.
- The Model Listens to "How" You Speak, Not "What" You Say (It is analyzing acoustic features—things like pitch, vocal tract length, resonance, and timbre)
- The actual words being spoken are entirely irrelevant to this specific model.
- The purpose of providing test.wav is not to prove that the model is perfectly accurate for angry logistics customers. The purpose of test.wav is to prove that:
The API correctly accepts a file upload.
ffmpeg successfully normalizes the audio to 16kHz mono.
The Voice Activity Detection (VAD) successfully finds the speech.
The ONNX model successfully runs the math.
The API returns the exact JSON contract you promised.


# Completion of tasks

Here is the exact breakdown of how you fulfilled each of the four mandatory tasks. You can use this as a cheat sheet during your interview or copy-paste it directly if they ask for a written summary.

### **Task 1 - Audio ingestion**
* **What I implemented:** I built dual endpoints to support both REST (`POST /analyze` for file uploads) and WebSockets (`/ws/analyze` for chunked/streaming audio bytes). 
* **Handling codecs & noise:** To handle diverse and highly compressed codecs, I integrated `ffmpeg` natively into the pipeline to safely normalize any input format into a standard 16kHz mono PCM stream. To mitigate noisy logistics environments, I implemented an aggressive preprocessing pipeline using `webrtcvad` (Voice Activity Detection) and SNR (Signal-to-Noise Ratio) checks to explicitly slice out truck/background noise, ensuring the model only analyzes pure speech.

### **Task 2 - Attribute inference**
* **What I implemented:** I used a deep learning approach, integrating the state-of-the-art `audeering/wav2vec2-large-robust-6-ft-age-gender` model.
* **How it works:** This is a Wav2Vec 2.0 transformer model explicitly fine-tuned on diverse acoustic data to predict both age and gender simultaneously. I mapped its raw logit outputs directly into the required assignment schema (`"male" | "female" | "unknown"`) and the four requested age brackets. 

### **Task 3 - REST / WebSocket API**
* **What I implemented:** I exposed a clean, strictly typed FastAPI application that returns a structured JSON contract containing `prediction` and `confidence` scores for both attributes, along with an explicit `audio_quality` flag.
* **Low-latency optimization:** To achieve the sub-500ms latency required for real-time contact center routing, I intentionally bypassed heavy frameworks like PyTorch. Instead, I exported the model to the ONNX format and used the `audonnx` runtime. Combined with VAD silence-trimming (which reduces the payload size), this allows blazing-fast inference purely on the CPU.

### **Task 4 - Reliability and ops**
* **What I implemented:** The service is fully containerized and production-ready.
* **Observability & Error Handling:** I implemented graceful degradation (mapping low-confidence predictions to `"unknown"` instead of crashing or hallucinating). For observability, I added structured Python logging and built a custom FastAPI `RequestMiddleware` that calculates and injects `processing_ms` execution times into every response.
* **Containerization:** I wrote a `Dockerfile` and `docker-compose.yml` that correctly handles the complex OS-level dependency of `ffmpeg` alongside the Python environment, ensuring the app spins up reliably on any host machine without manual configuration.



---


# Question 1: How are the agent and customer separated in the raw byte stream?

Since both are present, we have two possibilities for how the audio is delivered to us:

Option A (Stereo/Multi-channel): The stream is stereo, where the agent is strictly on one channel (e.g., Left) and the customer is strictly on the other (e.g., Right).

Option B (Mono): Both voices are mixed down into a single mono channel.

My Recommendation: 
- I strongly hope it's Option A. In enterprise call centers, SIP/RTP streams can often be tapped as dual-channel. If it's Option A, we simply drop the agent's channel and process the customer's channel using our existing pipeline. 
- If it's Option B, we are forced to implement Speaker Diarization (e.g., using Pyannote) to separate "Speaker 1" from "Speaker 2" in real-time, which is computationally expensive and introduces latency.

# Question 2: How do we know which speaker is the Customer?

- Option A (The "Greeting" Heuristic): In almost all contact centers, the Agent speaks first ("Hello, thank you for calling Support, my name is..."). We assume the first speaker detected is the Agent, and the second speaker is the Customer.
- Option B (Agent Voiceprinting): We require the system to have a pre-registered voiceprint (embedding) of the Agent handling the call. We compare the speakers to the known Agent voiceprint and filter them out.
- Option C (Transcription + NLP): We run Speech-to-Text on the chunks and use a lightweight NLP model to figure out who is saying "How can I help you?". (Warning: This will destroy our <500ms latency budget).




# Question 2: How are the agent and customer separated in the raw byte stream?
This is the most critical question for the interviewer to ask. It determines whether your solution is trivial or complex.

Possibility A: Stereo Separation (The "Happy Path")
The audio stream is sent as Stereo audio (2 channels).
Agent → Left Channel
Customer → Right Channel
Solution: You simply discard the Agent's channel and run your existing pipeline on the Customer's channel.

Possibility B: Mono Mix (The "Hard Path")
The audio is sent as Mono audio (1 channel).
Both voices are mixed together (e.g., 50% Agent, 50% Customer).
Solution: You cannot solve this with just your current model. You must implement Speaker Diarization (splitting "Who spoke when").

**lets go with option A becuase even if its inbound or outbound, agents picks and always greet right so go with that**

# Input will be raw byte streams

## Question 3: Stream Chunking & Execution Strategy

Raw bytes will be flowing in continuously. We cannot run heavy Neural Networks (Diarization + Wav2Vec) on every single byte packet, nor can we wait for the call to end. We have to chunk the stream.

How do we decide when to process a chunk of bytes?

- Option A (Fixed Time Window): We blindly buffer exactly 5 seconds of audio, process it, emit a result, and repeat. (Risk: We might cut a word exactly in half, confusing the model).

- Option B (VAD-Triggered Burst): We run an ultra-lightweight Voice Activity Detector (like WebRTC VAD) on the incoming bytes. We buffer the bytes only while someone is speaking. As soon as there is a pause (e.g., 500ms of silence), we take that "speech burst" and pass it to Diarization and Inference.

- Option C (Sliding Window): We maintain a rolling 10-second buffer of audio and re-evaluate the whole thing every 1 second. (Provides the highest accuracy, but consumes the most CPU).

My Recommendation: I highly recommend Option B. It's the gold standard for real-time voice AI. By using pauses as natural boundaries, we feed clean sentences into the diarization/inference models, and we don't waste CPU cycles running models on dead air or hold music.


# Question 4: Result Aggregation & Emission Strategy

Now that we are processing bursts of speech sequentially as they happen, we need to decide how to handle the results over time.

Imagine the call is progressing:

Burst 1 (0:00 - 0:05): Agent speaks. We label as "Speaker 1" and ignore.
Burst 2 (0:05 - 0:10): Customer speaks. We label as "Speaker 2", run inference: Male, 46-60, 80% confidence.
Burst 3 (0:15 - 0:18): Customer speaks again. We run inference: Male, 46-60, 85% confidence.
Since this is a real-time WebSocket, when and how often do we send the prediction back to the client?

- Option A (Continuous Stream): After every single customer burst, we recalculate our "majority vote" or "running average" for the customer's attributes and push an update over the WebSocket. The client gets continuous real-time updates.

- Option B (Threshold-Triggered Lock): We accumulate predictions silently. As soon as we have enough bursts to hit a highly confident threshold (e.g., 3 consecutive matches), we emit a final "Customer Confirmed" event and stop running the heavy models to save CPU for the rest of the call.

- Option C (End of Call): We accumulate silently and only emit the final, most accurate prediction when the WebSocket connection is closed.

My Recommendation: I recommend a hybrid approach leaning towards Option A. Clients using WebSockets usually want to see a live dashboard updating in real-time. We can emit an event after every burst, but maybe include a status: "in_progress" | "locked" flag once our confidence is high enough that it won't change.


# Optimization: Agent Inference Skip (Real-time Streaming)

Since our Agent (the AI) is our own voice, predicting their age and gender is a waste of CPU/GPU resources and confuses the client’s real-time dashboard.

* **How it works**: Using the **Greeting Heuristic**, the first detected speaker is labeled as the "Agent". 
* **The Skip**: For every subsequent "Agent" speech burst, we bypass the heavy `Wav2Vec2` machine learning models.
* **The Result**: 
    * **Agent Speaking**: We send a minimal event: `{"role": "Agent", "status": "speaking"}`. This allows the client to know the agent is talking without wasting compute.
    * **Customer Speaking**: We run the full inference and return gender, age, and confidence scores.
* **Benefit**: This effectively cuts our compute costs by ~50% in a balanced conversation and ensures the "Inference Results" are 100% focused on the caller.



# Question 5: The Codec & Sample Rate Contract
When a client sends a .wav file, the first few bytes are a header that tells us exactly how to decode the audio (e.g., "This is 16kHz, 16-bit PCM").

When a client streams "raw bytes" over a WebSocket, there is often no header. Just a continuous flow of numbers. If we don't know the sample rate or codec, the audio will just sound like high-pitched static or slow-motion noise, and the model will fail.

How do we know how to decode the incoming byte stream?

Option A (The Strict Contract): We hardcode the server to only accept one specific format (e.g., 16kHz, 16-bit Mono PCM). If the client sends anything else, it breaks.
Option B (The Setup Message): We require the client to send a JSON configuration message as the very first message over the WebSocket (e.g., {"event": "start", "sample_rate": 8000, "encoding": "mulaw"}). We use this config to decode all subsequent raw byte messages.
Option C (Container Streaming): We assume the bytes are actually chunks of a standard container (like a streaming .webm file from a browser), and we pipe them through an ffmpeg subprocess that can auto-detect the format on the fly.
My Recommendation: I highly recommend Option B. This is exactly how enterprise telephony APIs (like Twilio Media Streams) work. The client sends a start event with metadata, and then streams the raw bytes. It makes our service highly flexible to different call center platforms (some use 8kHz, some use 16kHz) without needing ffmpeg to guess the format.

## Who is the client here?

The Client is the software or server that is connecting to our API over the WebSocket.

For example, the "client" might be:

- A telephony server (like Twilio or a PBX system) that is routing the call.
- The call center software running on the Agent's computer.

The Customer and Agent are just the humans whose voices happen to be inside the audio bytes that the "client" (the software) is sending to us.

So, when the call connects, the telephony software (the client) opens a WebSocket to our backend. Since that software is going to stream raw bytes of the conversation to us, we need to agree with that software on how those bytes are formatted.

# Alternative Base Models (Trade-offs & Constraints)
While `audeering/wav2vec2` is currently used as an off-the-shelf ONNX solution, other SOTA architectures were evaluated but passed on for the following reasons:
- **Whisper (OpenAI Encoder Embeddings):** While Whisper is exceptional at handling noisy, real-world data, it strictly violates our latency constraints (<500ms) for real-time WebSocket streaming due to its heavy computational cost.
- **HuBERT:** It provides better phonetic stability on stutters than Wav2Vec2, but requires significant engineering effort (custom fine-tuning and training a classification head) while yielding a lower ROI for noisy conversational audio.
- **ECAPA-TDNN / x-vectors (SpeechBrain):** Although extremely lightweight and incredibly fast (we already load it for diarization), adopting it for age/gender prediction requires us to explicitly label our own domain-specific datasets and train a custom classification head from scratch, as there is no ready-to-use version for this specific task.

### Why do we need to add age/gender labels manually for ECAPA-TDNN? Didn't we do the same for Wav2Vec2?
We didn't have to label data or train a custom classification head for Wav2Vec2 because the specific model we downloaded (`audeering/wav2vec2-large-robust-6-ft-age-gender`) was **already fine-tuned** for this exact task by Audeering. They did the heavy lifting of gathering demographic datasets and training the final prediction layer. 

ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) was trained by SpeechBrain for a completely different task: **Speaker Verification** (figuring out if Speaker A is the same person as Speaker B). While its acoustic embeddings contain the physical acoustic features needed to guess age and gender, nobody has published a pre-trained classification head for it on Hugging Face. If we switch to ECAPA-TDNN to predict age/gender, we would have to gather the datasets and train that head ourselves.


# Improvements to try:

1.
We noticed earlier that sample spontaneous-speech-en-18389.mp3 completely failed on gender (Predicted: Female, True: Male) and flagged as degraded. If you want to invoke the Systematic Debugging workflow, we can treat this failure as a bug. We would:

- Isolate that exact audio file.
- Instrument the VAD, SNR, and ONNX layers to dump exactly what data is flowing into the model.
- Figure out the root cause (e.g., Did VAD clip the start of a word? Is the background noise frequency tricking the model?)
- Implement a targeted fix (like tuning the VAD threshold or adjusting SNR bounds).


2. 
If you want to build something new, we can invoke the Brainstorming workflow to design and spec out our next major feature before we write the code. Some high-value ideas based on our QUICK-NOTES.md:

Agent Inference Skip: Brainstorming the architectural logic for how the WebSocket will automatically detect the Agent's voice (via the Greeting Heuristic) and bypass the heavy ML model to save 50% CPU.

Continuous Age Regression: Brainstorming how we could extract the embeddings to output a continuous age (e.g., 42.5 years) instead of rigid brackets to solve our exact-accuracy problem.

Speaker Diarization Module: Designing the flow for how we would integrate Pyannote to handle Mono-mixed 2-party calls natively.

# Question 6: Diarizing Back-and-Forth Conversation

**Question:** When uploading an entire audio file (like a real call), we are diarizing and taking only the audio of the customer. How are the subsequent voices guessed during back-and-forth talking?

**Answer:** 
The diarization module uses **Speaker Embeddings** (specifically `speechbrain/spkrec-ecapa-voxceleb`). Whenever the pipeline detects a new burst of speech, it extracts an "acoustic fingerprint" (a vector representing the physical characteristics of the speaker's vocal tract, completely independent of *what* they are saying).

For every new burst, it compares its fingerprint against the known speakers using cosine similarity:
- **Burst 1:** "Hello, thanks for calling..." → New fingerprint created! Saved as `speaker_0` (Agent).
- **Burst 2:** "Hi, my package is late..." → Compared to `speaker_0`. No match! Saved as `speaker_1` (Customer).
- **Burst 3:** "I can help with that." → Compared to `speaker_0` and `speaker_1`. High match with `speaker_0`! Assigned to Agent.

No matter how many times they go back and forth or interrupt each other, the system mathematically groups all of `speaker_0`'s bursts into one bucket, and all of `speaker_1`'s bursts into another. Finally, the API grabs the customer's bucket (using our Greeting Heuristic), caps it, and runs the inference on just their voice.