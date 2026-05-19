# Speaker Identification & Separation

In voice call environments, separating the voices of the **Agent** (outbound representative or assistant) and the **Customer** (caller) is essential. Performing attribute inference on the agent's voice would waste system resources and skew predictions.

The Voice Attribute Inference Service implements **Stateful Speaker Diarization** and **Role Labeling** as core architectural features.

---

## 🛠️ The Architecture: Diarization & Role Labeling

Rather than relying on external platform-side channel separation (which is often unavailable on mono-channel telephony recordings), the service processes mixed audio through a sequential three-step pipeline:

```
  [Mixed Audio Chunks]
           │
           ▼
     [VadTrigger]  ──(Speech Bursts)──►  [SpeechBrain Diarizer]
                                                   │
                                                   ▼
                                         (Speaker ID Assignment)
                                                   │
                                                   ▼
                                          [Greeting Heuristic]
                                            /             \
                                  (Agent)  /               \  (Customer)
                                          ▼                 ▼
                                    [Skip Model]       [Run Inference]
                                    (Fast-track)       (Wav2Vec2 ONNX)
```

### 1. Stateful VAD Chunks (`app/audio/vad.py`)
Continuous streaming audio is ingested by `VadTrigger`. The module acts as a gatekeeper:
*   It filters out background noise and silences.
*   It groups contiguous speech segments into distinct **speech bursts** (completed utterances).
*   It triggers downstream diarization only when a speaker pauses (detected via a configurable silence window, e.g. `500ms`).

### 2. Embedded Diarization (`app/inference/diarizer.py`)
Each completed speech burst is analyzed by the `Diarizer` class:
*   **Vector Embeddings**: It extracts high-fidelity voice embeddings using a pre-trained **SpeechBrain ECAPA-TDNN** encoder (`spkrec-ecapa-voxceleb`).
*   **Cosine Similarity Clustering**: The system stores running averages (centroids) of each unique voice detected in the session.
*   **Speaker Assignment**: Incoming bursts are compared against existing centroids. If similarity matches above a threshold (default `0.5`), the burst is clustered under that speaker. If not, a new speaker ID (e.g. `speaker_1`, `speaker_2`) is initialized.

### 3. Role Labeling: The Greeting Heuristic (`app/audio/stream_session.py`)
Once speaker IDs are assigned, they are mapped to roles using the **Agent-First Greeting Heuristic**:
*   **The Heuristic**: Outbound logistics calls almost always begin with the agent saying a greeting (e.g., *"Hello, thank you for calling..."*). Therefore, the **first speaker to talk is designated as the Agent**.
*   **Customer Designation**: Any speaker who speaks *after* the agent is labeled as the **Customer**.
*   **Performance Optimization**: 
    *   **Agent bursts** skip the heavy Wav2Vec2 inference model entirely. They instantly return an event of `{ "role": "Agent", "status": "speaking" }` back to the client.
    *   **Customer bursts** are fed directly into the model to predict age and gender, which are then smoothed using a rolling consensus window.

---

## 🔁 Boundary Conditions & Fallbacks

To ensure robustness in real-world telephony systems, the diarization system implements three fail-safes:

### A. Single-Speaker Correction
If only one speaker is detected in the entire call (e.g., during testing with a clean single-speaker benchmark file), the greeting heuristic would normally label them as the "Agent" and skip all inference.
*   **The Fix**: Upon session finalization (`session.finalize()`), if the session contains exactly one speaker role mapping, the system automatically **reclassifies that speaker as the Customer** and retroactively triggers inference on their final burst.

### B. Embedding Failover (Mock Clustering)
In resource-constrained environments (or if local compilers fail to install the heavy C-bindings for PyTorch/SpeechBrain):
*   **The Fix**: The `Diarizer` catches initial import errors and falls back to a deterministic, high-performance mock clustering strategy (cycling speakers by burst sequence) to prevent app crashes while maintaining API compatibility.

### C. Large Burst Safety
Long speech segments can cause Out-Of-Memory (OOM) failures in deep learning inference engines.
*   **The Fix**: `StreamSession` automatically checks burst sample size and safely truncates incoming customer speech segments to a maximum of **7.0 seconds** before feeding them to Wav2Vec2.