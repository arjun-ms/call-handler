# WebSocket Stream API Flow (`/ws/stream`)

Here is the step-by-step flow of the WebSocket endpoint, incorporating performance optimizations for real-time processing:

### 1. Connection & Handshake
*   **Establishment**: The client connects to `ws://.../ws/stream`.
*   **Start Signal**: The client must immediately send a text message with a JSON payload (e.g., `{"type": "start", "call_id": "123", "sample_rate": 16000, "encoding": "linear16"}`).
*   **Initialization**: The server validates this config and creates a dedicated `StreamSession` for this specific call, setting up the VAD (Voice Activity Detection), Diarizer, and history trackers.

### 2. Streaming Audio (The VAD Gatekeeper)
*   **Ingestion**: The client continuously streams raw binary audio data over the WebSocket.
*   **Decoding**: The server receives these bytes, decodes them according to the negotiated encoding (e.g., converting µ-law to standard float32), and feeds them into the `VadTrigger`.
*   **Buffering**: The VAD listens silently. It buffers the audio *only* when someone is actively speaking. 
*   **Burst Emission**: Once the speaker pauses (e.g., 500ms of silence), the VAD releases the buffered audio as a single **"speech burst"** (a complete utterance/sentence) and passes it to the next stage.

### 3. Diarization (Who is speaking?)
*   The server takes the completed speech burst and feeds it to the `Diarizer`.
*   The Diarizer analyzes the voice embedding and returns a unique identifier (e.g., `speaker_0` or `speaker_1`).

### 4. Role Assignment (The Greeting Heuristic)
*   The server checks its memory for this `speaker_id`.
*   If it is the **very first time** anyone has spoken on this call, it permanently tags them as the **"Agent"** (assuming the agent says "Hello" first).
*   Any new speaker detected after the Agent is permanently tagged as the **"Customer"**.

### 5. The Split (The Optimization)
This is where the pipeline diverges based on who is speaking:

**Path A: The Agent is Speaking**
*   **Skip Inference**: The server immediately bypasses the heavy Wav2Vec2 machine learning models.
*   **Fast Response**: It instantly sends a lightweight JSON response back to the client over the WebSocket:
    ```json
    {
        "type": "inference_result",
        "call_id": "123",
        "speaker_id": "speaker_0",
        "role": "Agent",
        "status": "speaking"
    }
    ```

**Path B: The Customer is Speaking**
*   **Heavy Inference**: The burst is passed to the Wav2Vec2 model (`predict()`) to estimate gender and age bracket.
*   **Consensus Smoothing**: The server adds these raw predictions to the Customer's specific history tracker. It looks at the **last 5 bursts** from the Customer and takes a majority vote to filter out anomalies.
*   **Full Response**: It sends the rich analytical payload back to the client:
    ```json
    {
        "type": "inference_result",
        "call_id": "123",
        "speaker_id": "speaker_1",
        "role": "Customer",
        "gender": "Female",
        "age_bracket": "31-45",
        "confidence_gender": 0.88,
        "confidence_age": 0.72
    }
    ```

### 6. Cleanup & Disconnect
*   The loop continues processing chunks and emitting results asynchronously until the client sends a `{"type": "stop"}` JSON text message or terminates the connection.
*   The server catches the disconnect, deletes the `StreamSession` object from memory (freeing up the audio buffers and VAD state), and closes the thread.
