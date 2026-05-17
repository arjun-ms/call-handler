import asyncio
import json
import wave
import websockets

async def test_stream():
    uri = "ws://localhost:8000/ws/analyze"
    
    # We will use one of the test wav files
    wav_file = "E:/Work/call-handler-classification/tests/input-audio-files/real-call/female_17.wav"
    
    # Read the wav file
    try:
        wf = wave.open(wav_file, "rb")
        sample_rate = wf.getframerate()
        print(f"Opened {wav_file}, Sample Rate: {sample_rate}Hz")
    except Exception as e:
        print(f"Could not open {wav_file}: {e}")
        return

    async with websockets.connect(uri, ping_timeout=60) as websocket:
        # 1. Send start event
        start_event = {
            "type": "start",
            "call_id": "test_manual_1",
            "sample_rate": sample_rate,
            "encoding": "pcm_s16le"
        }
        await websocket.send(json.dumps(start_event))
        print("Sent start event.")
        
        # Start a background task to receive messages
        async def receive_events():
            try:
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print(f"\n[SERVER EVENT] {data}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")

        recv_task = asyncio.create_task(receive_events())
        
        # 2. Stream audio in chunks
        chunk_size = 4096  # bytes
        print("Streaming audio...")
        while True:
            data = wf.readframes(chunk_size // 2) # 2 bytes per sample
            if not data:
                break
            await websocket.send(data)
            # Sleep a tiny bit to simulate real-time streaming
            await asyncio.sleep(len(data) / (sample_rate * 2))
            
        print("Finished streaming audio.")
        
        # Give it a moment to process the last bursts
        await asyncio.sleep(2)
        
        # 3. Stop
        await websocket.send(json.dumps({"type": "stop"}))
        await websocket.close()
        recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_stream())
