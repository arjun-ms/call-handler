"""Download real human speech samples into tests/test-audio-files/REAL_SAMPLES/.

Sources: PolyAI/minds14 (simulated e-banking call center conversations).
This dataset provides a much more realistic test for call-handler-classification
compared to read audiobooks, as it contains natural conversational speech over
a phone channel.
"""
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "test-audio-files", "REAL_SAMPLES")

def download_samples():
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Error: 'datasets' and 'soundfile' are required.")
        print("Install them with: pip install datasets soundfile")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading PolyAI/minds14 (en-US) conversational dataset...")
    
    try:
        # Use streaming=True so we don't have to download the entire dataset
        ds = load_dataset("PolyAI/minds14", name="en-US", split="train", streaming=True)
        it = iter(ds)
        
        # Pull 5 distinct conversational samples
        samples_to_get = 5
        print(f"Downloading {samples_to_get} call center samples...\n")
        
        for i in range(samples_to_get):
            sample = next(it)
            audio = sample["audio"]
            
            # Use intent class name if available to give it a descriptive name
            # intent_class is usually an integer in minds14, but we'll just use a generic name
            filename = f"minds14_call_sample_{i+1:02d}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            print(f"  Saving {filename}...")
            # sf.write will automatically convert it appropriately
            sf.write(filepath, audio["array"], audio["sampling_rate"], subtype="PCM_16")
            
            dur = len(audio["array"]) / audio["sampling_rate"]
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  [ok] {filename}: {dur:.1f}s, {size_kb:.0f}KB")
            
        print(f"\nDone! Samples are located in: {os.path.abspath(OUTPUT_DIR)}")
        print("You can now use these for testing the call classification pipeline.")
        
    except Exception as e:
        print(f"Failed to download samples: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_samples()
