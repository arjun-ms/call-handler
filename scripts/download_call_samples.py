from datasets import load_dataset
import soundfile as sf
import os
import sys

def download_call_samples():
    """
    Downloads samples from PolyAI/minds14 which contains recordings of 
    people asking banking questions (mimicking call center interactions).
    """
    try:
        print("Loading PolyAI/minds14 dataset (en-US)...")
        # Use streaming=True to avoid downloading the whole dataset
        ds = load_dataset("PolyAI/minds14", "en-US", split="train", streaming=True)
        it = iter(ds)
        
        output_dir = "tests/test-audio-files/REAL_SAMPLES/minds14_calls"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving samples to {output_dir}...")
        for i in range(5):
            sample = next(it)
            audio = sample["audio"]
            filename = f"caller_en_US_{i+1}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Save using soundfile
            sf.write(filepath, audio["array"], audio["sampling_rate"])
            
            duration = len(audio["array"]) / audio["sampling_rate"]
            size_kb = os.path.getsize(filepath) / 1024
            print(f" - Saved {filename}: {duration:.2f}s, {size_kb:.1f} KB")
            
        print("\nSuccess! Added e-banking call samples.")
        
    except Exception as e:
        print(f"Error downloading minds14 samples: {e}")

if __name__ == "__main__":
    download_call_samples()
