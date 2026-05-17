import os
import csv
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets required.")
    sys.exit(1)

def main():
    out_dir = Path("tests/input-audio-files/CommonVoice")
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "validated.tsv"

    print("Attempting to stream mozilla-foundation/common_voice_11_0...")
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Note: Mozilla Common Voice requires an HF_TOKEN and accepting terms on HuggingFace.")
        sys.exit(1)

    samples = []
    count = 0
    max_samples = 10

    # Ensure we have a TSV writer ready
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents"])

        for sample in ds:
            # We only want samples that have both gender and age
            gender = sample.get("gender", "")
            age = sample.get("age", "")
            
            if not gender or not age:
                continue

            # Save the audio bytes
            audio_dict = sample["audio"]
            audio_array = audio_dict["array"]
            sr = audio_dict["sampling_rate"]
            path = sample["path"]
            
            if not path.endswith(".mp3"):
                path += ".mp3"
                
            fname = os.path.basename(path)
            out_file = out_dir / fname
            
            # Since we streamed, we might just have the raw array. We need to save it.
            # Using soundfile
            import soundfile as sf
            sf.write(str(out_file), audio_array, sr)

            writer.writerow([
                sample.get("client_id", ""),
                fname,
                sample.get("sentence", ""),
                sample.get("up_votes", 0),
                sample.get("down_votes", 0),
                age,
                gender,
                sample.get("accents", "")
            ])
            
            count += 1
            print(f"Saved {count}/{max_samples}: {fname}")
            if count >= max_samples:
                break
                
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
