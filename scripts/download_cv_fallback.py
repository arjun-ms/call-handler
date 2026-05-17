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

    print("Attempting to stream common_voice (old version)...")
    try:
        ds = load_dataset(
            "common_voice",
            "en",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Failed to load old common_voice dataset: {e}")
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

            # In older common_voice, gender might not be string mapped, but let's check
            if gender == "male":
                pass

            audio_dict = sample["audio"]
            audio_array = audio_dict["array"]
            sr = audio_dict["sampling_rate"]
            path = sample["path"]
            
            fname = os.path.basename(path)
            if not fname.endswith(".mp3"):
                fname += ".mp3"
                
            out_file = out_dir / fname
            
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
