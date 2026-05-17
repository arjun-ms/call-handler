import os
import csv
import shutil
from pathlib import Path

def main():
    out_dir = Path("tests/input-audio-files/CommonVoice")
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "validated.tsv"

    src_audio = Path("sample_audio/test.wav")
    if not src_audio.exists():
        print("Source audio not found.")
        return

    print("Generating mock CommonVoice dataset since HuggingFace requires authentication...")

    # Ensure we have a TSV writer ready
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents"])

        for i in range(1, 6):
            fname = f"sample_{i}.wav"
            out_file = out_dir / fname
            
            # Copy the file
            shutil.copy(src_audio, out_file)

            # Alternate genders for the mock
            gender = "female" if i % 2 == 0 else "male"

            writer.writerow([
                f"client_{i}",
                fname,
                "This is a mock sentence.",
                2,
                0,
                "thirties",
                gender,
                ""
            ])
            
            print(f"Saved: {fname} with gender={gender}")
                
    print(f"Done. Mock dataset saved to {out_dir}")

if __name__ == "__main__":
    main()
