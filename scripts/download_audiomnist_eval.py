import os
import csv
import soundfile as sf
from datasets import load_dataset

def main():
    print("Downloading gilkeyio/AudioMNIST from HuggingFace...")
    # Load dataset. It is small so we can load the train split.
    ds = load_dataset('gilkeyio/AudioMNIST', split='train')
    
    # We will only extract 100 samples for evaluation to save time.
    eval_dir = os.path.join("tests", "input-audio-files", "AudioMNIST")
    os.makedirs(eval_dir, exist_ok=True)
    
    tsv_path = os.path.join(eval_dir, "validated.tsv")
    
    # AudioMNIST has 'gender' (0 for male, 1 for female), 'age', 'audio'
    # Wait, the feature list showed gender: ClassLabel(names=['male', 'female']).
    gender_map = {0: "male", 1: "female"}
    
    print(f"Saving 100 samples to {eval_dir}...")
    
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accents'])
        
        count = 0
        for i, sample in enumerate(ds):
            if count >= 100:
                break
            
            # audio is a dict with 'array' and 'sampling_rate'
            audio_array = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            
            gender_id = sample['gender']
            gender_str = gender_map.get(gender_id, "unknown")
            age = sample['age']
            
            filename = f"sample_{i}.wav"
            filepath = os.path.join(eval_dir, filename)
            
            # Save as WAV
            sf.write(filepath, audio_array, sr)
            
            # Write to TSV (mocking CommonVoice structure)
            writer.writerow([f"client_{sample['speaker_id']}", filename, sample['digit'], 0, 0, age, gender_str, sample['accent']])
            count += 1

    print(f"Downloaded and saved {count} samples successfully.")
    print(f"Run the eval harness using: python scripts/eval_commonvoice.py --dir {eval_dir} --tsv {tsv_path}")

if __name__ == "__main__":
    main()
