import os
import csv
import requests
from huggingface_hub import list_repo_files, hf_hub_download

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_apptek(num_samples=20):
    print("Processing AppTek Call-Center Dialogues...")
    try:
        out_dir = "tests/test-audio-files/apptek"
        ensure_dir(out_dir)
        
        repo_id = "apptek-com/apptek_callcenter_dialogues"
        files = list_repo_files(repo_id, repo_type="dataset")
        wav_files = [f for f in files if f.endswith(".wav")]
        
        metadata = []
        for i, file_path in enumerate(wav_files[:num_samples]):
            local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=file_path, local_dir=out_dir)
            filename = os.path.basename(local_path)
            
            # Save metadata (mock since metadata is not explicitly provided)
            meta_row = {
                "filename": filename,
                "gender": "unknown",
                "age": "unknown",
                "accent": "unknown"
            }
            metadata.append(meta_row)
            
        with open(os.path.join(out_dir, "metadata.tsv"), "w", newline="", encoding="utf-8") as f:
            if metadata:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(metadata)
        print(f"Saved {len(metadata)} AppTek samples.")
    except Exception as e:
        print(f"Error loading AppTek: {e}")

def process_commonphone(num_samples=20):
    print("Processing CommonPhone Dataset...")
    try:
        out_dir = "tests/test-audio-files/commonphone"
        ensure_dir(out_dir)
        
        url = "https://datasets-server.huggingface.co/first-rows?dataset=pklumpp%2FCommonPhoneDataset&config=default&split=train"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        metadata = []
        for i, row_data in enumerate(data.get("rows", [])):
            if i >= num_samples:
                break
                
            row = row_data["row"]
            audio_info = row.get("audio", [])
            if not audio_info:
                continue
                
            # Usually a list with a dict inside
            audio_src = audio_info[0]["src"] if isinstance(audio_info, list) else audio_info.get("src")
            if not audio_src:
                continue
                
            # Download audio
            audio_resp = requests.get(audio_src)
            audio_resp.raise_for_status()
            
            filename = f"commonphone_{i}.wav"
            file_path = os.path.join(out_dir, filename)
            with open(file_path, "wb") as f:
                f.write(audio_resp.content)
            
            meta_row = {
                "filename": filename,
                "gender": row.get("gender", "unknown"),
                "age": row.get("age", "unknown")
            }
            metadata.append(meta_row)
            
        with open(os.path.join(out_dir, "metadata.tsv"), "w", newline="", encoding="utf-8") as f:
            if metadata:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(metadata)
        print(f"Saved {len(metadata)} CommonPhone samples.")
    except Exception as e:
        print(f"Error loading CommonPhone: {e}")

if __name__ == "__main__":
    process_apptek(20)
    process_commonphone(20)
