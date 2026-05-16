import argparse
import asyncio
import os
import json
import httpx
from typing import List, Dict
from pathlib import Path

# CommonVoice format usually has a TSV with client_id, path, sentence, up_votes, down_votes, age, gender, accents

async def evaluate_dataset(
    dataset_dir: str,
    tsv_path: str,
    api_url: str = "http://localhost:8000/v1/infer",
    max_samples: int = 100
):
    """
    Evaluates the model against a subset of Mozilla CommonVoice dataset.
    Requires a TSV file with 'path', 'gender', and 'age' columns.
    """
    # Read TSV
    import csv
    
    samples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('gender') and row.get('age'):
                samples.append({
                    "path": os.path.join(dataset_dir, row['path']),
                    "true_gender": row['gender'].lower(),
                    "true_age": row['age'] # Needs mapping to brackets
                })
                if len(samples) >= max_samples:
                    break
                    
    print(f"Loaded {len(samples)} valid samples.")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(*[
            _infer_sample(client, api_url, s) for s in samples
        ])
        
    _compute_metrics(samples, results)

async def _infer_sample(client: httpx.AsyncClient, api_url: str, sample: Dict) -> Dict:
    path = sample["path"]
    if not os.path.exists(path):
        # Try with .mp3 extension if missing
        path += ".mp3"
        if not os.path.exists(path):
            return {"error": "file_not_found"}
            
    try:
        with open(path, "rb") as f:
            resp = await client.post(
                api_url,
                files={"file": (os.path.basename(path), f, "audio/mpeg")}
            )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}

def _compute_metrics(samples: List[Dict], results: List[Dict]):
    correct_gender = 0
    total_valid = 0
    
    for sample, res in zip(samples, results):
        if "error" in res:
            continue
            
        total_valid += 1
        true_g = sample["true_gender"]
        pred_g = res["gender"]["prediction"]
        
        # Simple gender mapping for CommonVoice
        if true_g in ["male", "female"] and pred_g == true_g:
            correct_gender += 1
            
    if total_valid > 0:
        acc = correct_gender / total_valid
        print(f"Evaluated {total_valid} samples successfully.")
        print(f"Gender Accuracy: {acc:.1%} ({correct_gender}/{total_valid})")
    else:
        print("No valid results to evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on CommonVoice")
    parser.add_argument("--dir", type=str, required=True, help="Directory with audio clips")
    parser.add_argument("--tsv", type=str, required=True, help="Path to validated.tsv")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/infer")
    parser.add_argument("--n", type=int, default=100)
    
    args = parser.parse_args()
    asyncio.run(evaluate_dataset(args.dir, args.tsv, args.url, args.n))
