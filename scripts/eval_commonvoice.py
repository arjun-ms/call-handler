import argparse
import asyncio
import os
import json
import httpx
from typing import List, Dict
from pathlib import Path
import csv
import datetime

# Brackets mapping based on CREMA-D evaluation
# Our ordered brackets: 18-30, 31-45, 46-60, 60+
BRACKETS = ["18-30", "31-45", "46-60", "60+"]

def map_age_to_bracket(age_str: str) -> str:
    age_str = str(age_str).strip().lower()
    if age_str in ['teens', 'twenties']:
        return '18-30'
    elif age_str in ['thirties', 'fourties', 'forties']:
        return '31-45'
    elif age_str in ['fifties']:
        return '46-60'
    elif age_str in ['sixties', 'seventies', 'eighties', 'nineties', 'older']:
        return '60+'
    return None

def is_within_one_bracket(true_b: str, pred_b: str) -> bool:
    if true_b not in BRACKETS or pred_b not in BRACKETS:
        return False
    idx_t = BRACKETS.index(true_b)
    idx_p = BRACKETS.index(pred_b)
    return abs(idx_t - idx_p) <= 1

async def evaluate_dataset(
    dataset_dir: str,
    tsv_path: str,
    api_url: str = "http://localhost:8000/v1/infer",
    max_samples: int = 100
):
    """
    Evaluates the model against a subset of Mozilla CommonVoice dataset.
    Requires a TSV file with 'audio_file', 'gender', and 'age' columns.
    """
    samples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('split') != 'test' and row.get('split'):
                continue
            
            gender = row.get('gender')
            age = row.get('age')
            if gender and age:
                mapped_age = map_age_to_bracket(age)
                if not mapped_age:
                    continue
                    
                audio_path = row.get('audio_file') or row.get('path') or row.get('filename')
                if not audio_path:
                    continue
                    
                # The MDC Spontaneous Speech dataset stores audio in `audios/` dir
                # Make sure the path matches reality
                full_path = os.path.join(dataset_dir, audio_path)
                if not os.path.exists(full_path):
                    # Try under 'audios'
                    full_path = os.path.join(dataset_dir, "audios", audio_path)
                    
                samples.append({
                    "path": full_path,
                    "true_gender": gender.lower(),
                    "true_age": mapped_age
                })
                if len(samples) >= max_samples:
                    break
                    
    print(f"Loaded {len(samples)} valid samples.")
    
    # Run evaluation with a concurrency limit
    semaphore = asyncio.Semaphore(2)
    
    async def bounded_infer(s):
        async with semaphore:
            return await _infer_sample(client, api_url, s)
            
    async with httpx.AsyncClient(timeout=300.0) as client:
        results = await asyncio.gather(*[
            bounded_infer(s) for s in samples
        ])
        
    metrics_str = _compute_metrics(samples, results)

    # Save detailed results and metrics
    dataset_name = os.path.basename(os.path.normpath(dataset_dir)).lower()
    output_dir = os.path.join("tests", "test-results", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON results
    filename_json = f"eval_results_{timestamp}.json"
    results_path = os.path.join(output_dir, filename_json)
    detailed = []
    for s, r in zip(samples, results):
        detailed.append({"ground_truth": s, "prediction": r})
    with open(results_path, "w") as f:
        json.dump(detailed, f, indent=2)
        
    # Text metrics
    filename_metrics = f"eval_metrics_{timestamp}.txt"
    metrics_path = os.path.join(output_dir, filename_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_str)
        
    print(f"\nDetailed results and metrics saved to: {output_dir}")

async def _infer_sample(client: httpx.AsyncClient, api_url: str, sample: Dict) -> Dict:
    path = sample["path"]
    if not os.path.exists(path):
        # Try with .mp3 extension if missing
        path += ".mp3"
        if not os.path.exists(path):
            return {"error": f"file_not_found: {path}"}
            
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

def _compute_metrics(samples: List[Dict], results: List[Dict]) -> str:
    correct_gender = 0
    correct_age_exact = 0
    correct_age_tolerant = 0
    total_valid = 0
    
    age_bracket_stats = {b: {"correct_exact": 0, "correct_tolerant": 0, "total": 0} for b in BRACKETS}
    
    output = []
    
    for sample, res in zip(samples, results):
        if "error" in res:
            output.append(f"Error on {sample['path']}: {res['error']}")
            continue
            
        total_valid += 1
        true_g = sample["true_gender"].lower()
        if "female" in true_g:
            true_g = "female"
        elif "male" in true_g:
            true_g = "male"
            
        true_a = sample["true_age"]
        
        pred_g = res.get("gender", {}).get("prediction")
        pred_a = res.get("age_bracket", {}).get("prediction")
        
        # Gender
        if true_g in ["male", "female"] and pred_g == true_g:
            correct_gender += 1
            
        # Age
        if true_a in BRACKETS:
            age_bracket_stats[true_a]["total"] += 1
            if pred_a == true_a:
                correct_age_exact += 1
                age_bracket_stats[true_a]["correct_exact"] += 1
            if is_within_one_bracket(true_a, pred_a):
                correct_age_tolerant += 1
                age_bracket_stats[true_a]["correct_tolerant"] += 1
            
    if total_valid > 0:
        output.append(f"\nEvaluated {total_valid} samples successfully.")
        output.append(f"Gender Accuracy: {correct_gender / total_valid:.1%} ({correct_gender}/{total_valid})")
        output.append(f"Age Bracket Accuracy (Exact): {correct_age_exact / total_valid:.1%} ({correct_age_exact}/{total_valid})")
        output.append(f"Age Bracket Accuracy (±1 Bracket): {correct_age_tolerant / total_valid:.1%} ({correct_age_tolerant}/{total_valid})")
        
        output.append("\nAge Bracket Breakdown:")
        for b in BRACKETS:
            st = age_bracket_stats[b]
            if st["total"] > 0:
                acc_exact = st["correct_exact"] / st["total"]
                acc_tol = st["correct_tolerant"] / st["total"]
                output.append(f"  {b}: Exact {acc_exact:.1%} | ±1 Bracket {acc_tol:.1%} (Total: {st['total']})")
    else:
        output.append("No valid results to evaluate.")
        
    final_output = "\n".join(output)
    print(final_output)
    return final_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on CommonVoice")
    parser.add_argument("--dir", type=str, required=True, help="Directory with audio clips")
    parser.add_argument("--tsv", type=str, required=True, help="Path to TSV file")
    # Actually the README says /analyze, not /v1/infer, based on CREMA-D Report
    parser.add_argument("--url", type=str, default="http://localhost:8000/analyze")
    parser.add_argument("--n", type=int, default=100)
    
    args = parser.parse_args()
    asyncio.run(evaluate_dataset(args.dir, args.tsv, args.url, args.n))
