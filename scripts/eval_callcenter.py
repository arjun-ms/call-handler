import os
import glob
import time
import json
import asyncio
from pathlib import Path

from app.audio.preprocess import load_audio
from app.inference.pipeline import predict

async def eval_dataset(dataset_name, dir_pattern):
    print(f"\n--- Evaluating {dataset_name} ---")
    files = glob.glob(dir_pattern, recursive=True)
    if not files:
        print(f"No files found for {dataset_name} at {dir_pattern}")
        return
        
    print(f"Found {len(files)} files.")
    
    results = []
    start_time = time.time()
    
    for f_path in files:
        try:
            audio, sr = load_audio(f_path)
            res = predict(audio, sr)
            results.append({
                "file": os.path.basename(f_path),
                "gender": res.get("gender", {}).get("prediction"),
                "age": res.get("age_bracket", {}).get("prediction"),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "file": os.path.basename(f_path),
                "status": "error",
                "error": str(e)
            })
            
    end_time = time.time()
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    genders = {"male": 0, "female": 0, "unknown": 0}
    for r in successful:
        g = r.get("gender")
        if g in genders:
            genders[g] += 1
        else:
            genders["unknown"] += 1
            
    print(f"Evaluated {len(files)} files in {end_time - start_time:.2f}s")
    print(f"Success: {len(successful)}, Failed: {len(failed)}")
    print(f"Gender Distribution: {genders}")
    
    # Save detailed report
    report = {
        "dataset": dataset_name,
        "total_files": len(files),
        "success_count": len(successful),
        "failed_count": len(failed),
        "time_taken_seconds": round(end_time - start_time, 2),
        "average_time_per_file": round((end_time - start_time) / max(1, len(files)), 3),
        "gender_distribution": genders,
        "results": results
    }
    
    report_file = f"eval_report_{dataset_name.lower()}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to {report_file}")

async def main():
    await eval_dataset("AppTek", "tests/input-audio-files/apptek/**/*.wav")
    await eval_dataset("CommonPhone", "tests/input-audio-files/commonphone/**/*.wav")

if __name__ == "__main__":
    asyncio.run(main())
