import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

# Add the project root to sys.path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.audio.preprocess import normalize_audio, load_audio, cleanup_temp
from app.inference.pipeline import predict

def parse_labels_from_filename(filename: str) -> tuple[str, str]:
    """
    Tries to guess gender and age from filename.
    Supports formats:
    - ID_gender_age.wav (1001_male_25.wav)
    - gender_age_ID.wav (female_45_01.wav)
    - ID_gender_label.wav (1008_female_older.wav)
    """
    name = Path(filename).stem.lower()
    parts = name.split("_")
    
    gender = "unknown"
    age = "unknown"
    
    # Try to find gender
    for g in ["male", "female", "child"]:
        if g in parts:
            gender = g
            break
            
    # Try to find age or bracket
    for p in parts:
        if p.isdigit():
            val = int(p)
            if 10 < val < 100: # Likely an age
                if val < 30: age = "18-30"
                elif val < 45: age = "31-45"
                elif val < 60: age = "46-60"
                else: age = "60+"
                break
        if p in ["older", "senior"]:
            age = "60+"
            break
        if p in ["young", "teen"]:
            age = "18-30"
            break
            
    return gender, age

def run_eval(directory: str, output_name: str = None):
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {directory} not found.")
        return

    supported_exts = ["*.wav", "*.mp3", "*.aac", "*.ogg", "*.webm", "*.opus", "*.m4a"]
    audio_files = []
    for ext in supported_exts:
        audio_files.extend(dir_path.glob(ext))
    if not audio_files:
        print(f"No audio files found in {directory}")
        return

    print(f"\n[STARTING] Evaluation on {len(audio_files)} files...")
    print(f"{'Filename':<30} | {'Expected':<20} | {'Predicted':<20} | {'Status'}")
    print("-" * 85)

    results = []
    correct_gender = 0
    correct_age = 0
    start_time = time.time()

    for audio_file in audio_files:
        expected_g, expected_a = parse_labels_from_filename(audio_file.name)
        
        normalized_path = None
        try:
            # Normalize through ffmpeg so any format (.aac, .ogg, etc.) works
            normalized_path = normalize_audio(str(audio_file))
            audio, sr = load_audio(normalized_path)
            pred = predict(audio, sr)
            
            actual_g = pred["gender"]["prediction"]
            actual_a = pred["age_bracket"]["prediction"]
            
            g_match = expected_g == actual_g if expected_g != "unknown" else True
            a_match = expected_a == actual_a if expected_a != "unknown" else True
            
            if g_match: correct_gender += 1
            if a_match: correct_age += 1
            
            status = "PASS" if g_match and a_match else "FAIL"
            if expected_g == "unknown" and expected_a == "unknown":
                status = "INFO" # Informational
            
            print(f"{audio_file.name:<30} | {f'{expected_g}, {expected_a}':<20} | {f'{actual_g}, {actual_a}':<20} | {status}")
            
            results.append({
                "filename": audio_file.name,
                "expected": {"gender": expected_g, "age": expected_a},
                "actual": pred,
                "match": {"gender": g_match, "age": a_match}
            })
            
        except Exception as e:
            print(f"{audio_file.name:<30} | ERROR: {str(e)}")
        finally:
            cleanup_temp(normalized_path)

    total_time = time.time() - start_time
    avg_time = (total_time / len(audio_files)) * 1000 if audio_files else 0

    print("-" * 85)
    print(f"[COMPLETE] Evaluation Complete in {total_time:.2f}s (Avg: {avg_time:.0f}ms/file)")
    
    # Only show accuracy if we had labels
    labeled_count = len([r for r in results if r["expected"]["gender"] != "unknown"])
    if labeled_count > 0:
        print(f"Gender Accuracy: {correct_gender/labeled_count:.1%}")
        print(f"Age Accuracy:    {correct_age/labeled_count:.1%}")

    # Save report
    os.makedirs("logs", exist_ok=True)
    report_name = output_name or f"eval_report_{int(time.time())}.json"
    report_path = Path("logs") / report_name
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.ctime(),
            "directory": str(dir_path),
            "summary": {
                "total_files": len(audio_files),
                "labeled_files": labeled_count,
                "gender_accuracy": correct_gender/labeled_count if labeled_count > 0 else 0,
                "age_accuracy": correct_age/labeled_count if labeled_count > 0 else 0,
                "avg_latency_ms": avg_time
            },
            "details": results
        }, f, indent=4)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch evaluation on a directory of audio files.")
    parser.add_argument("dir", type=str, help="Directory containing audio files")
    parser.add_argument("--out", type=str, help="Custom name for the output JSON report")
    
    args = parser.parse_args()
    run_eval(args.dir, args.out)
