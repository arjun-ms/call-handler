import os
import csv
import json
import logging
from pathlib import Path
import pytest

from app.audio.preprocess import normalize_audio, load_audio
from app.inference.pipeline import predict

logger = logging.getLogger(__name__)

APPTEK_DIR = Path("tests/input-audio-files/apptek")
COMMONPHONE_DIR = Path("tests/input-audio-files/commonphone")
LOGS_DIR = Path("logs")

def load_metadata(tsv_path):
    metadata = {}
    if not tsv_path.exists():
        return metadata
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            metadata[row["filename"]] = row
    return metadata

@pytest.fixture(scope="session", autouse=True)
def setup_eval_logs():
    os.makedirs(LOGS_DIR, exist_ok=True)
    results_file = LOGS_DIR / "eval_datasets_results.json"
    
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {"apptek": [], "commonphone": []}
    else:
        all_results = {"apptek": [], "commonphone": []}
        
    yield all_results
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

def evaluate_dataset(dataset_dir, dataset_name, all_results):
    if not dataset_dir.exists():
        pytest.skip(f"Directory {dataset_dir} not found.")
        
    audio_files = list(dataset_dir.rglob("*.wav"))
    if not audio_files:
        pytest.skip(f"No audio files found in {dataset_name} directory.")
        
    metadata = load_metadata(dataset_dir / "metadata.tsv")
    
    for audio_file in audio_files:
        expected = metadata.get(audio_file.name, {})
        
        # Preprocess and load audio
        try:
            audio, sr = load_audio(str(audio_file))
            # Cap to 10 seconds to avoid memory OOM and long inference times
            max_eval_sec = 10.0
            max_samples = int(max_eval_sec * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            result = predict(audio, sr)
            status = "success"
        except Exception as e:
            result = {"error": str(e)}
            status = "error"
            
        test_record = {
            "filename": audio_file.name,
            "expected": expected,
            "actual": result,
            "status": status
        }
        
        if dataset_name not in all_results:
            all_results[dataset_name] = []
        all_results[dataset_name].append(test_record)
        
        if status == "success":
            assert "gender" in result
            assert "age_bracket" in result
            print(f"\nProcessed {audio_file.name}: got ({result.get('gender', {}).get('prediction')}, {result.get('age_bracket', {}).get('prediction')})")
        else:
            print(f"\nFailed {audio_file.name}: {result['error']}")

def test_apptek_evaluation(setup_eval_logs):
    evaluate_dataset(APPTEK_DIR, "apptek", setup_eval_logs)

def test_commonphone_evaluation(setup_eval_logs):
    evaluate_dataset(COMMONPHONE_DIR, "commonphone", setup_eval_logs)
