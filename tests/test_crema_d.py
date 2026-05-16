import os
import json
import logging
from pathlib import Path
import pytest

from app.audio.preprocess import normalize_audio, load_audio
from app.inference.pipeline import predict

logger = logging.getLogger(__name__)

CREMA_D_DIR = Path(r"E:\Work\call-handler-classification\test-audio-files\CREMA-D")
LOGS_DIR = Path(r"E:\Work\call-handler-classification\logs")

def get_expected_from_filename(filename: str):
    # e.g. "1001_male_51.wav" -> parts = ["1001", "male", "51"]
    # "1008_female_older.wav" -> parts = ["1008", "female", "older"]
    parts = filename.replace(".wav", "").split("_")
    if len(parts) >= 3:
        expected_gender = parts[1]
        age_str = parts[2]
        
        if age_str.isdigit():
            age = int(age_str)
            if age < 30:
                expected_age = "18-30"
            elif age < 45:
                expected_age = "31-45"
            elif age < 60:
                expected_age = "46-60"
            else:
                expected_age = "60+"
        elif age_str == "older":
            expected_age = "60+"
        else:
            expected_age = "unknown"
            
        return expected_gender, expected_age
    return "unknown", "unknown"

@pytest.fixture(scope="session", autouse=True)
def setup_logs():
    os.makedirs(LOGS_DIR, exist_ok=True)
    results_file = LOGS_DIR / "crema_d_results.json"
    
    # Initialize or clear results for this test run
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []
        
    yield all_results
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

def test_crema_d_audio_files(setup_logs):
    all_results = setup_logs
    if not CREMA_D_DIR.exists():
        pytest.skip(f"Directory {CREMA_D_DIR} not found.")
        
    audio_files = list(CREMA_D_DIR.glob("*.wav"))
    if not audio_files:
        pytest.skip("No audio files found in CREMA-D directory.")
        
    for audio_file in audio_files:
        expected_gender, expected_age = get_expected_from_filename(audio_file.name)
        
        # Preprocess and load audio
        audio, sr = load_audio(str(audio_file))
        
        # Run inference
        result = predict(audio, sr)
        
        actual_gender = result.get("gender", {}).get("prediction")
        actual_age = result.get("age_bracket", {}).get("prediction")
        
        # Log results
        test_record = {
            "filename": audio_file.name,
            "expected": {
                "gender": expected_gender,
                "age_bracket": expected_age
            },
            "actual": result
        }
        all_results.append(test_record)
        
        # Basic structural assertions
        assert "gender" in result
        assert "age_bracket" in result
        
        # We can also add soft assertions here or let the user review the JSON.
        # print for local debugging
        print(f"\nProcessed {audio_file.name}: expected ({expected_gender}, {expected_age}), got ({actual_gender}, {actual_age})")
