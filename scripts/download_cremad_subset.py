"""
Download a stratified subset of CREMA-D audio files from GitHub and
generate a ground-truth metadata.tsv for evaluation.

CREMA-D filenames follow: {ActorID}_{Sentence}_{Emotion}_{Intensity}.wav
Demographics are from VideoDemographics.csv in the repo.

Strategy: pick 1 neutral clip per actor for a diverse set of actors,
covering a spread of ages and both genders.
"""
import argparse
import csv
import io
import os
import random
import time
from pathlib import Path
from urllib.request import urlretrieve

# ── CREMA-D demographics (from VideoDemographics.csv) ──────────────────
DEMOGRAPHICS = {
    1001: {"age": 51, "sex": "Male"},
    1002: {"age": 21, "sex": "Female"},
    1003: {"age": 21, "sex": "Female"},
    1004: {"age": 42, "sex": "Female"},
    1005: {"age": 29, "sex": "Male"},
    1006: {"age": 58, "sex": "Female"},
    1007: {"age": 38, "sex": "Female"},
    1008: {"age": 46, "sex": "Female"},
    1009: {"age": 24, "sex": "Female"},
    1010: {"age": 27, "sex": "Female"},
    1011: {"age": 32, "sex": "Male"},
    1012: {"age": 23, "sex": "Female"},
    1013: {"age": 22, "sex": "Female"},
    1014: {"age": 24, "sex": "Male"},
    1015: {"age": 32, "sex": "Male"},
    1016: {"age": 61, "sex": "Male"},
    1017: {"age": 42, "sex": "Male"},
    1018: {"age": 25, "sex": "Female"},
    1019: {"age": 29, "sex": "Male"},
    1020: {"age": 61, "sex": "Female"},
    1021: {"age": 30, "sex": "Female"},
    1022: {"age": 22, "sex": "Male"},
    1023: {"age": 22, "sex": "Male"},
    1024: {"age": 59, "sex": "Female"},
    1025: {"age": 48, "sex": "Female"},
    1026: {"age": 33, "sex": "Male"},
    1027: {"age": 44, "sex": "Male"},
    1028: {"age": 57, "sex": "Female"},
    1029: {"age": 33, "sex": "Female"},
    1030: {"age": 42, "sex": "Female"},
    1031: {"age": 31, "sex": "Male"},
    1032: {"age": 30, "sex": "Male"},
    1033: {"age": 31, "sex": "Male"},
    1034: {"age": 74, "sex": "Male"},
    1035: {"age": 48, "sex": "Male"},
    1036: {"age": 49, "sex": "Male"},
    1037: {"age": 45, "sex": "Female"},
    1038: {"age": 21, "sex": "Male"},
    1039: {"age": 51, "sex": "Male"},
    1040: {"age": 42, "sex": "Male"},
    1041: {"age": 42, "sex": "Male"},
    1042: {"age": 37, "sex": "Male"},
    1043: {"age": 25, "sex": "Female"},
    1044: {"age": 40, "sex": "Male"},
    1045: {"age": 22, "sex": "Male"},
    1046: {"age": 22, "sex": "Female"},
    1047: {"age": 22, "sex": "Female"},
    1048: {"age": 38, "sex": "Male"},
    1049: {"age": 25, "sex": "Female"},
    1050: {"age": 62, "sex": "Male"},
    1051: {"age": 56, "sex": "Male"},
    1052: {"age": 33, "sex": "Female"},
    1053: {"age": 35, "sex": "Female"},
    1054: {"age": 36, "sex": "Female"},
    1055: {"age": 57, "sex": "Female"},
    1056: {"age": 52, "sex": "Female"},
    1057: {"age": 25, "sex": "Male"},
    1058: {"age": 36, "sex": "Female"},
    1059: {"age": 21, "sex": "Male"},
    1060: {"age": 28, "sex": "Female"},
    1061: {"age": 51, "sex": "Female"},
    1062: {"age": 56, "sex": "Male"},
    1063: {"age": 33, "sex": "Female"},
    1064: {"age": 53, "sex": "Male"},
    1065: {"age": 38, "sex": "Male"},
    1066: {"age": 25, "sex": "Male"},
    1067: {"age": 66, "sex": "Male"},
    1068: {"age": 34, "sex": "Male"},
    1069: {"age": 27, "sex": "Male"},
    1070: {"age": 25, "sex": "Male"},
    1071: {"age": 41, "sex": "Male"},
    1072: {"age": 33, "sex": "Female"},
    1073: {"age": 24, "sex": "Female"},
    1074: {"age": 31, "sex": "Female"},
    1075: {"age": 40, "sex": "Female"},
    1076: {"age": 25, "sex": "Female"},
    1077: {"age": 20, "sex": "Male"},
    1078: {"age": 21, "sex": "Female"},
    1079: {"age": 21, "sex": "Female"},
    1080: {"age": 21, "sex": "Male"},
    1081: {"age": 30, "sex": "Male"},
    1082: {"age": 20, "sex": "Female"},
    1083: {"age": 45, "sex": "Male"},
    1084: {"age": 46, "sex": "Female"},
    1085: {"age": 34, "sex": "Male"},
    1086: {"age": 33, "sex": "Male"},
    1087: {"age": 62, "sex": "Male"},
    1088: {"age": 23, "sex": "Male"},
    1089: {"age": 24, "sex": "Female"},
    1090: {"age": 50, "sex": "Male"},
    1091: {"age": 29, "sex": "Female"},
}

# Neutral sentence codes used in CREMA-D (we pick one per actor for consistency)
NEUTRAL_SENTENCES = ["IEO", "DFA", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]

RAW_BASE_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"


def age_to_bracket(age: int) -> str:
    """Map raw age to the bracket format our model uses."""
    if age < 25:
        return "18-25"
    elif age < 35:
        return "25-35"
    elif age < 50:
        return "35-50"
    elif age < 65:
        return "50-65"
    else:
        return "65+"


def select_actors(n: int = 50) -> list:
    """
    Select a stratified subset of actors ensuring gender & age diversity.
    We pick ~equal male/female and spread across age brackets.
    """
    males = [aid for aid, d in DEMOGRAPHICS.items() if d["sex"] == "Male"]
    females = [aid for aid, d in DEMOGRAPHICS.items() if d["sex"] == "Female"]
    
    random.seed(42)  # Reproducible
    random.shuffle(males)
    random.shuffle(females)
    
    # Take equal from each gender (or as close as possible)
    n_each = n // 2
    selected = males[:n_each] + females[:n_each]
    
    # If n is odd, add one more from whichever has more
    if len(selected) < n:
        remaining = [a for a in list(DEMOGRAPHICS.keys()) if a not in selected]
        selected.extend(remaining[:n - len(selected)])
    
    return sorted(selected)


def download_subset(output_dir: str, n_actors: int = 50):
    """Download one neutral WAV per selected actor."""
    os.makedirs(output_dir, exist_ok=True)
    
    actors = select_actors(n_actors)
    metadata_rows = []
    
    print(f"Downloading {len(actors)} CREMA-D samples to {output_dir}...")
    
    for actor_id in actors:
        demo = DEMOGRAPHICS[actor_id]
        
        # Try to download a neutral clip: {ActorID}_IEO_NEU_XX.wav
        filename = f"{actor_id}_IEO_NEU_XX.wav"
        url = f"{RAW_BASE_URL}/{filename}"
        dest = os.path.join(output_dir, filename)
        
        if os.path.exists(dest):
            print(f"  [skip] {filename} (exists)")
        else:
            try:
                urlretrieve(url, dest)
                print(f"  [ok]   {filename}")
            except Exception as e:
                print(f"  [fail] {filename}: {e}")
                # Try alternate sentence
                for sent in NEUTRAL_SENTENCES[1:]:
                    alt_filename = f"{actor_id}_{sent}_NEU_XX.wav"
                    alt_url = f"{RAW_BASE_URL}/{alt_filename}"
                    try:
                        urlretrieve(alt_url, dest.replace(filename, alt_filename))
                        filename = alt_filename
                        print(f"  [ok]   {alt_filename} (fallback)")
                        break
                    except Exception:
                        continue
                else:
                    print(f"  [SKIP] Could not download any clip for actor {actor_id}")
                    continue
            
            # Be nice to GitHub - small delay
            time.sleep(0.3)
        
        metadata_rows.append({
            "filename": filename,
            "actor_id": actor_id,
            "gender": demo["sex"].lower(),
            "age": demo["age"],
            "age_bracket": age_to_bracket(demo["age"]),
            "emotion": "neutral",
        })
    
    # Write metadata.tsv
    tsv_path = os.path.join(output_dir, "metadata.tsv")
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "actor_id", "gender", "age", "age_bracket", "emotion"], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)
    
    print(f"\nDone! {len(metadata_rows)} samples downloaded.")
    print(f"Metadata written to: {tsv_path}")
    
    # Print summary stats
    males = sum(1 for r in metadata_rows if r["gender"] == "male")
    females = sum(1 for r in metadata_rows if r["gender"] == "female")
    brackets = {}
    for r in metadata_rows:
        b = r["age_bracket"]
        brackets[b] = brackets.get(b, 0) + 1
    
    print(f"\nGender split: {males} male, {females} female")
    print(f"Age bracket distribution:")
    for b in sorted(brackets.keys()):
        print(f"  {b}: {brackets[b]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CREMA-D evaluation subset")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tests/input-audio-files/CREMA-D",
        help="Output directory for audio files and metadata.tsv"
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=50,
        help="Number of actors to sample (max 91)"
    )
    args = parser.parse_args()
    download_subset(args.output, args.n)
