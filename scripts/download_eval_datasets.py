"""
Download small evaluation subsets from two HuggingFace datasets:
  1. AppTek Call-Center Dialogues - spontaneous agent/customer conversations with gender labels
  2. Common Phone Dataset         - diverse hardware/acoustic conditions with age & gender labels

Usage:
    python scripts/download_eval_datasets.py
    python scripts/download_eval_datasets.py --n 50 --out test-audio-files

Requirements:
    pip install datasets soundfile
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("[ERROR] soundfile is not installed. Run: pip install soundfile")
    sys.exit(1)

try:
    from datasets import load_dataset, Audio
except ImportError:
    print("[ERROR] datasets is not installed. Run: pip install datasets")
    sys.exit(1)

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_audio(array: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save a numpy audio array to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, sample_rate, subtype="PCM_16")


def _safe_str(v) -> str:
    """Return a clean string for use in filenames."""
    return str(v).lower().replace(" ", "_").replace("/", "-").replace("\\", "-")


# ---------------------------------------------------------------------------
# 1. AppTek Call-Center Dialogues
# ---------------------------------------------------------------------------

def download_apptek(out_dir: Path, n_samples: int) -> list[dict]:
    """
    Download n_samples from apptek-com/apptek_callcenter_dialogues.

    Available metadata per sample:
        - speaker_id, gender, age_group, accent, role (agent/customer)
    """
    print(f"\n[AppTek] Loading dataset (streaming, will fetch first {n_samples} samples)...")
    
    # Stream so we don't download the full 128-hour dataset
    try:
        ds = load_dataset(
            "apptek-com/apptek_callcenter_dialogues",
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).cast_column("audio", Audio(sampling_rate=16_000))
    except Exception as e:
        print(f"[AppTek] Could not load dataset: {e}")
        print("[AppTek] This dataset may require accepting terms on HuggingFace Hub.")
        print("         Visit: https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues")
        return []

    metadata = []
    count = 0

    for sample in ds:
        if count >= n_samples:
            break

        audio = sample["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]

        # Build a descriptive filename from available fields
        gender = _safe_str(sample.get("gender", "unknown"))
        age    = _safe_str(sample.get("age_group", "unknown"))
        role   = _safe_str(sample.get("role", "unknown"))
        spk_id = _safe_str(sample.get("speaker_id", count))
        fname  = f"{spk_id}_{role}_{gender}_{age}.wav"

        out_path = out_dir / fname
        save_audio(array, sr, out_path)

        meta_row = {
            "filename": fname,
            "speaker_id": sample.get("speaker_id"),
            "gender": sample.get("gender"),
            "age_group": sample.get("age_group"),
            "accent": sample.get("accent"),
            "role": sample.get("role"),
        }
        metadata.append(meta_row)
        count += 1
        print(f"  [{count}/{n_samples}] Saved: {fname}")

    return metadata


# ---------------------------------------------------------------------------
# 2. Common Phone Dataset
# ---------------------------------------------------------------------------

def download_commonphone(out_dir: Path, n_samples: int) -> list[dict]:
    """
    Download n_samples from pklumpp/CommonPhoneDataset (English subset).

    Available metadata per sample:
        - client_id, gender, age, locale, sentence
    """
    print(f"\n[CommonPhone] Loading dataset (streaming, will fetch first {n_samples} samples)...")

    try:
        ds = load_dataset(
            "pklumpp/CommonPhoneDataset",
            "en",                          # English subset
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).cast_column("audio", Audio(sampling_rate=16_000))
    except Exception as e:
        print(f"[CommonPhone] Could not load 'en' config. Trying default config... ({e})")
        try:
            ds = load_dataset(
                "pklumpp/CommonPhoneDataset",
                split="train",
                streaming=True,
                trust_remote_code=True,
            ).cast_column("audio", Audio(sampling_rate=16_000))
        except Exception as e2:
            print(f"[CommonPhone] Failed to load dataset: {e2}")
            return []

    metadata = []
    count = 0

    for sample in ds:
        if count >= n_samples:
            break

        audio = sample["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]

        gender = _safe_str(sample.get("gender", "unknown"))
        age    = _safe_str(sample.get("age", "unknown"))
        cid    = _safe_str(sample.get("client_id", count))[:12]  # truncate long hash
        fname  = f"{cid}_{gender}_{age}.wav"

        out_path = out_dir / fname
        save_audio(array, sr, out_path)

        meta_row = {
            "filename": fname,
            "client_id": sample.get("client_id"),
            "gender": sample.get("gender"),
            "age": sample.get("age"),
            "locale": sample.get("locale"),
            "sentence": sample.get("sentence"),
        }
        metadata.append(meta_row)
        count += 1
        print(f"  [{count}/{n_samples}] Saved: {fname}")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download small evaluation subsets from AppTek and CommonPhone datasets."
    )
    parser.add_argument(
        "--n", type=int, default=30,
        help="Number of samples to download per dataset (default: 30)"
    )
    parser.add_argument(
        "--out", type=str, default="test-audio-files",
        help="Root output directory (default: test-audio-files)"
    )
    args = parser.parse_args()

    root = Path(args.out)
    apptek_dir      = root / "AppTek-CallCenter"
    commonphone_dir = root / "CommonPhone"

    apptek_dir.mkdir(parents=True, exist_ok=True)
    commonphone_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Downloading {args.n} samples per dataset → {root}")
    print("=" * 60)

    # -- AppTek --
    apptek_meta = download_apptek(apptek_dir, args.n)
    if apptek_meta:
        meta_path = apptek_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(apptek_meta, f, indent=2)
        print(f"\n[AppTek] Done! {len(apptek_meta)} samples → {apptek_dir}")
        print(f"         Metadata → {meta_path}")
    else:
        print("\n[AppTek] No samples downloaded.")

    # -- CommonPhone --
    cp_meta = download_commonphone(commonphone_dir, args.n)
    if cp_meta:
        meta_path = commonphone_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(cp_meta, f, indent=2)
        print(f"\n[CommonPhone] Done! {len(cp_meta)} samples → {commonphone_dir}")
        print(f"              Metadata → {meta_path}")
    else:
        print("\n[CommonPhone] No samples downloaded.")

    print("\n" + "=" * 60)
    print("Next step: run evaluation")
    print("  python scripts/run_evaluation.py test-audio-files/AppTek-CallCenter")
    print("  python scripts/run_evaluation.py test-audio-files/CommonPhone")
    print("=" * 60)


if __name__ == "__main__":
    main()
