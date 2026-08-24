"""
Eval harness — Dialflo Voice Attribute Inference Service.

Runs the inference pipeline against a Mozilla Common Voice subset and prints:
  1. Overall accuracy per attribute (gender, age bracket)
  2. Accuracy broken out by accent/locale subgroup (the India-calibration signal)
  3. Confidence calibration: Expected Calibration Error (ECE) per attribute

Usage:
  python eval/run_eval.py --dataset-path /path/to/common-voice --max-samples 200

Mozilla Common Voice download:
  https://commonvoice.mozilla.org/en/datasets
  Download the TSV + clips for 'en' and 'hi' (Hindi) locales.
  Required TSV columns: client_id, path, sentence, age, gender, accents

Design note on India calibration:
  The model is trained on VoxCeleb and similar predominantly Western-English
  datasets. This harness intentionally breaks accuracy out by locale/accent
  so the performance gap on Indian-accented English (en-IN) and Hindi (hi-IN)
  is visible, not hidden in an aggregate number. That gap is a known limitation
  documented in DESIGN_DECISIONS.md — the right response is to surface it and
  propose a fix (fine-tune on Common Voice hi + Indian English), not pretend
  it doesn't exist.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Age bracket mapping: CommonVoice age labels → our schema
CV_AGE_MAP = {
    "teens":    "18-30",
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "fifties":  "46-60",
    "sixties":  "60+",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}

# Gender mapping: CommonVoice gender labels → our schema
CV_GENDER_MAP = {
    "male":   "male",
    "female": "female",
    "other":  "unknown",
}


def load_cv_samples(dataset_path: str, locale: str, max_samples: int) -> list[dict]:
    """
    Load validated Common Voice samples for the given locale.
    Returns a list of dicts: {path, age_label, gender_label, accent, locale}
    """
    validated_tsv = Path(dataset_path) / locale / "validated.tsv"
    clips_dir     = Path(dataset_path) / locale / "clips"

    if not validated_tsv.exists():
        print(f"  [WARN] {validated_tsv} not found — skipping locale {locale}")
        return []

    samples = []
    with open(validated_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if len(samples) >= max_samples:
                break
            age    = row.get("age", "").strip()
            gender = row.get("gender", "").strip()
            accent = row.get("accents", "").strip() or locale
            clip   = clips_dir / row["path"]
            if not clip.exists():
                continue
            if not age or not gender:
                continue   # skip samples without labels
            samples.append({
                "path":   str(clip),
                "age":    CV_AGE_MAP.get(age, None),
                "gender": CV_GENDER_MAP.get(gender, "unknown"),
                "accent": accent,
                "locale": locale,
            })
    return samples


def _load_audio(path: str) -> Optional[np.ndarray]:
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16_000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16_000)
        return data
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def _ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """
    Expected Calibration Error — measures how well confidence scores predict
    actual accuracy. A perfectly calibrated model has ECE = 0.
    Lower is better. 0.1 means on average confidence is off by 10 percentage points.
    """
    if not confidences:
        return float("nan")
    confs = np.array(confidences)
    cors  = np.array(corrects, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for lo, hi in zip(bin_edges, bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if not mask.any():
            continue
        bin_acc  = cors[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def run_eval(dataset_path: str, locales: list[str], max_samples: int):
    from app import audio_quality, decision_policy, inference

    all_samples = []
    for locale in locales:
        locale_samples = load_cv_samples(dataset_path, locale, max_samples)
        print(f"Loaded {len(locale_samples)} samples for locale '{locale}'")
        all_samples.extend(locale_samples)

    if not all_samples:
        print("\nNo samples loaded. Check --dataset-path and locale directories.")
        print("See samples/SOURCING.md for download instructions.")
        return

    # Per-attribute accumulators
    gender_results  = defaultdict(lambda: {"correct": [], "conf": []})
    age_results     = defaultdict(lambda: {"correct": [], "conf": []})
    skipped = 0

    print(f"\nRunning inference on {len(all_samples)} samples...")
    t_start = time.monotonic()

    for i, sample in enumerate(all_samples):
        samples_arr = _load_audio(sample["path"])
        if samples_arr is None or len(samples_arr) / 16_000 < 1.0:
            skipped += 1
            continue

        quality = audio_quality.assess(samples_arr, 16_000)
        if quality.quality == "insufficient":
            skipped += 1
            continue

        try:
            pred = inference.predict(samples_arr, 16_000)
        except Exception as e:
            print(f"  [WARN] inference failed on {sample['path']}: {e}")
            skipped += 1
            continue

        locale_key = sample["locale"]

        # Gender
        if sample["gender"] != "unknown":
            correct = pred.gender_label == sample["gender"]
            gender_results[locale_key]["correct"].append(correct)
            gender_results[locale_key]["conf"].append(pred.gender_confidence)
            gender_results["ALL"]["correct"].append(correct)
            gender_results["ALL"]["conf"].append(pred.gender_confidence)

        # Age bracket
        if sample["age"] is not None:
            correct = pred.age_bracket == sample["age"]
            age_results[locale_key]["correct"].append(correct)
            age_results[locale_key]["conf"].append(pred.age_confidence)
            age_results["ALL"]["correct"].append(correct)
            age_results["ALL"]["conf"].append(pred.age_confidence)

        if (i + 1) % 20 == 0:
            elapsed = time.monotonic() - t_start
            print(f"  [{i+1}/{len(all_samples)}] elapsed={elapsed:.1f}s")

    total_elapsed = time.monotonic() - t_start
    print(f"\nDone. {skipped} samples skipped (short/corrupt/insufficient).")
    print(f"Total eval time: {total_elapsed:.1f}s\n")

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    print("=" * 72)
    print(f"{'GENDER ACCURACY':^72}")
    print("=" * 72)
    print(f"{'Locale':<20} {'N':>6} {'Accuracy':>10} {'ECE':>8}")
    print("-" * 72)

    for locale in ["ALL"] + [l for l in sorted(gender_results) if l != "ALL"]:
        r = gender_results.get(locale)
        if not r or not r["correct"]:
            continue
        n   = len(r["correct"])
        acc = sum(r["correct"]) / n
        ece = _ece(r["conf"], r["correct"])
        flag = " ← India gap?" if locale in ("hi", "hi-IN", "en-IN") else ""
        print(f"  {locale:<18} {n:>6} {acc:>10.1%} {ece:>8.3f}{flag}")

    print()
    print("=" * 72)
    print(f"{'AGE BRACKET ACCURACY':^72}")
    print("=" * 72)
    print(f"{'Locale':<20} {'N':>6} {'Accuracy':>10} {'ECE':>8}")
    print("-" * 72)

    for locale in ["ALL"] + [l for l in sorted(age_results) if l != "ALL"]:
        r = age_results.get(locale)
        if not r or not r["correct"]:
            continue
        n   = len(r["correct"])
        acc = sum(r["correct"]) / n
        ece = _ece(r["conf"], r["correct"])
        flag = " ← India gap?" if locale in ("hi", "hi-IN", "en-IN") else ""
        print(f"  {locale:<18} {n:>6} {acc:>10.1%} {ece:>8.3f}{flag}")

    print()
    print("NOTE: ECE closer to 0.0 = better-calibrated confidence scores.")
    print("NOTE: Rows marked '← India gap?' are the key differentiator.")
    print("      Lower accuracy on these rows vs 'en' confirms the model's")
    print("      Western-dataset bias and motivates fine-tuning on Indian audio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eval harness for Dialflo Voice Attribute Inference"
    )
    parser.add_argument(
        "--dataset-path", required=True,
        help="Path to extracted Mozilla Common Voice dataset directory"
    )
    parser.add_argument(
        "--locales", nargs="+", default=["en", "hi"],
        help="Common Voice locale codes to evaluate (default: en hi)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Max samples per locale (default: 200)"
    )
    args = parser.parse_args()
    run_eval(args.dataset_path, args.locales, args.max_samples)
