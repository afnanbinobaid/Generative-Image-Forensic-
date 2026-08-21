"""
validate_port.py - proves extract_features.py agrees with feature_extractor.m

The model is trained on MATLAB-computed features, so the Python port has to
reproduce them or the live demo will score images against a subtly different
feature space and the decision threshold will quietly stop meaning what it
meant. This script re-extracts a sample of the dataset in Python and compares
against the MATLAB CSV, feature by feature.

    python validate_port.py [n_samples]

Anything at 1e-6 or below is floating-point noise. Anything larger is a real
disagreement and is reported per column so it can be traced.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from extract_features import extract_features, IMG_SIZE

CSV_PATH   = Path("dataset_crop.csv")
NAMES_PATH = Path("filenames_crop.txt")
BLOCKS = [("A spatial", 0, 51), ("B wavelet", 51, 198), ("C fft", 198, 218),
          ("D residual", 218, 227), ("E corr", 227, 230)]


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    for p in (CSV_PATH, NAMES_PATH):
        if not p.exists():
            sys.exit(f"ERROR: cannot find {p.resolve()}\\n"
                     f"Run this from the folder holding the MATLAB output.")

    M = np.loadtxt(CSV_PATH, delimiter=",")
    names = NAMES_PATH.read_text().splitlines()
    if len(names) != M.shape[0]:
        sys.exit(f"ERROR: {NAMES_PATH} has {len(names)} lines, CSV has "
                 f"{M.shape[0]} rows - they must stay row-aligned.")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(names), size=min(n_samples, len(names)), replace=False)

    print(f"Comparing {len(idx)} images: Python port vs MATLAB output\\n")

    worst = np.zeros(230)
    checked = skipped_small = missing = 0

    for k, i in enumerate(idx, 1):
        path = Path(names[i])
        if not path.exists():
            missing += 1
            continue
        h, w = np.asarray(Image.open(path).convert("RGB")).shape[:2]
        if h < IMG_SIZE or w < IMG_SIZE:
            # these take the upscale path, where the two resize implementations
            # legitimately differ; they are reported separately, not as failures
            skipped_small += 1
            continue

        py, ml = extract_features(path), M[i, :230]
        biggest = np.maximum(np.abs(py), np.abs(ml))
        rel = np.where(biggest < 1e-9, 0.0, np.abs(py - ml) / np.maximum(biggest, 1e-30))
        worst = np.maximum(worst, rel)
        checked += 1
        if k % 10 == 0:
            print(f"  compared {k} / {len(idx)}")

    if checked == 0:
        sys.exit("No comparable images found - are the paths in "
                 f"{NAMES_PATH} still valid on this machine?")

    print(f"\\n{'block':<14}{'worst rel. diff':>18}{'worst column':>14}")
    print("-" * 46)
    for name, lo, hi in BLOCKS:
        seg = worst[lo:hi]
        print(f"{name:<14}{seg.max():>18.3e}{lo + int(seg.argmax()) + 1:>14}")
    print("-" * 46)
    print(f"{'OVERALL':<14}{worst.max():>18.3e}{int(worst.argmax()) + 1:>14}")

    print(f"\\ncompared {checked} images"
          + (f", skipped {skipped_small} below {IMG_SIZE}px" if skipped_small else "")
          + (f", {missing} paths not found" if missing else ""))

    if worst.max() < 1e-6:
        print("\\nPASS - the Python port reproduces the MATLAB features.")
    else:
        over = np.flatnonzero(worst > 1e-6)
        print(f"\\nFAIL - {over.size} columns disagree by more than 1e-6:")
        for c in over[:25]:
            block = next(n for n, lo, hi in BLOCKS if lo <= c < hi)
            print(f"  col {c + 1:3d}  [{block}]  rel diff {worst[c]:.3e}")
        print("\\nDo not use the demo until this is resolved - the model would be "
              "scoring a different feature space than it was trained on.")


if __name__ == "__main__":
    main()
