"""
score_folder.py - run the trained detector over a folder of images

For testing the detector against a generator it has never seen: point it at a
folder, say what those images actually are, and get the accuracy plus a score
distribution. This is the honest generalisation test - none of these images
were involved in training.

    python score_folder.py <folder> ai      # folder of generated images
    python score_folder.py <folder> real    # folder of photographs
    python score_folder.py <folder>         # no ground truth, just predictions

Results go to score_<folder>.csv, one row per image with its score and verdict.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

from extract_features import extract_features, list_images, size_warning

MODEL_PATH = Path("model.joblib")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    truth = sys.argv[2].lower() if len(sys.argv) > 2 else None
    if truth not in (None, "ai", "real"):
        sys.exit("Second argument must be 'ai', 'real', or omitted.")

    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: {MODEL_PATH.resolve()} not found. Run train_model.py first.")
    if not folder.is_dir():
        sys.exit(f"ERROR: {folder.resolve()} is not a folder.")

    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], float(bundle["threshold"])

    files = list_images(folder)
    if not files:
        sys.exit(f"No .jpg/.jpeg/.png images found in {folder}")
    print(f"Scoring {len(files)} images from {folder}")
    print(f"Model threshold: {threshold:.4f}\n")

    scores, kept, small, skipped = [], [], 0, 0
    for i, f in enumerate(files, 1):
        try:
            if size_warning(f):
                small += 1
            v = extract_features(f)
            scores.append(float(model.predict_proba(v.reshape(1, -1))[0, 1]))
            kept.append(f)
        except Exception as exc:
            skipped += 1
            print(f"SKIPPED: {f.name}  ({exc})", file=sys.stderr)
        if i % 50 == 0:
            print(f"  {i} / {len(files)}")

    scores = np.asarray(scores)
    pred_ai = scores >= threshold

    print(f"\n{'=' * 58}")
    print(f"Scored          : {len(scores)}"
          + (f"   (skipped {skipped})" if skipped else ""))
    if small:
        print(f"Below 256px     : {small}  <- unreliable, see the size warning")
    print(f"Predicted AI    : {pred_ai.sum()}  ({pred_ai.mean():.1%})")
    print(f"Predicted real  : {(~pred_ai).sum()}  ({(~pred_ai).mean():.1%})")
    print(f"Score mean/med  : {scores.mean():.3f} / {np.median(scores):.3f}")
    print(f"Score range     : {scores.min():.3f} - {scores.max():.3f}")

    if truth is not None:
        correct = pred_ai if truth == "ai" else ~pred_ai
        print(f"\nGround truth    : all {truth.upper()}")
        print(f"Accuracy        : {correct.mean():.1%}  "
              f"({correct.sum()} / {len(correct)})")
        print("\nThis is a clean generalisation number - no image here was in "
              "training.")

    out = Path(f"score_{folder.name}.csv")
    with out.open("w") as fh:
        fh.write("filename,score,verdict\n")
        for f, sc in zip(kept, scores):
            fh.write(f"{f.name},{sc:.6f},{'AI' if sc >= threshold else 'real'}\n")
    print(f"\nPer-image results: {out.resolve()}")


if __name__ == "__main__":
    main()
