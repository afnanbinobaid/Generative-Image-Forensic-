"""
score_folder.py - score a folder of images the MATLAB extractor has measured

For testing against a generator the model has never seen. Extract the folder in
MATLAB first:

    extract_folder('E:\\path\\to\\new_images', 'new.csv')

then score it here, saying what those images actually are:

    python score_folder.py new.csv ai
    python score_folder.py new.csv real
    python score_folder.py new.csv          (predictions only)

None of these images took part in training, so the accuracy reported is a clean
generalisation figure.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    csv_path = Path(sys.argv[1])
    truth = sys.argv[2].lower() if len(sys.argv) > 2 else None
    if truth not in (None, "ai", "real"):
        sys.exit("Second argument must be 'ai', 'real', or omitted.")
    if not csv_path.exists():
        sys.exit(f"ERROR: cannot find {csv_path}")
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: cannot find {MODEL_PATH}. Run train_model.py first.")

    X = np.loadtxt(csv_path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # extract_folder.m writes features only; tolerate a trailing label column.
    if X.shape[1] == 231:
        X = X[:, :230]
    if X.shape[1] != 230:
        sys.exit(f"ERROR: expected 230 feature columns, found {X.shape[1]}")

    bundle = joblib.load(MODEL_PATH)
    scores = bundle["model"].predict_proba(X)[:, 1]
    threshold = float(bundle["threshold"])
    pred_ai = scores >= threshold

    names_path = csv_path.with_suffix(".filenames.txt")
    names = (names_path.read_text().splitlines()
             if names_path.exists() and
             len(names_path.read_text().splitlines()) == len(scores)
             else [f"row {i+1}" for i in range(len(scores))])

    print(f"Scored {len(scores)} images from {csv_path.name}")
    print(f"Model threshold: {threshold:.4f}\n")
    print("=" * 58)
    print(f"Predicted AI    : {pred_ai.sum()}  ({pred_ai.mean():.1%})")
    print(f"Predicted real  : {(~pred_ai).sum()}  ({(~pred_ai).mean():.1%})")
    print(f"Score mean/med  : {scores.mean():.3f} / {np.median(scores):.3f}")
    print(f"Score range     : {scores.min():.3f} - {scores.max():.3f}")

    if truth is not None:
        correct = pred_ai if truth == "ai" else ~pred_ai
        print(f"\nGround truth    : all {truth.upper()}")
        print(f"Accuracy        : {correct.mean():.1%}  "
              f"({correct.sum()} / {len(correct)})")
        print("\nNo image here took part in training, so this is a clean "
              "generalisation figure.")

    out = csv_path.with_name(f"scores_{csv_path.stem}.csv")
    with out.open("w") as fh:
        fh.write("filename,score,verdict\n")
        for n, sc in zip(names, scores):
            fh.write(f"{Path(n).name},{sc:.6f},{'AI' if sc >= threshold else 'real'}\n")
    print(f"\nPer-image results: {out.resolve()}")


if __name__ == "__main__":
    main()
