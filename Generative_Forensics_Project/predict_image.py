"""
predict_image.py - classify one image from the features MATLAB extracted

The demo splits the work the way the project does: MATLAB measures the image,
Python decides what it is. demo_image.m writes a one-row feature CSV and calls
this script; the verdict comes back here.

    python predict_image.py demo_features.csv

The first two lines of output are machine-readable so MATLAB can place the
score on its gauge; everything after them is the human-facing report.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

from feature_names import describe_feature

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python predict_image.py <features.csv>")

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        sys.exit(f"ERROR: cannot find {csv_path}")
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: cannot find {MODEL_PATH}\n"
                 f"Run  python train_model.py  first.")

    vec = np.loadtxt(csv_path, delimiter=",").ravel()
    if vec.size != 230:
        sys.exit(f"ERROR: expected 230 features, found {vec.size}")

    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], float(bundle["threshold"])

    prob = float(model.predict_proba(vec.reshape(1, -1))[0, 1])
    is_ai = prob >= threshold

    # distance from the threshold, scaled by the room available on that side
    room = (1 - threshold) if is_ai else threshold
    margin = abs(prob - threshold) / max(room, 1e-9)
    strength = "strong" if margin > 0.6 else "moderate" if margin > 0.25 else "weak"
    verdict = "AI-GENERATED" if is_ai else "REAL PHOTOGRAPH"

    # machine-readable header for MATLAB
    print(f"SCORE {prob:.6f}")
    print(f"THRESHOLD {threshold:.6f}")
    print(f"VERDICT {verdict}")

    print("=" * 62)
    print(f"  Verdict    : {verdict}")
    print(f"  Score      : {prob:.4f}   (threshold {threshold:.4f})")
    print(f"  Separation : {strength}")
    print("=" * 62)

    m_real, m_ai, d = bundle["mean_real"], bundle["mean_ai"], bundle["cohens_d"]
    print(f"\n  {'measurement':<34}{'this':>11}{'real':>11}{'AI':>11}{'leans':>8}")
    print("  " + "-" * 73)
    for c in bundle["top_features"][:8]:
        c = int(c)
        v, r, a = vec[c], m_real[c], m_ai[c]
        leans = "real" if abs(v - r) < abs(v - a) else "AI"
        print(f"  {describe_feature(c):<34}{v:>11.4g}{r:>11.4g}{a:>11.4g}{leans:>8}")

    print(f"\n  Model: held-out AUC {bundle['roc_auc']:.3f}, "
          f"accuracy {bundle['accuracy']:.1%}"
          + (f", trained on {', '.join(bundle['generators'])}"
             if bundle.get("generators") else ""))


if __name__ == "__main__":
    main()
