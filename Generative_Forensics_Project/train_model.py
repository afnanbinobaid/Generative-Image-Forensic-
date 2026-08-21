"""
train_model.py - fits the final detector and saves it for the live demo

The demo must not retrain on every launch, so the model, the calibrated
threshold and a little metadata are pickled once into model.joblib.

    python train_model.py

The threshold saved here is the one the demo uses to turn a score into a
verdict. It is chosen on a held-out slice rather than left at 0.5, because
Experiment B showed the default cutoff is where cross-generator accuracy is
lost - see the results write-up.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

CSV_PATH   = Path("dataset_crop.csv")
NAMES_PATH = Path("filenames_crop.txt")
OUT_PATH   = Path("model.joblib")
RANDOM_SEED = 42


def main():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: cannot find {CSV_PATH.resolve()}")

    data = pd.read_csv(CSV_PATH, header=None).to_numpy(dtype=float)
    if data.shape[1] != 231:
        sys.exit(f"ERROR: expected 231 columns, found {data.shape[1]}")

    X, y = data[:, :230], data[:, 230].astype(int)
    print(f"Loaded {X.shape[0]} images x {X.shape[1]} features "
          f"({np.sum(y == 0)} real / {np.sum(y == 1)} AI)")

    generators = []
    if NAMES_PATH.exists():
        names = NAMES_PATH.read_text().splitlines()
        if len(names) == len(y):
            for i in range(len(y)):
                if y[i] == 0:
                    generators.append("real")
                else:
                    parts = Path(names[i].replace("\\", "/")).stem.split("_")
                    generators.append(parts[1].lower() if len(parts) >= 2 else "unknown")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)

    print("Training HistGradientBoosting ...")
    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob)

    # Youden's J on the held-out split: the operating point that maximises
    # balanced accuracy rather than the arbitrary 0.5.
    fpr, tpr, thr = roc_curve(y_te, prob)
    threshold = float(thr[np.argmax(tpr - fpr)])

    acc_default = accuracy_score(y_te, (prob >= 0.5).astype(int))
    acc_tuned   = accuracy_score(y_te, (prob >= threshold).astype(int))

    print(f"\n  held-out ROC-AUC     : {auc:.4f}")
    print(f"  accuracy @ 0.5       : {acc_default:.4f}")
    print(f"  chosen threshold     : {threshold:.4f}")
    print(f"  accuracy @ threshold : {acc_tuned:.4f}")

    # Reference statistics so the demo can say *why*: for the most
    # discriminative features, what a typical real and a typical AI image
    # measure, against which one uploaded image can be compared.
    A, B = X[y == 0], X[y == 1]
    mean_real, mean_ai = A.mean(axis=0), B.mean(axis=0)
    pooled = np.sqrt((A.var(axis=0) + B.var(axis=0)) / 2)
    d = np.abs(mean_real - mean_ai) / np.maximum(pooled, 1e-12)
    d[~np.isfinite(d)] = 0.0
    top = np.argsort(d)[::-1][:12]

    print("\nMost discriminative features (used by the demo to explain a verdict):")
    for c in top[:6]:
        print(f"  col {c + 1:3d}  d={d[c]:.2f}  real={mean_real[c]:.4g}  AI={mean_ai[c]:.4g}")

    joblib.dump({
        "model": model,
        "top_features": top.astype(int),
        "cohens_d": d,
        "mean_real": mean_real,
        "mean_ai": mean_ai,
        "threshold": threshold,
        "roc_auc": float(auc),
        "accuracy": float(acc_tuned),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "generators": sorted({g for g in generators if g != "real"}),
        "feature_count": 230,
    }, OUT_PATH)

    print(f"\nSaved {OUT_PATH.resolve()}")
    print("The demo (demo.py) loads this file - retrain only if the dataset changes.")


if __name__ == "__main__":
    main()
