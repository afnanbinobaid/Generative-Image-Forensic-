"""
train_model.py - fits the detector and saves it for the live demo

MATLAB extracts the features; Python owns everything to do with the model. This
reads dataset_crop.csv, trains a classifier, picks a decision threshold on
held-out data, and writes model.joblib. predict_image.py and score_folder.py
load that file, so the demo starts instantly instead of retraining.

    python train_model.py

The saved threshold is not 0.5: Experiment B in classify.py showed that is
exactly where cross-generator accuracy is lost, so it is chosen to maximise
balanced accuracy on held-out data instead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from feature_names import describe_feature
from split_utils import make_split, describe_split

CSV_PATH    = Path("dataset_crop.csv")
NAMES_PATH  = Path("filenames_crop.txt")
OUT_PATH    = Path("model.joblib")
RANDOM_SEED = 42


def main():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: cannot find {CSV_PATH.resolve()}\n"
                 f"Run feature_extractor.m in MATLAB first.")

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
                if y[i] == 1:
                    parts = Path(names[i].replace("\\", "/")).stem.split("_")
                    if len(parts) >= 2:
                        generators.append(parts[1].lower())

    names = None
    if NAMES_PATH.exists():
        candidate = NAMES_PATH.read_text().splitlines()
        if len(candidate) == len(y):
            names = candidate

    tr_idx, te_idx, grouped = make_split(y, names, seed=RANDOM_SEED)
    print(describe_split(y, tr_idx, te_idx, grouped))
    X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    print("\nTraining ...")
    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob)

    fpr, tpr, thr = roc_curve(y_te, prob)
    threshold = float(thr[np.argmax(tpr - fpr)])
    if not np.isfinite(threshold):
        threshold = 0.5

    acc_default = accuracy_score(y_te, (prob >= 0.5).astype(int))
    acc_tuned   = accuracy_score(y_te, (prob >= threshold).astype(int))

    print(f"\n  held-out ROC-AUC     : {auc:.4f}")
    print(f"  accuracy @ 0.5       : {acc_default:.4f}")
    print(f"  chosen threshold     : {threshold:.4f}")
    print(f"  accuracy @ threshold : {acc_tuned:.4f}")

    # Reference statistics so the demo can justify a verdict: for the most
    # discriminative features, what a typical real and a typical AI image
    # measure, against which one image can be compared.
    A, B = X[y == 0], X[y == 1]
    mean_real, mean_ai = A.mean(axis=0), B.mean(axis=0)
    std_real, std_ai = A.std(axis=0), B.std(axis=0)
    pooled = np.sqrt((A.var(axis=0) + B.var(axis=0)) / 2)
    d = np.abs(mean_real - mean_ai) / np.maximum(pooled, 1e-12)
    d[~np.isfinite(d)] = 0.0
    top = np.argsort(d)[::-1][:12]
    # Kept for models built before the SHAP explainer below - see predict_image.py.
    candidates = np.argsort(d)[::-1][:60]

    print("\nMost discriminative features (globally, not per-image):")
    for c in top[:6]:
        print(f"  col {c + 1:3d}  d={d[c]:.2f}  real={mean_real[c]:<11.4g} "
              f"AI={mean_ai[c]:<11.4g} {describe_feature(int(c))}")

    # A background sample for the demo's per-image SHAP explanation. Distance
    # to the class mean over a fixed feature pool (the old approach) is only a
    # heuristic proxy for what actually drove one image's score - post-fix the
    # model is measurably non-linear (see classify.py's LR-vs-HGB gap), so that
    # heuristic and the real decision now diverge often enough to mislead. SHAP
    # values are exact per-feature contributions to this specific prediction,
    # computed against the training distribution, and sum to the model's actual
    # output - so what predict_image.py shows can never contradict the verdict.
    bg_rng = np.random.default_rng(RANDOM_SEED)
    bg_size = min(200, len(X_tr))
    bg_idx = bg_rng.choice(len(X_tr), size=bg_size, replace=False)
    background = X_tr[bg_idx]

    joblib.dump({
        "model": model,
        "threshold": threshold,
        "roc_auc": float(auc),
        "accuracy": float(acc_tuned),
        "top_features": top.astype(int),
        "candidate_features": candidates.astype(int),
        "cohens_d": d,
        "pooled_std": pooled,
        "mean_real": mean_real,
        "mean_ai": mean_ai,
        "std_real": std_real,
        "std_ai": std_ai,
        "background": background,
        "generators": sorted(set(generators)),
        "grouped_split": bool(grouped),
    }, OUT_PATH)

    print(f"\nSaved {OUT_PATH.resolve()}")
    print("Next: demo_image in MATLAB, which extracts features and calls "
          "predict_image.py for the verdict.")


if __name__ == "__main__":
    main()
