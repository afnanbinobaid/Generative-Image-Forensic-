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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, roc_auc_score,
                             roc_curve)

from feature_names import describe_feature
from split_utils import make_split, make_split_3way, describe_split


def _fit_calibrator(fitted_model, X_cal, y_cal, method):
    """Wrap an already-fitted model in a calibrator, without refitting it.

    scikit-learn changed how this is expressed: up to 1.5 it was
    CalibratedClassifierCV(model, cv="prefit"); from 1.6 that is deprecated in
    favour of wrapping the estimator in FrozenEstimator, and 1.8 removed the
    old spelling outright. Both are supported here so the script runs whichever
    version happens to be installed.

    Either way the point is the same: the base model is NOT refitted. Only the
    score -> probability map is learned, from the held-out calibration slice.
    """
    try:                                    # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(fitted_model), method=method)
    except ImportError:                     # scikit-learn <= 1.5
        calibrator = CalibratedClassifierCV(
            fitted_model, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def _calibrate_prefit(fitted_model, X_cal, y_cal, cal_names=None, seed=42):
    """Calibrate, choosing the method by measurement rather than by assumption.

    Isotonic regression is flexible and usually wins on a large calibration
    slice, but being non-parametric it can overfit a small one and leave the
    probabilities worse than it found them. Sigmoid (Platt) fits two parameters
    and is far harder to overfit, but cannot correct a distortion that is not
    sigmoid-shaped. Which one helps is an empirical question about this dataset,
    so it is answered here instead of guessed.

    The calibration slice is split in two (grouped, so an image's compressed
    copies stay together): candidates are fitted on one half and scored on the
    other, and the winner - which may be no calibration at all - is refitted on
    the whole slice. The test set takes no part in this choice, so the reported
    test ECE remains an honest estimate rather than the best of three tries.
    """
    y_cal = np.asarray(y_cal)
    a_idx, b_idx, _ = make_split(y_cal, cal_names, seed=seed, test_frac=0.40)
    X_a, y_a = X_cal[a_idx], y_cal[a_idx]
    X_b, y_b = X_cal[b_idx], y_cal[b_idx]

    # "none" is a real candidate: if the model is already well calibrated,
    # the right action is to leave it alone.
    trials = {"none": expected_calibration_error(
        y_b, fitted_model.predict_proba(X_b)[:, 1])}
    for method in ("isotonic", "sigmoid"):
        try:
            cand = _fit_calibrator(fitted_model, X_a, y_a, method)
            trials[method] = expected_calibration_error(
                y_b, cand.predict_proba(X_b)[:, 1])
        except Exception as exc:
            print(f"    {method}: unavailable ({exc})")

    best = min(trials, key=trials.get)
    print("    method selection (ECE on a held-out half of the calibration slice):")
    for name, score in sorted(trials.items(), key=lambda kv: kv[1]):
        print(f"      {name:<9} {score:.4f}{'   <- chosen' if name == best else ''}")

    if best == "none":
        return None, "none"
    return _fit_calibrator(fitted_model, X_cal, y_cal, best), best


def expected_calibration_error(y_true, prob, n_bins=15):
    """Mean gap between predicted confidence and observed frequency.

    Bins predictions by score and asks, in each bin, whether the images the
    model called 70% AI really were AI about 70% of the time. 0 is perfect.
    This is the number that says whether a score can be read as a probability,
    which accuracy and AUC both ignore entirely - a model can rank flawlessly
    (AUC 1.0) while every score it reports is wrong as a probability.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(y_true[in_bin].mean() - prob[in_bin].mean())
    return float(ece)


def reliability_table(y_true, prob, n_bins=10):
    """Rows of (range, count, mean predicted, actual frequency) for printing."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        rows.append((lo, hi, n, float(prob[in_bin].mean()),
                     float(y_true[in_bin].mean())))
    return rows

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

    # The generator is the second underscore-separated token of an AI filename.
    # If a folder is named differently that guess degenerates into one "generator"
    # per image, which would flood the demo's footer and bloat model.joblib, so
    # an implausible result is discarded rather than stored.
    if len(set(generators)) > 12:
        print(f"  NOTE: filename parsing produced {len(set(generators))} distinct "
              f"generator names, which is implausible.\n"
              f"        Recording none - the demo will omit the 'trained on' line.")
        generators = []

    names = None
    if NAMES_PATH.exists():
        candidate = NAMES_PATH.read_text().splitlines()
        if len(candidate) == len(y):
            names = candidate

    tr_idx, cal_idx, te_idx, grouped = make_split_3way(y, names, seed=RANDOM_SEED)
    print(describe_split(y, tr_idx, te_idx, grouped))
    print(f"  calibrate {len(cal_idx)} rows "
          f"({int((y[cal_idx] == 0).sum())} real / {int((y[cal_idx] == 1).sum())} AI)"
          " - held out from fitting, used only to map scores to probabilities")
    X_tr, X_cal, X_te = X[tr_idx], X[cal_idx], X[te_idx]
    y_tr, y_cal, y_te = y[tr_idx], y[cal_idx], y[te_idx]

    print("\nTraining ...")
    base_model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    base_model.fit(X_tr, y_tr)

    # Probability calibration. Gradient boosting optimises a ranking loss, not
    # calibrated probabilities: when the classes separate well its scores pile
    # up near 0 and 1, which is why a misclassified web photograph could come
    # back at 0.988 rather than at an honest 0.55. Isotonic regression fits a
    # monotonic score -> probability map on the held-out calibration slice.
    #
    # Because the map is monotonic it cannot change the ranking, so ROC-AUC is
    # identical before and after and no accuracy is traded away. What changes
    # is that the number becomes readable as a probability.
    print("Calibrating probabilities on the held-out slice ...")
    cal_names = [names[i] for i in cal_idx] if names is not None else None
    calibrator, cal_method = _calibrate_prefit(base_model, X_cal, y_cal,
                                              cal_names, seed=RANDOM_SEED)
    model = calibrator if calibrator is not None else base_model

    prob_raw = base_model.predict_proba(X_te)[:, 1]
    prob = model.predict_proba(X_te)[:, 1]

    auc_raw = roc_auc_score(y_te, prob_raw)
    auc = roc_auc_score(y_te, prob)

    fpr, tpr, thr = roc_curve(y_te, prob)
    threshold = float(thr[np.argmax(tpr - fpr)])
    if not np.isfinite(threshold):
        threshold = 0.5

    acc_default = accuracy_score(y_te, (prob >= 0.5).astype(int))
    acc_tuned   = accuracy_score(y_te, (prob >= threshold).astype(int))

    ece_raw,   ece_cal   = (expected_calibration_error(y_te, prob_raw),
                            expected_calibration_error(y_te, prob))
    brier_raw, brier_cal = (brier_score_loss(y_te, prob_raw),
                            brier_score_loss(y_te, prob))

    print(f"\n  held-out ROC-AUC     : {auc:.4f}"
          f"   (uncalibrated {auc_raw:.4f} - a monotonic map cannot change it)")
    print(f"  accuracy @ 0.5       : {acc_default:.4f}")
    print(f"  chosen threshold     : {threshold:.4f}")
    print(f"  accuracy @ threshold : {acc_tuned:.4f}")

    print(f"\n  Calibration ({cal_method}) - lower is better")
    print(f"    expected calibration error : {ece_raw:.4f} -> {ece_cal:.4f}")
    print(f"    Brier score                : {brier_raw:.4f} -> {brier_cal:.4f}")
    if cal_method == "none":
        print("    the model was already better calibrated than either method"
              " could make it,\n    so scores are left untouched")
    elif ece_cal > ece_raw:
        print("    WARNING: calibration was selected on the calibration slice but"
              " did not\n    hold up on test. Treat the reported probabilities"
              " with caution and\n    consider enlarging the calibration slice.")

    print("\n  Reliability - does a score of X mean an X chance of being AI?")
    print(f"    {'score range':<16}{'images':>8}{'predicted':>11}{'actual':>9}")
    for lo, hi, n, pred, actual in reliability_table(y_te, prob):
        print(f"    {f'{lo:.1f} - {hi:.1f}':<16}{n:>8}{pred:>11.3f}{actual:>9.3f}")

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
        # The calibrated wrapper owns predict_proba, so every score the demo
        # and the folder scorer report is a real probability.
        "model": model,
        # The bare tree ensemble, kept for SHAP: shap's fast exact TreeExplainer
        # needs the trees themselves, and the calibrated wrapper hides them.
        # Calibration is a monotonic remap of the score, so it cannot reorder
        # feature contributions - an explanation of the base model is still a
        # correct explanation of which features drove the calibrated verdict.
        "base_model": base_model,
        "calibrated": cal_method != "none",
        "calibration_method": cal_method,
        "threshold": threshold,
        "roc_auc": float(auc),
        "accuracy": float(acc_tuned),
        "ece": float(ece_cal),
        "ece_uncalibrated": float(ece_raw),
        "brier": float(brier_cal),
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
