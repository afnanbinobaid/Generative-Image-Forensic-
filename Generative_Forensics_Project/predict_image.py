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


def shap_contributions(model, vec, background):
    """Per-feature SHAP contributions to this image's score, plus how to verify them.

    Returns (contributions, base_value, space). Converting base_value +
    contributions.sum() back to a probability - through a sigmoid if space is
    "logit", directly if space is "probability" - reproduces the model's
    actual predict_proba output. That additivity is what makes this a
    faithful explanation of the verdict rather than a heuristic that can
    disagree with it.

    Tries the fast, exact tree explainer first: for a tree ensemble this is
    additive to machine precision in the model's native log-odds space and
    needs no background data (tens of milliseconds). Falls back to a
    background-sampled explainer in probability space for any model shap's
    TreeExplainer doesn't directly support - much slower (SHAP's permutation
    path triggers a one-off numba JIT compile that costs several seconds even
    on a single image, on top of ~2x230 model evaluations), so it is very
    much the fallback rather than an equivalent alternative.
    """
    import shap  # optional dependency: pip install shap

    x = vec.reshape(1, -1)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x)
        arr = np.asarray(sv)
        # (n_samples, n_features) for a single-output model, or
        # (n_samples, n_features, n_classes) - take the positive (AI) class.
        contrib = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        # expected_value is a scalar/1-element array for a single-output
        # model, or one entry per class - either way the AI class is last.
        base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        return contrib, base, "logit"
    except Exception:
        f = lambda X: model.predict_proba(X)[:, 1]
        explainer = shap.Explainer(f, background)
        sv = explainer(x)
        contrib = np.asarray(sv.values)[0]
        base = float(np.asarray(sv.base_values).reshape(-1)[0])
        return contrib, base, "probability"


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

    # Which features actually drove THIS image's score. SHAP values are exact
    # per-feature contributions to this specific prediction - unlike distance
    # to a class mean over a fixed feature pool, they are computed from the
    # model itself and sum to its actual output, so the panel below can never
    # show evidence that contradicts the verdict above it.
    m_real, m_ai = bundle["mean_real"], bundle["mean_ai"]
    background = bundle.get("background")
    contributions = base_value = space = None

    if background is not None:
        try:
            contributions, base_value, space = shap_contributions(model, vec, background)
        except Exception as exc:
            print(f"  (SHAP explanation unavailable: {exc}; "
                  f"falling back to the distance heuristic)\n")

    if contributions is not None:
        order = np.argsort(np.abs(contributions))[::-1][:8]
        unit = "logit" if space == "logit" else "prob"
        print(f"\n  {'measurement':<34}{'this':>11}{'real':>11}{'AI':>11}"
              f"{unit:>10}{'leans':>8}")
        print("  " + "-" * 83)
        for c in order:
            c = int(c)
            v, r, a, sv = vec[c], m_real[c], m_ai[c], contributions[c]
            leans = "AI" if sv > 0 else "real"
            print(f"  {describe_feature(c):<34}{v:>11.4g}{r:>11.4g}{a:>11.4g}"
                  f"{sv:>+10.4f}{leans:>8}")
        # A verifiable check, not just a claim: every feature's contribution,
        # not only the 8 shown, plus the base rate should reproduce the score.
        # Contributions are additive in the model's own scoring space, which
        # for the fast tree path is log-odds rather than probability - convert
        # through a sigmoid before comparing to predict_proba.
        total = base_value + contributions.sum()
        reconstructed = 1 / (1 + np.exp(-total)) if space == "logit" else total
        print(f"  (top 8 of 230 by |{unit} contribution|; all 230 reconstruct "
              f"a score of {reconstructed:.4f} vs actual {prob:.4f})")
    else:
        # Older model.joblib without a saved background sample: fall back to
        # the pre-SHAP heuristic, honestly labelled as an approximation rather
        # than as what drove the verdict.
        pooled = bundle.get("pooled_std")
        if pooled is None:
            pooled = np.maximum(np.abs(m_real - m_ai), 1e-12) / np.maximum(bundle["cohens_d"], 1e-6)
        pooled = np.maximum(pooled, 1e-12)

        candidates = bundle.get("candidate_features", bundle["top_features"])
        dist_real = np.abs(vec - m_real) / pooled
        dist_ai   = np.abs(vec - m_ai) / pooled
        decisiveness = dist_real[candidates] - dist_ai[candidates]
        order = candidates[np.argsort(np.abs(decisiveness))[::-1]][:8]

        print(f"\n  {'measurement':<34}{'this':>11}{'real':>11}{'AI':>11}{'leans':>8}")
        print("  " + "-" * 73)
        for c in order:
            c = int(c)
            v, r, a = vec[c], m_real[c], m_ai[c]
            leans = "real" if abs(v - r) < abs(v - a) else "AI"
            print(f"  {describe_feature(c):<34}{v:>11.4g}{r:>11.4g}{a:>11.4g}{leans:>8}")
        print("  (closest match among the 60 globally most discriminative features -\n"
              "   APPROXIMATE, and can disagree with the verdict; retrain with the\n"
              "   current train_model.py for a SHAP-based explanation that cannot)")

    print(f"\n  Model: held-out AUC {bundle['roc_auc']:.3f}, "
          f"accuracy {bundle['accuracy']:.1%}"
          + (f", trained on {', '.join(bundle['generators'])}"
             if bundle.get("generators") else ""))


if __name__ == "__main__":
    main()
