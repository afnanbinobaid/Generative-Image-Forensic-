"""
diagnose_domain.py - why are real photographs being flagged AI?

Three diagnostics, matching the plan:

  --baseline            D1: does the model still work on the images it was
                        trained for? Re-creates the exact held-out split from
                        dataset_crop.csv and reports accuracy on the real images
                        in it. No re-extraction needed.

  --compare A.csv B.csv D4: which measurements shift, and in which direction,
                        for each feature CSV produced by extract_folder.m
                        D5: where the scores land, as a chart

Run --baseline first. If it comes back low, the problem is not the test images
and the rest of this does not apply.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from feature_names import describe_feature, block_of, BLOCKS

CSV_PATH    = Path("dataset_crop.csv")
MODEL_PATH  = Path("model.joblib")
OUT_DIR     = Path("results")
RANDOM_SEED = 42        # must match train_model.py for the split to line up

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
REAL_COLOUR, AI_COLOUR, OTHER_COLOUR = "#2a78d6", "#eb6834", "#1baf7a"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: cannot find {MODEL_PATH.resolve()}\n"
                 f"Run  python train_model.py  first.")
    return joblib.load(MODEL_PATH)


def load_training():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: cannot find {CSV_PATH.resolve()}")
    data = pd.read_csv(CSV_PATH, header=None).to_numpy(dtype=float)
    if data.shape[1] != 231:
        sys.exit(f"ERROR: expected 231 columns, found {data.shape[1]}")
    return data[:, :230], data[:, 230].astype(int)


def load_features(path):
    """A feature CSV from extract_folder.m: 230 columns, no label."""
    X = np.loadtxt(path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] == 231:            # tolerate a labelled CSV too
        X = X[:, :230]
    if X.shape[1] != 230:
        sys.exit(f"ERROR: {path} has {X.shape[1]} feature columns, expected 230")
    if not np.all(np.isfinite(X)):
        print(f"  WARNING: {path} contains {np.sum(~np.isfinite(X))} non-finite "
              f"values", file=sys.stderr)
    return X


# ------------------------------------------------------------------------ D1
def baseline():
    """Accuracy on the held-out slice of the data the model was trained for."""
    bundle = load_model()
    X, y = load_training()

    # Same call as train_model.py, so this reproduces that exact split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)

    prob = bundle["model"].predict_proba(X_te)[:, 1]
    threshold = float(bundle["threshold"])
    pred = (prob >= threshold).astype(int)

    is_real = y_te == 0
    is_ai   = y_te == 1

    print("=" * 62)
    print("D1  BASELINE - the images the model was trained for")
    print("=" * 62)
    print(f"held-out images     : {len(y_te)}  "
          f"({is_real.sum()} real / {is_ai.sum()} AI)")
    print(f"threshold           : {threshold:.4f}")
    print(f"overall accuracy    : {accuracy_score(y_te, pred):.1%}")
    print(f"accuracy on REAL    : {accuracy_score(y_te[is_real], pred[is_real]):.1%}"
          f"   <- compare this against your web photos")
    print(f"accuracy on AI      : {accuracy_score(y_te[is_ai], pred[is_ai]):.1%}")
    print(f"\nmean score, real    : {prob[is_real].mean():.3f}")
    print(f"mean score, AI      : {prob[is_ai].mean():.3f}")

    real_acc = accuracy_score(y_te[is_real], pred[is_real])
    print()
    if real_acc >= 0.90:
        print("The model handles its own kind of real photographs correctly, so")
        print("the pipeline and model are healthy. Any failure on web photos is")
        print("about those images differing from the training set - continue to D2.")
    else:
        print("STOP: the model is ALSO failing on the real photographs it was")
        print("trained for. This is not about web images. Investigate the model")
        print("and threshold before going further.")
    return prob, y_te


# ------------------------------------------------------------------- D4 + D5
def compare(paths):
    bundle = load_model()
    X, y = load_training()
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    train_real = X[y == 0]
    mean_real  = train_real.mean(axis=0)
    mean_ai    = X[y == 1].mean(axis=0)
    pooled     = np.maximum(bundle.get(
        "pooled_std", np.sqrt((train_real.var(axis=0) + X[y == 1].var(axis=0)) / 2)),
        1e-12)

    # direction that means "more AI-like" for each feature
    toward_ai = np.sign(mean_ai - mean_real)

    sets = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  skipping {p} (not found)", file=sys.stderr)
            continue
        sets[p.stem] = load_features(p)

    if not sets:
        sys.exit("No usable feature CSVs given.")

    print("\n" + "=" * 78)
    print("D4  WHICH MEASUREMENTS SHIFT, AND TOWARD WHICH CLASS")
    print("=" * 78)
    print("Shift is in pooled standard deviations away from the average training")
    print("real photograph. Positive = moved toward the AI class.\n")

    for name, Xv in sets.items():
        prob = model.predict_proba(Xv)[:, 1]
        flagged = (prob >= threshold).mean()

        shift = (Xv.mean(axis=0) - mean_real) / pooled
        signed = shift * toward_ai            # + means "toward AI"
        order = np.argsort(signed)[::-1][:6]

        print(f"--- {name}   ({len(Xv)} images, {flagged:.0%} flagged AI, "
              f"mean score {prob.mean():.3f})")
        for c in order:
            c = int(c)
            print(f"      {signed[c]:+6.2f} sd  {block_of(c):<11} "
                  f"{describe_feature(c)}")

        # which block moved most overall
        per_block = [(nm, float(np.mean(signed[lo:hi]))) for nm, lo, hi in BLOCKS]
        per_block.sort(key=lambda t: t[1], reverse=True)
        summary = "  ".join(f"{nm} {v:+.2f}" for nm, v in per_block)
        print(f"      by block: {summary}\n")

    chart_scores(sets, model, threshold, X, y)


def chart_scores(sets, model, threshold, X, y):
    """D5 - where each group's scores land, against the threshold."""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "domain_scores.png"

    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    base = model.predict_proba(X_te)[:, 1]

    n = len(sets) + 1
    fig, axes = plt.subplots(n, 1, figsize=(8.2, 2.1 * n), facecolor=SURFACE,
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    bins = np.linspace(0, 1, 41)

    def panel(ax, title, groups):
        for values, colour, label in groups:
            if len(values) == 0:
                continue
            ax.hist(values, bins=bins, color=colour, alpha=0.5,
                    histtype="stepfilled", label=label)
            ax.hist(values, bins=bins, color=colour, linewidth=2, histtype="step")
        ax.axvline(threshold, color=INK, linewidth=2)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold",
                     loc="left", pad=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
        for t in leg.get_texts():
            t.set_color(INK)
        # Rotated alongside the line and at mid height, so it clears the legend
        # however the histogram happens to be shaped.
        ax.text(threshold, ax.get_ylim()[1] * 0.5, f"  threshold {threshold:.2f}",
                color=INK, fontsize=8.5, fontweight="bold", rotation=90,
                ha="left", va="center")

    panel(axes[0], "Training distribution (held-out)",
          [(base[y_te == 0], REAL_COLOUR, "real"),
           (base[y_te == 1], AI_COLOUR, "AI")])

    for ax, (name, Xv) in zip(axes[1:], sets.items()):
        prob = model.predict_proba(Xv)[:, 1]
        panel(ax, f"{name}  (all of these are real photographs)",
              [(prob, OTHER_COLOUR, name)])

    axes[-1].set_xlabel("Model score   (0 = real, 1 = AI)", color=INK_MUTED,
                        fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"D5  score distributions written to {path.resolve()}")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)

    if args[0] == "--baseline":
        baseline()
    elif args[0] == "--compare":
        if len(args) < 2:
            sys.exit("usage: python diagnose_domain.py --compare a.csv [b.csv ...]")
        compare(args[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
