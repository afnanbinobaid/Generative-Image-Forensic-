"""
classify.py - trains and evaluates AI-vs-real classifiers on the DSP features
produced by feature_extractor.m

Reads dataset_crop.csv (230 features + label) and filenames_crop.txt, then runs
two experiments:

  Experiment A  stratified random split. Answers "can these features separate
                real from AI when the generator has been seen in training?"

  Experiment B  leave-one-generator-out. Trains on one generator and tests on a
                different one, with disjoint real images on each side, so the
                score cannot come from memorising particular photographs.
                Answers "does this generalise to a generator never seen before?"

The gap between A and B is the interesting result, not a failure.

Run with this file's folder as the working directory:
    python classify.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write files, never open a window
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------- configuration
CSV_PATH    = Path("dataset_crop.csv")
NAMES_PATH  = Path("filenames_crop.txt")
OUT_DIR     = Path("results")
RANDOM_SEED = 42
TEST_FRAC   = 0.20

# Feature block layout, matching feature_extractor.m. Ranges are 0-based and
# end-exclusive, so they can index numpy arrays directly.
BLOCKS = [
    ("A spatial",    0,   51),
    ("B wavelet",   51,  198),
    ("C fft",      198,  218),
    ("D residual", 218,  227),
    ("E corr",     227,  230),
]

# Light-mode palette. These charts are for a printed report, so they commit to
# one theme rather than adapting. Categorical slots validated for all-pairs
# colour-blind separation; identity is also carried by line style and by text,
# never by hue alone.
SURFACE    = "#fcfcfb"
INK        = "#0b0b0b"
INK_MUTED  = "#52514e"
GRID       = "#e3e2de"
SERIES     = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ_LIGHT  = "#cde2fb"
SEQ_DARK   = "#184f95"


def style_axes(ax, xlabel="", ylabel="", title=""):
    """Recessive grid and axes, so the data marks carry the chart."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                     loc="left", pad=12)


def feature_labels():
    """'col 95 - B wavelet' style names, 1-based to match MATLAB column numbers."""
    labels = []
    for i in range(230):
        block = next(name for name, lo, hi in BLOCKS if lo <= i < hi)
        labels.append(f"col {i + 1} - {block}")
    return labels


def block_of(index0):
    return next(name for name, lo, hi in BLOCKS if lo <= index0 < hi)


# ------------------------------------------------------------------- load data
def load_data():
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: cannot find {CSV_PATH.resolve()}\n"
                 f"Run this script from the folder that contains it.")

    data = pd.read_csv(CSV_PATH, header=None).to_numpy(dtype=float)
    if data.shape[1] != 231:
        sys.exit(f"ERROR: expected 231 columns, found {data.shape[1]}. "
                 f"Is this the right CSV?")

    X = data[:, :230]
    y = data[:, 230].astype(int)

    # Generator identity comes from the filename, and only for AI rows - a real
    # filename like ILSVRC2012_val_00001277 would otherwise parse to 'val'.
    generator = np.array(["real"] * len(y), dtype=object)
    if NAMES_PATH.exists():
        names = NAMES_PATH.read_text().splitlines()
        if len(names) != len(y):
            print(f"WARNING: {NAMES_PATH} has {len(names)} lines but the CSV has "
                  f"{len(y)} rows. Skipping Experiment B.")
        else:
            for i in np.flatnonzero(y == 1):
                stem = Path(names[i].replace("\\", "/")).stem
                parts = stem.split("_")
                generator[i] = parts[1].lower() if len(parts) >= 2 else "unknown"
    else:
        print(f"WARNING: {NAMES_PATH} not found. Skipping Experiment B.")

    return X, y, generator


# ------------------------------------------------------------------- the models
def build_models():
    """Three classifiers spanning linear, bagged trees, and boosted trees.

    Only the linear model needs scaling, and it gets it inside a pipeline so the
    scaler is fitted on training folds alone - fitting it on all the data first
    would leak test statistics into training.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy":  accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred, zero_division=0),
        "f1":        f1_score(y_te, pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, prob),
        "pred": pred, "prob": prob, "model": model,
    }


def print_table(rows, title):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'ROC-AUC':>10}")
    for name, r in rows.items():
        print(f"{name:<22}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
              f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['roc_auc']:>10.4f}")


# ----------------------------------------------------------------- experiment A
def experiment_a(X, y):
    print("\n" + "=" * 66)
    print("EXPERIMENT A - stratified random split")
    print("=" * 66)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=RANDOM_SEED)
    print(f"train {len(y_tr)} rows ({np.sum(y_tr==0)} real / {np.sum(y_tr==1)} AI)")
    print(f"test  {len(y_te)} rows ({np.sum(y_te==0)} real / {np.sum(y_te==1)} AI)")

    results = {}
    for name, model in build_models().items():
        print(f"  training {name} ...")
        results[name] = evaluate(model, X_tr, y_tr, X_te, y_te)

    print_table(results, "Results on the held-out 20%")
    return results, y_te


# ----------------------------------------------------------------- experiment B
def experiment_b(X, y, generator):
    gens = sorted({g for g in generator[y == 1]})
    if len(gens) < 2:
        print("\nSkipping Experiment B - need at least two generators, found "
              f"{gens}")
        return None

    print("\n" + "=" * 66)
    print("EXPERIMENT B - leave-one-generator-out")
    print("=" * 66)
    print(f"generators found: {', '.join(gens)}")

    real_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(real_idx)
    half = len(real_idx) // 2
    real_tr, real_te = real_idx[:half], real_idx[half:]

    rows = {}
    for held_out in gens:
        train_gen = [g for g in gens if g != held_out]
        tr_ai = np.flatnonzero((y == 1) & np.isin(generator, train_gen))
        te_ai = np.flatnonzero((y == 1) & (generator == held_out))

        tr = np.concatenate([real_tr, tr_ai])
        te = np.concatenate([real_te, te_ai])

        model = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                       random_state=RANDOM_SEED)
        label = f"train {'+'.join(train_gen)} -> test {held_out}"
        print(f"  {label}  ({len(tr)} train / {len(te)} test rows)")
        rows[label] = evaluate(model, X[tr], y[tr], X[te], y[te])

    print_table(rows, "Cross-generator generalisation (Random Forest)")
    return rows


# ---------------------------------------------------------------------- charts
def chart_roc(results, y_te, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    styles = ["-", "--", ":"]

    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, zorder=1)
    ax.text(0.62, 0.55, "random guessing", color=INK_MUTED, fontsize=8.5,
            rotation=38, ha="center", va="center")

    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_te, r["prob"])
        ax.plot(fpr, tpr, color=SERIES[i % 3], linestyle=styles[i % 3],
                linewidth=2, zorder=3,
                label=f"{name}  (AUC {r['roc_auc']:.3f})")

    style_axes(ax, "False positive rate", "True positive rate",
               "Which classifier separates real from AI best?")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_confusion(y_te, pred, name, path):
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", [SEQ_LIGHT, SEQ_DARK])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())

    labels = ["Real", "AI"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for r in range(2):
        for c in range(2):
            frac = cm[r, c] / cm[r].sum()
            ax.text(c, r, f"{cm[r, c]:,}\n{frac:.1%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="#ffffff" if frac > 0.5 else INK)
    # 2px surface gap so the four cells read as separate marks
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.grid(which="major", visible=False)
    style_axes(ax, "Predicted", "Actual", f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_importance(rf, path, top_n=20):
    imp = rf.feature_importances_
    labels = feature_labels()
    order = np.argsort(imp)[::-1][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: the individual features that matter most.
    ax = axes[0]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, imp[order], color=SERIES[0], height=0.68)
    ax.set_yticks(ypos, [labels[i] for i in order], fontsize=8.5)
    style_axes(ax, "Importance", "", f"Top {top_n} features")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Right: total importance per block, which says which DSP method carried
    # the signal. Bars are axis-labelled, so colour encodes nothing here.
    ax = axes[1]
    names = [b[0] for b in BLOCKS]
    totals = [imp[lo:hi].sum() for _, lo, hi in BLOCKS]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, totals, color=SERIES[0], height=0.6)
    ax.set_yticks(ypos, names, fontsize=10)
    for i, v in zip(ypos, totals):
        ax.text(v + max(totals) * 0.015, i, f"{v:.1%}", va="center",
                fontsize=9.5, color=INK)
    ax.set_xlim(0, max(totals) * 1.18)
    style_axes(ax, "Summed importance", "", "Which feature block carried it")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print("\nTop 15 features by Random Forest importance")
    print("-" * 43)
    for i in order[:15]:
        print(f"  {labels[i]:<22} {imp[i]:.4f}")
    print("\nSummed importance per block")
    print("-" * 43)
    for name, total in zip(names, totals):
        print(f"  {name:<22} {total:.1%}")


# ------------------------------------------------------------------------ main
def main():
    X, y, generator = load_data()
    print(f"Loaded {X.shape[0]} images x {X.shape[1]} features")
    print(f"  real : {np.sum(y == 0)}")
    print(f"  AI   : {np.sum(y == 1)}")
    for g in sorted({g for g in generator[y == 1]}):
        print(f"    {g:<14} {np.sum(generator == g)}")

    if not np.all(np.isfinite(X)):
        bad = np.sum(~np.isfinite(X))
        sys.exit(f"ERROR: {bad} non-finite values in the features. "
                 f"Re-check the MATLAB output.")

    OUT_DIR.mkdir(exist_ok=True)

    results, y_te = experiment_a(X, y)
    experiment_b(X, y, generator)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest by ROC-AUC: {best}")

    chart_roc(results, y_te, OUT_DIR / "roc_curves.png")
    chart_confusion(y_te, results[best]["pred"], best,
                    OUT_DIR / "confusion_matrix.png")
    chart_importance(results["Random Forest"]["model"],
                     OUT_DIR / "feature_importance.png")

    print(f"\nCharts written to {OUT_DIR.resolve()}")
    print("  roc_curves.png  confusion_matrix.png  feature_importance.png")


if __name__ == "__main__":
    main()
