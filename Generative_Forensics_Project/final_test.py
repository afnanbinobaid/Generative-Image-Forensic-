"""
final_test.py - the in-practice accuracy, on one folder of real and one of AI

score_folder.py answers "how does the detector do on this one folder". This
answers the question that actually matters for a writeup: given a realistic mix
of real photographs and generated images it has never seen, how often is it
right, and what kind of mistakes does it make?

Extract both folders in MATLAB first:

    extract_folder('E:\\test\\real_images', 'final_real.csv')
    extract_folder('E:\\test\\ai_images',   'final_ai.csv')

then:

    python final_test.py final_real.csv final_ai.csv

Reports a confusion matrix, accuracy/precision/recall/F1 with confidence
intervals, ROC-AUC, whether the calibrated probabilities held up out of sample,
and how the verdict would change at a different threshold.

IMPORTANT: the images must be ones the model has never trained on. The script
checks for overlap with filenames_crop.txt and refuses to report a headline
number if it finds any - an accuracy measured on training images is not an
accuracy, and this is the last place that mistake could slip through.
"""

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import joblib

from split_utils import group_key

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
TRAIN_NAMES = Path("filenames_crop.txt")


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion - honest at small n, unlike +-1.96*SE.

    At n=79 and p=0.87 the normal approximation runs past 0.95 and understates
    the downside; Wilson stays inside [0, 1] and is the interval to quote.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_features(path):
    """A feature CSV from extract_folder.m: 230 columns, no label."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: cannot find {path}")
    X = np.loadtxt(path, delimiter=",")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] == 231:                 # tolerate a labelled CSV
        X = X[:, :230]
    if X.shape[1] != 230:
        sys.exit(f"ERROR: {path} has {X.shape[1]} feature columns, expected 230")
    if not np.all(np.isfinite(X)):
        print(f"  WARNING: {path.name} has {np.sum(~np.isfinite(X))} non-finite "
              f"values", file=sys.stderr)
    return X


def load_names(csv_path):
    """The row-aligned filename list extract_folder.m writes next to the CSV."""
    p = Path(csv_path).with_suffix(".filenames.txt")
    if not p.exists():
        return None
    return p.read_text().splitlines()


def check_unseen(real_csv, ai_csv):
    """Refuse to report a headline number measured on training images."""
    if not TRAIN_NAMES.exists():
        print("  NOTE: filenames_crop.txt not found, so overlap with the training\n"
              "        set could not be checked. Make sure these images are unseen.")
        return True

    trained = {group_key(n) for n in TRAIN_NAMES.read_text().splitlines()}
    clean = True
    for csv_path in (real_csv, ai_csv):
        names = load_names(csv_path)
        if names is None:
            print(f"  NOTE: no filename list beside {Path(csv_path).name}, so its\n"
                  f"        overlap with training could not be checked.")
            continue
        overlap = {group_key(n) for n in names} & trained
        if overlap:
            clean = False
            print(f"  CONTAMINATED: {len(overlap)} of {len(names)} images in "
                  f"{Path(csv_path).name} are in the training set.")
            for ex in sorted(overlap)[:3]:
                print(f"      e.g. {ex}")
    return clean


def expected_calibration_error(y, prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(ece)


def roc_auc(y, prob):
    """AUC via the rank identity, so scipy/sklearn are not needed here."""
    order = np.argsort(prob, kind="mergesort")
    ranks = np.empty(len(prob), float)
    sp = prob[order]
    i = 0
    while i < len(sp):                     # average ranks within ties
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    real_csv, ai_csv = sys.argv[1], sys.argv[2]

    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: cannot find {MODEL_PATH}. Run train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    X_real, X_ai = load_features(real_csv), load_features(ai_csv)
    X = np.vstack([X_real, X_ai])
    y = np.concatenate([np.zeros(len(X_real), int), np.ones(len(X_ai), int)])

    print("=" * 70)
    print("FINAL TEST - in-practice accuracy on unseen folders")
    print("=" * 70)
    print(f"real images : {len(X_real):>5}   ({Path(real_csv).name})")
    print(f"AI images   : {len(X_ai):>5}   ({Path(ai_csv).name})")
    print(f"threshold   : {threshold:.4f}"
          + (f"   (calibrated: {bundle.get('calibration_method')})"
             if bundle.get("calibrated") else "   (uncalibrated)"))
    print()
    clean = check_unseen(real_csv, ai_csv)
    print()

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    print("Confusion matrix")
    print(f"    {'':<14}{'called REAL':>13}{'called AI':>12}")
    print(f"    {'is REAL':<14}{tn:>13}{fp:>12}")
    print(f"    {'is AI':<14}{fn:>13}{tp:>12}")
    print()

    n = len(y)
    acc = (tp + tn) / n
    lo, hi = wilson(tp + tn, n)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")

    ra_lo, ra_hi = wilson(tn, len(X_real))
    ai_lo, ai_hi = wilson(tp, len(X_ai))

    print("Headline")
    print(f"    accuracy            {acc:.1%}   (95% CI {lo:.1%} - {hi:.1%})   {tp+tn}/{n}")
    print(f"    accuracy on REAL    {tn/len(X_real):.1%}   (95% CI {ra_lo:.1%} - {ra_hi:.1%})   {tn}/{len(X_real)}")
    print(f"    accuracy on AI      {tp/len(X_ai):.1%}   (95% CI {ai_lo:.1%} - {ai_hi:.1%})   {tp}/{len(X_ai)}")
    print()
    print("Detection quality")
    print(f"    precision           {prec:.3f}   of images called AI, this share really were")
    print(f"    recall              {rec:.3f}   of AI images, this share were caught")
    print(f"    specificity         {spec:.3f}   of real photos, this share were left alone")
    print(f"    F1                  {f1:.3f}")
    print(f"    ROC-AUC             {roc_auc(y, prob):.4f}   threshold-independent ranking")
    print()

    print("Score distribution")
    for lab, name in ((0, "real"), (1, "AI")):
        s = prob[y == lab]
        print(f"    {name:<5} mean {s.mean():.3f}   median {np.median(s):.3f}   "
              f"range {s.min():.3f} - {s.max():.3f}")
    print()

    if bundle.get("calibrated"):
        ece = expected_calibration_error(y, prob)
        print(f"Calibration held out of sample?   ECE {ece:.4f}"
              f"   (training-time {bundle.get('ece', float('nan')):.4f})")
        print("    A score of X should mean about an X chance of being AI.")
        print()

    # Where else the threshold could sit. The saved one maximises balanced
    # accuracy on held-out training data; on a different mix a different cutoff
    # may serve better, and it is worth seeing the trade rather than assuming.
    print("If the threshold moved")
    print(f"    {'threshold':>10}{'accuracy':>10}{'recall':>9}{'specificity':>13}")
    for t in (0.3, 0.4, 0.5, threshold, 0.6, 0.7):
        p = (prob >= t).astype(int)
        a = (p == y).mean()
        r = ((p == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        sp = ((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        mark = "  <- saved" if abs(t - threshold) < 1e-9 else ""
        print(f"    {t:>10.3f}{a:>10.1%}{r:>9.3f}{sp:>13.3f}{mark}")
    print()

    out = Path(real_csv).with_name("final_test_scores.csv")
    rn = load_names(real_csv) or [f"real_{i+1}" for i in range(len(X_real))]
    an = load_names(ai_csv) or [f"ai_{i+1}" for i in range(len(X_ai))]
    allnames = (rn if len(rn) == len(X_real) else [f"real_{i+1}" for i in range(len(X_real))]) + \
               (an if len(an) == len(X_ai) else [f"ai_{i+1}" for i in range(len(X_ai))])
    with out.open("w") as fh:
        fh.write("filename,truth,score,verdict,correct\n")
        for nm, t, s, p in zip(allnames, y, prob, pred):
            fh.write(f"{Path(nm).name},{'AI' if t else 'real'},{s:.6f},"
                     f"{'AI' if p else 'real'},{int(t == p)}\n")
    print(f"Per-image results: {out.resolve()}")

    if not clean:
        print()
        print("!" * 70)
        print("Some test images were also in training. The numbers above are NOT")
        print("a generalisation figure - remove those images and run this again.")
        print("!" * 70)


if __name__ == "__main__":
    main()
