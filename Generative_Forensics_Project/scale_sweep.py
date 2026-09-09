"""
scale_sweep.py - does the verdict follow the picture, or the size of the file?

    matlab -batch "scale_sweep('Dataset/Real_Images', 40)"
    python scale_sweep.py scale_sweep.csv
    python scale_sweep.py scale_sweep.csv --label real

scale_sweep.m measures the SAME photograph at several source resolutions. This
scores every one of those rows and asks the only question that matters about
them: does the score stay put?

Resizing a photograph does not change whether it was generated. So a detector
reading generation must return nearly the same score at every rung, and one
whose score slides with resolution is reading resolution - a property of the
file, not of the picture. In this dataset the two are confounded (real images
were 450-500px, AI images 512-1024), so reading resolution scores well on the
test set and fails on every image a person actually uploads.

What it reports, per image and overall:

    swing        highest score minus lowest across that image's rungs
    trend        Spearman correlation between rung and score, -1..+1
    flip         the resolution band where the verdict changes, if it does

PASS needs a median swing under 0.15 with under 10% of images flipping. That
is not a statistical convention, it is what "the same picture gets the same
answer" means when the threshold sits in the middle of the range.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

MAX_SWING = 0.15        # median score range across one image's rungs
MAX_FLIP  = 0.10        # share of images whose verdict changes with resolution

N_META = 4              # image index, rung, native short side, preScale


def load_rows(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if data.shape[1] != N_META + 230:
        sys.exit(f"ERROR: expected {N_META + 230} columns in {csv_path}, "
                 f"found {data.shape[1]}.\n"
                 f"Re-run scale_sweep.m - it writes 4 metadata columns then "
                 f"the 230 features.")
    return data


def load_names(csv_path):
    """The row-aligned filename list, if scale_sweep.m managed to write it."""
    list_path = Path(str(csv_path) + ".images.txt")
    names = {}
    if list_path.exists():
        for line in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0].strip().isdigit():
                names[int(parts[0])] = parts[1].strip()
    return names


def spearman(a, b):
    """Rank correlation, without pulling in scipy for one number."""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def flip_band(rungs, verdicts):
    """The two adjacent rungs the verdict changes between, or None."""
    order = np.argsort(rungs)
    r, v = np.asarray(rungs)[order], np.asarray(verdicts)[order]
    for i in range(1, len(r)):
        if v[i] != v[i - 1]:
            return int(r[i - 1]), int(r[i])
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    csv_path = Path(sys.argv[1])
    truth = None
    if "--label" in sys.argv:
        truth = sys.argv[sys.argv.index("--label") + 1].lower()
        if truth not in ("real", "ai"):
            sys.exit("--label must be 'real' or 'ai'")

    if not csv_path.exists():
        sys.exit(f"ERROR: cannot find {csv_path}")
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: cannot find {MODEL_PATH}. Run train_model.py first.")

    data = load_rows(csv_path)
    names = load_names(csv_path)

    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], float(bundle["threshold"])

    scores = model.predict_proba(data[:, N_META:])[:, 1]
    idx, rungs = data[:, 0].astype(int), data[:, 1].astype(int)

    all_rungs = sorted(set(rungs.tolist()))

    print()
    print("=" * 78)
    print("  SCALE SWEEP - the same photograph, measured at several resolutions")
    print("=" * 78)
    print(f"  rows {len(scores)}   images {len(set(idx.tolist()))}   "
          f"threshold {threshold:.4f}")
    print(f"  a score above {threshold:.4f} is a verdict of AI-GENERATED")
    print()

    head = "  image                             " + "".join(f"{r:>7}" for r in all_rungs)
    print(head)
    print("  " + "-" * (len(head) - 2))

    swings, trends, flips, per_image = [], [], 0, []
    for i in sorted(set(idx.tolist())):
        rows = idx == i
        r, s = rungs[rows], scores[rows]
        order = np.argsort(r)
        r, s = r[order], s[order]

        swing = float(s.max() - s.min())
        trend = spearman(r, s)
        band = flip_band(r, s >= threshold)

        swings.append(swing)
        trends.append(trend)
        flips += band is not None
        per_image.append((i, r, s, swing, trend, band))

        label = names.get(i, f"#{i}")
        if len(label) > 32:
            label = label[:29] + "..."
        cells = {int(rr): ss for rr, ss in zip(r, s)}
        line = f"  {label:<34}"
        for rung in all_rungs:
            v = cells.get(rung)
            line += "      -" if v is None else f"{v:>7.3f}"
        print(line + f"   swing {swing:.3f}")

    swings = np.asarray(swings)
    trends = np.asarray(trends)
    med_swing = float(np.median(swings))
    flip_rate = flips / len(swings)

    print()
    print("  " + "-" * 76)
    print(f"  median swing across an image's rungs : {med_swing:.3f}   "
          f"(want < {MAX_SWING:.2f})")
    print(f"  worst swing                          : {swings.max():.3f}")
    print(f"  images whose verdict flips           : {flips}/{len(swings)}"
          f"  ({flip_rate:.0%})   (want < {MAX_FLIP:.0%})")
    print(f"  median rung-vs-score correlation     : {np.median(trends):+.2f}   "
          f"(want ~0.00)")

    # The direction is worth naming: it says which way the shortcut points, and
    # a reader who has seen big images called AI will recognise it immediately.
    med_trend = float(np.median(trends))
    if abs(med_trend) > 0.5:
        way = ("the bigger the upload, the more AI it looks"
               if med_trend > 0 else
               "the smaller the upload, the more AI it looks")
        print(f"  direction                            : {way}")

    if truth is not None:
        want_ai = truth == "ai"
        per_rung = []
        for rung in all_rungs:
            sel = rungs == rung
            acc = float(((scores[sel] >= threshold) == want_ai).mean())
            per_rung.append((rung, acc, int(sel.sum())))
        print()
        print(f"  accuracy at each resolution, all of these being {truth.upper()}:")
        for rung, acc, n in per_rung:
            bar = "#" * int(round(acc * 40))
            print(f"    {rung:>5}px  {acc:6.1%}  ({n:>3} images)  {bar}")
        spread = max(a for _, a, _ in per_rung) - min(a for _, a, _ in per_rung)
        print(f"    spread across resolutions: {spread:.1%}")
        print("    A detector reading generation has no spread here. Whatever")
        print("    spread there is, is the part of the accuracy that is really")
        print("    a measurement of image size.")

    passed = med_swing < MAX_SWING and flip_rate < MAX_FLIP
    print()
    print("=" * 78)
    if passed:
        print("  PASS - the verdict is a property of the picture, not of its size.")
    else:
        print("  FAIL - the verdict follows the resolution of the upload.")
        print()
        print("  The same photograph is getting different answers depending only on")
        print("  how many pixels it was stored in, so no score from this model is")
        print("  evidence about origin until it is fixed. What to check, in order:")
        print("    1. normalize_defaults.m - scaleMode must be 'shortside'. In")
        print("       'native' mode a big upload is measured at a magnification no")
        print("       training image was ever seen at.")
        print("    2. model.joblib must have been trained AFTER the dataset was")
        print("       re-normalised in that mode. A model trained on the old")
        print("       treatment still expects the old input.")
        print("    3. make_augmented's scale ladder must have been applied to BOTH")
        print("       classes, so detail level carries no label information.")
        print("    4. If it still fails, the training set's native resolutions are")
        print("       separable (real 450-500px, AI 512-1024px) and no preprocessing")
        print("       fixes that. The dataset needs matched resolutions.")
    print("=" * 78)
    print()

    plot(per_image, threshold, all_rungs, csv_path)
    return 0 if passed else 1


def plot(per_image, threshold, all_rungs, csv_path):
    """One line per photograph. A flat line is a detector; a slope is a ruler."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - no chart written)")
        return

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(csv_path).stem + "_scale.png")

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.axhspan(threshold, 1, color="#eb6834", alpha=0.07)
    ax.axhspan(0, threshold, color="#2a78d6", alpha=0.07)
    ax.axhline(threshold, color="#0b0b0b", lw=1.2)
    ax.text(all_rungs[-1], threshold + 0.015, "  threshold", fontsize=8,
            color="#52514e", ha="right")

    for _, r, s, _, _, _ in per_image:
        ax.plot(r, s, marker="o", ms=3, lw=1.1, alpha=0.55, color="#1baf7a")

    ax.set_xscale("log")
    ax.set_xticks(all_rungs)
    ax.set_xticklabels([str(r) for r in all_rungs])
    ax.minorticks_off()
    ax.set_xlabel("source resolution, short side (px) - the SAME photograph at each")
    ax.set_ylabel("score  (1 = AI-generated)")
    ax.set_ylim(0, 1)
    ax.set_title("Every line should be flat: resizing a photograph does not change "
                 "what it is", fontsize=10, color="#0b0b0b")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e3e2de")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  chart -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
