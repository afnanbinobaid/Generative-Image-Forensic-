"""
audit_encoding.py - can the two classes be told apart WITHOUT looking at pixels?

    python audit_encoding.py
    python audit_encoding.py path/to/real_folder path/to/ai_folder

audit_folders.m compares what the images look like - dimensions, bytes per
pixel, how many fall below the crop window. This asks a narrower and more
uncomfortable question: how much of the label is recoverable from the file
container alone, before a single DSP feature is computed.

The distinction matters because a JPEG carries its encoder's signature. The
quantisation tables, the chroma subsampling mode and the extension a tool
chose to write are all properties of the SOFTWARE THAT SAVED THE FILE, not of
whether the picture is a photograph. If the two classes came from different
sources - one folder written by ImageNet's pipeline, the other by whatever
GenImage used - then that signature is perfectly correlated with the label,
and the quantisation tables shape the DCT coefficient statistics that Blocks
B, C and D go on to measure. A model can score well by reading the encoder
and never look at the generation artefact at all.

The last section is the decisive one. It fits a classifier on metadata ONLY -
file size, dimensions, extension, quantisation table, subsampling - with no
access to pixels whatsoever, on a split grouped by source photograph. A model
that cannot see the image should be at chance. Whatever it scores above
chance is the size of the shortcut available in the dataset.

    AUC ~ 0.50   no container shortcut; in-distribution scores are clean
    AUC ~ 0.65   a shortcut exists and is probably inflating the headline
    AUC > 0.80   the dataset is separable without pixels; the DSP result
                 cannot be interpreted until this is fixed

Reports the native files and the augmented copies separately, because
make_augmented.m re-encodes through MATLAB's own encoder for both classes -
which homogenises the copies while leaving each original's native signature
intact.
"""

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: needs pillow.  pip install pillow")

AUG_SUFFIXES = ("_qhi", "_qlo", "_rweb", "_soft", "_q85", "_q60", "_r75q85")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_REAL = Path("Dataset") / "Real_Images"
DEFAULT_AI = Path("Dataset") / "AI_Images"


def is_augmented(stem):
    low = stem.lower()
    return any(low.endswith(s) for s in AUG_SUFFIXES)


def source_stem(stem):
    """The source photograph a file belongs to, ignoring any augmentation."""
    low = stem.lower()
    for suffix in AUG_SUFFIXES:
        if low.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def probe(path):
    """Everything about the file that does not require looking at the pixels."""
    rec = {
        "path": path,
        "stem": path.stem,
        "source": source_stem(path.stem),
        "ext": path.suffix.lower(),
        "augmented": is_augmented(path.stem),
        "bytes": path.stat().st_size,
    }
    try:
        with Image.open(path) as im:
            rec["width"], rec["height"] = im.size
            rec["format"] = im.format or "?"
            rec["mode"] = im.mode
            # Quantisation tables are the encoder's fingerprint. Two files from
            # the same tool at the same quality share them exactly.
            q = getattr(im, "quantization", None)
            if q:
                flat = [v for table in sorted(q) for v in q[table]]
                rec["qhash"] = hashlib.md5(
                    ",".join(map(str, flat)).encode()).hexdigest()[:8]
                rec["q_dc"] = flat[0] if flat else np.nan
                rec["q_mean"] = float(np.mean(flat)) if flat else np.nan
            else:
                rec["qhash"] = "none"
                rec["q_dc"] = np.nan
                rec["q_mean"] = np.nan
            sub = im.info.get("subsampling", None) if hasattr(im, "info") else None
            try:
                from PIL import JpegImagePlugin
                if rec["format"] == "JPEG":
                    sub = JpegImagePlugin.get_sampling(im)
            except Exception:
                pass
            rec["subsampling"] = str(sub)
    except Exception as exc:
        rec["error"] = str(exc)
        return rec

    px = rec["width"] * rec["height"]
    rec["bpp"] = rec["bytes"] / px if px else np.nan
    return rec


def scan(folder, label):
    folder = Path(folder)
    if not folder.is_dir():
        sys.exit(f"ERROR: not a folder: {folder.resolve()}")
    rows = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            rec = probe(p)
            rec["label"] = label
            rows.append(rec)
    if not rows:
        sys.exit(f"ERROR: no images in {folder.resolve()}")
    return rows


def pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "    - "


def crosstab(rows, key, title, note=None):
    """Distribution of `key` within each class, side by side."""
    classes = ("real", "ai")
    counts = {c: Counter(r.get(key, "?") for r in rows if r["label"] == c)
              for c in classes}
    totals = {c: sum(counts[c].values()) for c in classes}
    values = sorted(set(counts["real"]) | set(counts["ai"]),
                    key=lambda v: -(counts["real"][v] + counts["ai"][v]))

    print(f"\n  {title}")
    if note:
        print(f"  {note}")
    print(f"    {'value':<22}{'real':>14}{'AI':>14}   verdict")
    print("    " + "-" * 62)
    exclusive = 0
    for v in values[:12]:
        r, a = counts["real"][v], counts["ai"][v]
        mark = ""
        if r and not a:
            mark = "real only"
            exclusive += r
        elif a and not r:
            mark = "AI only"
            exclusive += a
        print(f"    {str(v)[:22]:<22}"
              f"{r:>7} {pct(r, totals['real'])}"
              f"{a:>7} {pct(a, totals['ai'])}   {mark}")
    if len(values) > 12:
        print(f"    ... and {len(values) - 12} more values")

    n = totals["real"] + totals["ai"]
    if exclusive:
        print(f"    -> {exclusive} of {n} files ({100.0*exclusive/n:.1f}%) carry a value "
              f"that occurs in ONE class only")
    return exclusive, n


def best_single_threshold(a, b):
    """Best accuracy any single cut on this one number can reach.

    A median ratio can look harmless while the two ranges do not overlap at
    all - 450-500 against 512-1024 is a median gap of 1.02x and a perfect
    separator. Comparing medians alone misses exactly the confound that
    matters most, so the separability is measured directly instead.
    """
    vals = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = np.argsort(vals, kind="mergesort")
    vals, labels = vals[order], labels[order]
    n = len(vals)

    # Always-guess-the-majority is the floor; a column that carries no
    # information must score exactly this and not a point more.
    baseline = float(max(labels.mean(), 1.0 - labels.mean()))

    # Only a cut BETWEEN two distinct values is a real threshold. Sweeping
    # every index instead would let ties be split in whatever order the sort
    # happened to leave them, and a constant column would score 100%.
    boundaries = np.nonzero(np.diff(vals))[0]
    if not len(boundaries):
        return baseline

    ones_below = np.cumsum(labels)
    zeros_below = np.cumsum(1 - labels)
    total_ones = labels.sum()

    # predict "real" at or below the cut, "AI" above it
    correct = zeros_below[boundaries] + (total_ones - ones_below[boundaries])
    acc = correct / n
    return float(max(baseline, acc.max(), (1.0 - acc).max()))


def summarise_numeric(rows, key, title):
    classes = ("real", "ai")
    print(f"\n  {title}")
    print(f"    {'class':<10}{'median':>12}{'mean':>12}{'min':>12}{'max':>12}")
    print("    " + "-" * 58)
    med, arrays = {}, {}
    for c in classes:
        vals = np.array([r[key] for r in rows
                         if r["label"] == c and not np.isnan(r.get(key, np.nan))])
        if not len(vals):
            continue
        med[c] = float(np.median(vals))
        arrays[c] = vals
        print(f"    {c:<10}{np.median(vals):>12.4g}{vals.mean():>12.4g}"
              f"{vals.min():>12.4g}{vals.max():>12.4g}")

    ratio = 1.0
    if len(med) == 2 and min(med.values()) > 0:
        ratio = max(med.values()) / min(med.values())
        bigger = max(med, key=med.get)
        print(f"    -> median gap {ratio:.2f}x  ({bigger} larger)")

    if len(arrays) == 2:
        a, b = arrays["real"], arrays["ai"]
        sep = best_single_threshold(a, b)
        overlap_lo, overlap_hi = max(a.min(), b.min()), min(a.max(), b.max())
        disjoint = overlap_lo > overlap_hi
        print(f"    -> this number ALONE classifies at {sep*100:.1f}% "
              f"with one threshold", end="")
        if disjoint:
            print("   <- RANGES DO NOT OVERLAP")
        elif sep >= 0.90:
            print("   <- near-perfect separator")
        else:
            print()
        return ratio, sep
    return ratio, 0.5


def trivial_cue_test(rows):
    """Fit on container metadata only. Pixels are never read."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        print("\n  (scikit-learn not available - skipping the trivial-cue test)")
        return None

    usable = [r for r in rows if "error" not in r]
    qhashes = sorted({r["qhash"] for r in usable})
    subs = sorted({r["subsampling"] for r in usable})
    exts = sorted({r["ext"] for r in usable})

    X, y, groups = [], [], []
    for r in usable:
        X.append([
            r["bytes"], r["bpp"], r["width"], r["height"],
            r["width"] * r["height"],
            r["width"] / max(r["height"], 1),
            r["q_dc"] if not np.isnan(r["q_dc"]) else -1,
            r["q_mean"] if not np.isnan(r["q_mean"]) else -1,
            qhashes.index(r["qhash"]),
            subs.index(r["subsampling"]),
            exts.index(r["ext"]),
        ])
        y.append(1 if r["label"] == "ai" else 0)
        groups.append(r["source"])
    X = np.array(X, dtype=float)
    y = np.array(y)
    groups = np.array(groups)

    if len(np.unique(y)) < 2:
        print("\n  (only one class present - skipping)")
        return None

    aucs, accs = [], []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in splitter.split(X, y, groups):
        model = HistGradientBoostingClassifier(random_state=42).fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        accs.append(accuracy_score(y[te], (prob >= 0.5).astype(int)))

    auc, auc_sd = float(np.mean(aucs)), float(np.std(aucs))
    acc = float(np.mean(accs))

    print("\n  Metadata-only classifier, 5-fold grouped CV, NO pixels read")
    print(f"    ROC-AUC   {auc:.4f}  (sd {auc_sd:.4f})")
    print(f"    accuracy  {acc:.4f}")
    print("    features  file size, bytes/pixel, dimensions, aspect,")
    print("              quantisation table, chroma subsampling, extension")

    # At 40,000 rows a small AUC is still many standard errors from chance, so
    # "close to 0.50" is not the same claim as "indistinguishable from 0.50".
    # Reporting the distance in sd keeps a residual visible instead of letting
    # a loose threshold wave it through.
    if auc_sd > 0:
        sigma = (auc - 0.5) / auc_sd
        if sigma >= 3:
            print(f"    -> {sigma:.1f} sd above chance: small, but real, not noise")
        else:
            print(f"    -> {sigma:.1f} sd above chance: consistent with no shortcut")
    return auc, auc_sd


def verdict(auc, ext_exclusive_frac, q_exclusive_frac, bpp_ratio, res_ratio,
            res_sep=0.5, bpp_sep=0.5, auc_sd=0.0):
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    problems = []
    if auc is not None and auc >= 0.80:
        problems.append(
            f"A model that never sees a pixel separates the classes at AUC {auc:.3f}.\n"
            "     The DSP accuracy cannot be interpreted until this is closed - any\n"
            "     part of it could be the container rather than the content.")
    elif auc is not None and auc >= 0.65:
        problems.append(
            f"Metadata alone reaches AUC {auc:.3f}. A real shortcut exists and is\n"
            "     probably inflating the in-distribution number.")

    if q_exclusive_frac >= 0.5:
        problems.append(
            f"{q_exclusive_frac*100:.0f}% of files carry a quantisation table seen in only\n"
            "     one class. That is an encoder fingerprint, and it shapes exactly the\n"
            "     DCT statistics Blocks B, C and D go on to measure.")

    if ext_exclusive_frac >= 0.5:
        problems.append(
            f"{ext_exclusive_frac*100:.0f}% of files have a class-exclusive extension.\n"
            "     The extension never reaches the features, but it is a marker that the\n"
            "     classes came from different sources - and the encoder did too.")

    if bpp_ratio >= 1.35:
        problems.append(
            f"Bytes-per-pixel gap of {bpp_ratio:.2f}x exceeds the project's own 1.35x limit.")

    if res_ratio >= 1.25:
        problems.append(
            f"Resolution gap of {res_ratio:.2f}x exceeds the project's own 1.25x limit.")

    for name, sep, ratio in (("Native width", res_sep, res_ratio),
                             ("Bytes per pixel", bpp_sep, bpp_ratio)):
        if sep >= 0.90 and ratio < 1.25:
            problems.append(
                f"{name} alone classifies at {sep*100:.1f}% with a single threshold,\n"
                f"     while its median gap is only {ratio:.2f}x. A median comparison -\n"
                "     including audit_folders.m's - passes this cleanly and should not.")

    if not problems:
        sigma = (auc - 0.5) / auc_sd if (auc is not None and auc_sd) else 0.0
        if sigma >= 3:
            print("\n  No container SHORTCUT - nothing here decides the label on its own.")
            print(f"  But the metadata AUC of {auc:.4f} sits {sigma:.1f} sd above chance, which at")
            print("  this sample size is a real residual rather than noise. Every categorical")
            print("  property is now identical across classes, so the only thing left varying")
            print("  is file size at fixed encoder settings - a proxy for how compressible the")
            print("  picture is.")
            print("\n  That is genuinely ambiguous and should be reported, not rounded away:")
            print("    - it may be the signal itself, if generated images really are smoother")
            print("    - it may be leftover history, if one class arrived already compressed")
            print("      harder; resampling removes the DCT grid signature but not the")
            print("      bandwidth that earlier compression already took away")
            print("  Quote this number as the control's floor. Do not quote 0.50.")
            return 0
        print("\n  No container-level shortcut found. The classes are not separable")
        print("  from metadata, so an in-distribution score is measuring content.")
        return 0

    for i, p in enumerate(problems, 1):
        print(f"\n  {i}. {p}")

    print("\n  The fix is to make the container carry no label information, the same")
    print("  way compression augmentation was made to: re-encode EVERY image in both")
    print("  classes through one identical encoder at one quality before extraction,")
    print("  then retrain. If in-distribution accuracy falls, the gap was the shortcut.")
    return len(problems)


def main():
    args = sys.argv[1:]
    real_dir = Path(args[0]) if len(args) > 0 else DEFAULT_REAL
    ai_dir = Path(args[1]) if len(args) > 1 else DEFAULT_AI

    print("=" * 70)
    print("  ENCODING AUDIT - is the label recoverable without pixels?")
    print("=" * 70)
    print(f"  real : {Path(real_dir).resolve()}")
    print(f"  AI   : {Path(ai_dir).resolve()}")

    rows = scan(real_dir, "real") + scan(ai_dir, "ai")
    bad = [r for r in rows if "error" in r]
    if bad:
        print(f"\n  {len(bad)} file(s) could not be read; excluded.")
        rows = [r for r in rows if "error" not in r]

    native = [r for r in rows if not r["augmented"]]
    augmented = [r for r in rows if r["augmented"]]

    print(f"\n  files      {len(rows):>6}   "
          f"real {sum(r['label']=='real' for r in rows)}   "
          f"AI {sum(r['label']=='ai' for r in rows)}")
    print(f"  native     {len(native):>6}   (originals, each with its own encoder)")
    print(f"  augmented  {len(augmented):>6}   (re-encoded by MATLAB, both classes alike)")

    # Stem collisions - make_augmented.m derives its output name from the stem
    # alone, so two originals sharing a stem across extensions collide, and the
    # resume guard then skips the second silently.
    by_source = defaultdict(set)
    for r in native:
        by_source[r["source"].lower()].add(r["ext"])
    collisions = {k: v for k, v in by_source.items() if len(v) > 1}
    if collisions:
        print(f"\n  WARNING  {len(collisions)} stem(s) exist with more than one extension.")
        print("           make_augmented.m names its output from the stem alone, so the")
        print("           second file's variants collide with the first's and are skipped.")
        for k in list(collisions)[:5]:
            print(f"             {k}  ->  {sorted(collisions[k])}")

    print("\n" + "-" * 70)
    print("  NATIVE FILES  (what the originals carry)")
    print("-" * 70)

    ext_excl, ext_n = crosstab(
        native, "ext", "File extension by class",
        "the extension never reaches the features - it is a provenance marker")
    crosstab(native, "format", "Container format by class")
    q_excl, q_n = crosstab(
        native, "qhash", "JPEG quantisation table by class",
        "the encoder's fingerprint; it shapes the DCT statistics the features read")
    crosstab(native, "subsampling", "Chroma subsampling by class")

    res_ratio, res_sep = summarise_numeric(native, "width", "Width (px)")
    bpp_ratio, bpp_sep = summarise_numeric(native, "bpp", "Bytes per pixel")

    if augmented:
        print("\n" + "-" * 70)
        print("  AUGMENTED COPIES  (should look identical across classes)")
        print("-" * 70)
        crosstab(augmented, "qhash", "JPEG quantisation table by class",
                 "both classes went through MATLAB's encoder, so these should overlap")

    print("\n" + "-" * 70)
    print("  THE DECISIVE TEST")
    print("-" * 70)
    print("\n  Native files only:")
    auc_native, sd_native = trivial_cue_test(native)
    if augmented:
        print("\n  All files, native and augmented together (what training actually saw):")
        trivial_cue_test(rows)

    verdict(auc_native,
            ext_excl / max(ext_n, 1),
            q_excl / max(q_n, 1),
            bpp_ratio, res_ratio, res_sep, bpp_sep, sd_native)


if __name__ == "__main__":
    main()
