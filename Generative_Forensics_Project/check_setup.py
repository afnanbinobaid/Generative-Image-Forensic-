"""
check_setup.py - is this folder actually running the code you think it is?

    python check_setup.py

Version drift has caused four separate wrong numbers in this project: a stale
train_model.py that silently used an ungrouped split, a diagnose_domain.py run
from a folder with the old suffix regex, a classify.py that never used the
shared splitter, and a model.joblib left over from before calibration existed.
Every one of them produced a plausible-looking figure and no error.

This script checks the three things that can silently disagree:

  1. the .py files on disk - do they contain the features they should?
  2. model.joblib - which features did the run that produced it have?
  3. the clock - was model.joblib saved BEFORE the sources were last changed?

That third check is the one that catches "I downloaded the new file but did not
re-run it", which is the most common cause and the least visible.
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# marker -> (file, what having it means)
SOURCE_MARKERS = {
    "split_utils.py": [
        ("qhi|qlo|rweb", "recognises the current augmentation suffixes"),
        ("make_split_3way", "can carve a calibration slice"),
    ],
    "train_model.py": [
        ("describe_split", "reports which split it used"),
        ("make_split_3way", "uses a three-way train/calibrate/test split"),
        ("_calibrate_prefit", "calibrates probabilities"),
        ("background", "saves a SHAP background sample"),
    ],
    "predict_image.py": [
        ("shap_contributions", "explains verdicts with SHAP"),
        ("base_model", "explains the base model under a calibrated wrapper"),
    ],
    "classify.py": [
        ("make_split", "uses the shared grouped splitter"),
    ],
    "diagnose_domain.py": [
        ("make_split", "uses the shared grouped splitter"),
    ],
    "final_test.py": [
        ("check_unseen", "refuses to score training images"),
    ],
}

# key in model.joblib -> what its presence proves about the run
BUNDLE_KEYS = {
    "grouped_split": "the split was grouped by source photograph",
    "background": "a SHAP background sample was saved",
    "base_model": "the uncalibrated model was kept for explanations",
    "calibrated": "calibration was attempted",
}


def check_sources():
    print("1. SOURCE FILES")
    print("-" * 66)
    stale = []
    newest = 0.0
    for fname, markers in SOURCE_MARKERS.items():
        path = HERE / fname
        if not path.exists():
            print(f"  {fname:<22} MISSING")
            stale.append(fname)
            continue
        newest = max(newest, path.stat().st_mtime)
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [(m, why) for m, why in markers
                   if not any(part in text for part in m.split("|"))]
        if missing:
            stale.append(fname)
            print(f"  {fname:<22} OUT OF DATE")
            for m, why in missing:
                print(f"      missing '{m}' - so it cannot: {why}")
        else:
            print(f"  {fname:<22} current")
    return stale, newest


def check_model(newest_source):
    print()
    print("2. model.joblib")
    print("-" * 66)
    path = HERE / "model.joblib"
    if not path.exists():
        print("  MISSING - run  python train_model.py")
        return False, True

    try:
        import joblib
        bundle = joblib.load(path)
    except Exception as exc:
        print(f"  could not be read: {exc}")
        return False, True

    for key, meaning in BUNDLE_KEYS.items():
        present = key in bundle and bundle[key] is not None
        mark = "yes" if present else "NO "
        print(f"  {mark}  {key:<16} {meaning}")

    if bundle.get("calibrated"):
        print(f"       calibration method : {bundle.get('calibration_method', '?')}")
        if bundle.get("ece") is not None:
            print(f"       ECE {bundle.get('ece_uncalibrated', float('nan')):.4f}"
                  f" -> {bundle['ece']:.4f}")
    elif "calibrated" in bundle:
        print("       calibration ran and chose to leave the scores alone")

    saved = path.stat().st_mtime
    print(f"\n  saved: {time.strftime('%Y-%m-%d %H:%M', time.localtime(saved))}")
    stale_model = saved < newest_source
    if stale_model:
        print("  STALE: this model was saved BEFORE the .py files were last")
        print("         changed, so it does not contain their improvements.")
    else:
        print("  newer than every source file, so it reflects the current code")
    return bool(bundle.get("calibrated")), stale_model


def check_env():
    print()
    print("3. ENVIRONMENT")
    print("-" * 66)
    ok = True
    for mod, needed_for in (("numpy", "everything"),
                            ("sklearn", "training and scoring"),
                            ("joblib", "loading the model"),
                            ("pandas", "reading the dataset CSV"),
                            ("shap", "the demo's per-feature explanation")):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {mod:<10} {ver}")
        except ImportError:
            print(f"  {mod:<10} NOT INSTALLED - needed for {needed_for}")
            if mod != "shap":
                ok = False
    return ok


def check_data():
    print()
    print("4. DATA")
    print("-" * 66)
    csv = Path("dataset_crop.csv")
    names = Path("filenames_crop.txt")
    if not csv.exists():
        print("  dataset_crop.csv    MISSING - run feature_extractor in MATLAB")
        return
    n_rows = sum(1 for _ in csv.open())
    print(f"  dataset_crop.csv    {n_rows} rows")
    if not names.exists():
        print("  filenames_crop.txt  MISSING")
        print("      Without it the split CANNOT group by source photograph,")
        print("      and any accuracy on an augmented dataset is inflated.")
        return
    n_names = sum(1 for _ in names.open())
    print(f"  filenames_crop.txt  {n_names} lines")
    if n_names != n_rows:
        print("      MISMATCH with the CSV. The splitter silently falls back to")
        print("      an ungrouped split when these disagree - re-extract.")
    else:
        print("      row counts match, so grouping can engage")


def main():
    print("=" * 66)
    print("SETUP CHECK")
    print("=" * 66)
    stale_sources, newest = check_sources()
    calibrated, stale_model = check_model(newest)
    env_ok = check_env()
    check_data()

    print()
    print("=" * 66)
    print("WHAT TO DO")
    print("=" * 66)
    todo = []
    if stale_sources:
        todo.append(f"Re-download these files: {', '.join(stale_sources)}")
    if not env_ok:
        todo.append("Install the missing packages above")
    if stale_model or not calibrated:
        todo.append("Run:  python train_model.py    "
                    "(model.joblib is only rebuilt by running it)")
    if todo:
        for i, t in enumerate(todo, 1):
            print(f"  {i}. {t}")
    else:
        print("  Nothing - sources, model and environment all agree.")
    print()


if __name__ == "__main__":
    main()
