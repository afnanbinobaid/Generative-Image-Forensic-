"""
split_utils.py - one definition of the train/test split

Augmentation makes the split non-trivial. Several rows can come from the same
photograph (the original plus its _qhi, _qlo and _rweb copies), and if some
land in training while others land in test the model can memorise the image
itself and report an accuracy it has not earned.

So rows are grouped by source photograph and whole groups are assigned to one
side. train_model.py and diagnose_domain.py both call this, so the split they
reason about is the same one.
"""

import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# The suffixes make_augmented.m appends. Stripped to recover the source photo.
# qhi/qlo/rweb are the current random-quality names; q85/q60/r75q85 are kept so
# a dataset augmented before that change still groups correctly.
AUG_SUFFIX = re.compile(r"_(qhi|qlo|rweb|soft|q85|q60|r75q85)$", re.IGNORECASE)

TEST_FRAC = 0.20


def group_key(path):
    """The source photograph a row came from, ignoring any augmentation."""
    stem = Path(str(path).replace("\\", "/")).stem
    return AUG_SUFFIX.sub("", stem)


def make_split(y, names=None, seed=42, test_frac=TEST_FRAC):
    """Return (train_idx, test_idx, grouped) for labels y.

    With filenames, groups by source photograph so augmented copies cannot
    straddle the split. Without them, falls back to a stratified random split
    and reports that it did, since the caller needs to know the number may be
    optimistic on an augmented dataset.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))

    if names is not None and len(names) == len(y):
        groups = np.array([group_key(n) for n in names])
        if len(np.unique(groups)) < len(groups):        # augmentation present
            n_splits = max(2, int(round(1 / test_frac)))
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed)
            train_idx, test_idx = next(splitter.split(idx, y, groups))
            return train_idx, test_idx, True

    train_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed)
    return train_idx, test_idx, False


def make_split_3way(y, names=None, seed=42, test_frac=TEST_FRAC,
                    calib_frac=0.15):
    """Return (train_idx, calib_idx, test_idx, grouped) for labels y.

    Probability calibration needs a slice the base model never saw. Fitting the
    calibrator on training data would learn to correct over-confidence the model
    does not exhibit there - it is over-confident precisely on data it has not
    seen - so the mapping would be fitted to the wrong distribution and the
    resulting probabilities would still be wrong, just differently.

    CALIB_FRAC is taken out of the training portion, not the test portion, so
    the test set stays exactly the one make_split() would have produced and the
    accuracy figures remain comparable across runs.

    Every one of the three slices is grouped by source photograph when the
    dataset is augmented. This matters for the calibration slice as much as for
    test: a calibrator fitted on compressed copies of photographs the model
    trained on would see falsely-confident-and-correct predictions and conclude
    no correction was needed.
    """
    y = np.asarray(y)
    train_all, test_idx, grouped = make_split(y, names, seed=seed,
                                              test_frac=test_frac)

    # Split the training portion again, into fit and calibration parts. The
    # sub-split needs the same grouping guarantee, so it is done on the labels
    # and names restricted to the training rows.
    sub_names = None
    if grouped and names is not None:
        sub_names = [names[i] for i in train_all]

    # calib_frac is expressed against the whole dataset; convert it to a
    # fraction of the training portion so the resulting slice is the size asked
    # for rather than a fraction of a fraction.
    sub_frac = min(0.5, max(1e-6, calib_frac / max(len(train_all) / len(y), 1e-9)))

    fit_local, calib_local, _ = make_split(y[train_all], sub_names, seed=seed + 1,
                                           test_frac=sub_frac)
    train_idx = train_all[fit_local]
    calib_idx = train_all[calib_local]
    return train_idx, calib_idx, test_idx, grouped


def describe_split(y, train_idx, test_idx, grouped):
    """A short line saying what the split did, for the caller to print."""
    y = np.asarray(y)
    lines = [
        f"train {len(train_idx)} rows "
        f"({int((y[train_idx] == 0).sum())} real / {int((y[train_idx] == 1).sum())} AI)",
        f"test  {len(test_idx)} rows "
        f"({int((y[test_idx] == 0).sum())} real / {int((y[test_idx] == 1).sum())} AI)",
    ]
    if grouped:
        lines.append("split grouped by source photograph, so augmented copies of "
                     "one image cannot straddle it")
    else:
        lines.append("stratified random split (no augmentation detected)")
    return "\n".join("  " + l for l in lines)
