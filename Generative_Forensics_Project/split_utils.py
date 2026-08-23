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
AUG_SUFFIX = re.compile(r"_(qhi|qlo|rweb|q85|q60|r75q85)$", re.IGNORECASE)

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
