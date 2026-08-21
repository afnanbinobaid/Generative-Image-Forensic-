"""
feature_names.py - what each feature column measures

The CSV that feature_extractor.m writes has no header, so this is the single
place that maps a column number back to the DSP measurement behind it. Used by
the trainer and by the live demo to explain a verdict.
"""

BLOCKS = [
    ("A spatial",    0,   51),
    ("B wavelet",   51,  198),
    ("C fft",      198,  218),
    ("D residual", 218,  227),
    ("E corr",     227,  230),
]

_CHANNELS   = ("red", "green", "blue")
_BASE_STATS = ("mean", "std", "variance", "energy", "entropy", "skewness", "kurtosis")
_GLCM_STATS = ("GLCM contrast", "GLCM correlation", "GLCM energy", "GLCM homogeneity")
_EDGE_STATS = ("mean Gx", "var Gx", "mean Gy", "var Gy", "mean |G|", "var |G|")
_SUBBANDS   = ("cA2", "cH2", "cV2", "cD2", "cH1", "cV1", "cD1")
_RESIDUALS  = ("Laplacian residual", "median residual", "Gaussian residual")
_RES_STATS  = ("mean", "std", "kurtosis")
_PAIRS      = ("R-G", "R-B", "G-B")


def describe_feature(col0):
    """0-based column index -> what it actually measures."""
    if col0 < 51:                                   # Block A
        ch, k = divmod(col0, 17)
        if k < 7:
            what = _BASE_STATS[k]
        elif k < 11:
            what = _GLCM_STATS[k - 7]
        else:
            what = _EDGE_STATS[k - 11]
        return f"{what} ({_CHANNELS[ch]})"

    if col0 < 198:                                  # Block B
        ch, k = divmod(col0 - 51, 49)
        sub, stat = divmod(k, 7)
        return f"wavelet {_SUBBANDS[sub]} {_BASE_STATS[stat]} ({_CHANNELS[ch]})"

    if col0 < 218:                                  # Block C
        return f"FFT ring {col0 - 197} of 20"

    if col0 < 227:                                  # Block D
        f, k = divmod(col0 - 218, 3)
        return f"{_RESIDUALS[f]} {_RES_STATS[k]}"

    return f"channel correlation {_PAIRS[col0 - 227]}"


def block_of(col0):
    for name, lo, hi in BLOCKS:
        if lo <= col0 < hi:
            return name
    raise IndexError(col0)
