"""
extract_features.py - Python port of feature_extractor.m

Produces the same 230-feature vector, in the same order, so a model trained on
the MATLAB output can score images live without MATLAB in the loop.

Several MATLAB conventions differ from the obvious Python default, and getting
any of them wrong shifts a feature enough to move the classifier's decision
threshold. The ones that matter are marked MATLAB-CONVENTION below. Run
validate_port.py to confirm the port still agrees with your MATLAB CSV.

Use as a library:
    from extract_features import extract_features
    v = extract_features("photo.jpg")        # -> np.ndarray, shape (230,)

Or as a command, to score a whole folder into a CSV:
    python extract_features.py <folder> <out.csv> [label]
"""

from pathlib import Path
import sys

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage
from scipy.stats import skew, kurtosis

IMG_SIZE   = 256
NUM_RINGS  = 20
HIST_BINS  = 256
GLCM_LEVELS = 8
WAVELET    = "db4"
WAVE_LEVEL = 2

# MATLAB-CONVENTION: rgb2gray uses these exact ITU-R BT.601 weights, and
# returns uint8 - so the result is rounded before anything else touches it.
RGB2GRAY = (0.298936021293776, 0.587043074451121, 0.114020904255103)


# --------------------------------------------------------------- preprocessing
def load_rgb_uint8(source):
    """Any image -> HxWx3 uint8, matching feature_extractor.m's defensive path."""
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        arr = np.asarray(source)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Palette / grayscale / RGBA / CMYK all collapse to RGB here, the same way
    # ind2rgb + the channel fixups do on the MATLAB side.
    if img.mode != "RGB":
        img = img.convert("RGB")

    return np.asarray(img, dtype=np.uint8)


def standardise_size(img, mode="crop", target=IMG_SIZE):
    """Centre-crop at native scale, or scale the whole frame.

    Crop is the default: it resamples nothing, so images of different native
    resolutions are measured identically. See standardiseSize() in the MATLAB.
    """
    h, w = img.shape[:2]

    if mode == "resize":
        return np.asarray(
            Image.fromarray(img).resize((target, target), Image.BICUBIC),
            dtype=np.uint8)

    if mode != "crop":
        raise ValueError(f"mode must be 'crop' or 'resize', got {mode!r}")

    if h < target or w < target:
        scale = max(target / h, target / w)
        new = (int(np.ceil(w * scale)), int(np.ceil(h * scale)))
        img = np.asarray(Image.fromarray(img).resize(new, Image.BICUBIC),
                         dtype=np.uint8)
        h, w = img.shape[:2]

    # MATLAB-CONVENTION: origin snaps down to a multiple of 8 so the crop stays
    # aligned with the JPEG DCT block grid.
    r0 = (h - target) // 2
    c0 = (w - target) // 2
    r0 -= r0 % 8
    c0 -= c0 % 8
    return img[r0:r0 + target, c0:c0 + target, :]


def to_luminance(img):
    """MATLAB rgb2gray: weighted sum, rounded to uint8, then used as double."""
    g = (RGB2GRAY[0] * img[:, :, 0].astype(np.float64)
         + RGB2GRAY[1] * img[:, :, 1].astype(np.float64)
         + RGB2GRAY[2] * img[:, :, 2].astype(np.float64))
    return np.round(g).clip(0, 255)


# ------------------------------------------------------------------- statistics
def shannon_entropy(x, n_bins=HIST_BINS):
    """Histogram-based entropy in bits, valid for signed non-integer data."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = x.min(), x.max()
    if lo == hi:
        return 0.0
    counts, _ = np.histogram(x, bins=np.linspace(lo, hi, n_bins + 1))
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def higher_moments(x, mu):
    """Skewness and kurtosis, with degenerate cases zeroed.

    MATLAB-CONVENTION: skewness() and kurtosis() default to the *biased*
    estimators, and MATLAB's kurtosis is raw (3 for a normal), not excess.
    scipy defaults to the opposite on both counts.
    """
    if not np.any(x - mu):
        return 0.0, 0.0
    sk = float(skew(x, bias=True))
    ku = float(kurtosis(x, fisher=False, bias=True))
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 0.0
    return sk, ku


def compute_base_stats(data):
    """[Mean, Std, Variance, Energy, Entropy, Skewness, Kurtosis]."""
    x = np.asarray(data, dtype=np.float64).ravel()
    mu = float(x.mean())
    # MATLAB-CONVENTION: std/var normalise by N-1, numpy defaults to N.
    sd = float(x.std(ddof=1))
    vr = float(x.var(ddof=1))
    en = float(np.sum(x ** 2) / x.size)
    ent = shannon_entropy(x)
    sk, ku = higher_moments(x, mu)
    return [mu, sd, vr, en, ent, sk, ku]


# ------------------------------------------------------------------------ GLCM
def graycomatrix_matlab(channel, levels=GLCM_LEVELS, lo=0.0, hi=255.0):
    """MATLAB graycomatrix with NumLevels, GrayLimits, Offset [0 1], Symmetric.

    Written out rather than taken from skimage because the two libraries differ
    on quantisation and on what several properties mean - see below.
    """
    img = np.asarray(channel, dtype=np.float64)
    slope = (levels - 1) / (hi - lo)
    si = np.round(slope * img + (1 - slope * lo))
    si = np.clip(si, 1, levels).astype(np.int64) - 1     # 0-based bins

    a = si[:, :-1].ravel()      # offset [0 1]: pair with the neighbour to the right
    b = si[:, 1:].ravel()
    glcm = np.bincount(a * levels + b, minlength=levels * levels)
    glcm = glcm.reshape(levels, levels).astype(np.float64)
    return glcm + glcm.T        # Symmetric


def graycoprops_matlab(glcm):
    """Contrast, Correlation, Energy, Homogeneity - MATLAB's definitions.

    MATLAB-CONVENTION: MATLAB's Energy is the angular second moment, sum(p^2),
    while skimage's 'energy' is its square root; and MATLAB's Homogeneity
    divides by (1 + |i-j|) where skimage divides by (1 + (i-j)^2). Using
    skimage's versions here would silently produce different features.
    """
    p = glcm / glcm.sum()
    n = p.shape[0]
    i, j = np.mgrid[0:n, 0:n] + 1

    contrast    = float(np.sum(p * (i - j) ** 2))
    energy      = float(np.sum(p ** 2))
    homogeneity = float(np.sum(p / (1.0 + np.abs(i - j))))

    mi = np.sum(i * p)
    mj = np.sum(j * p)
    si = np.sqrt(np.sum(p * (i - mi) ** 2))
    sj = np.sqrt(np.sum(p * (j - mj) ** 2))
    corr = float(np.sum(p * (i - mi) * (j - mj)) / (si * sj)) if si > 0 and sj > 0 else 0.0
    if not np.isfinite(corr):
        corr = 0.0

    return [contrast, corr, energy, homogeneity]


# ---------------------------------------------------------------- feature blocks
# MATLAB-CONVENTION: imgradientxy('sobel') correlates with these kernels and
# replicates the border. imfilter defaults to correlation, not convolution.
_SOBEL_Y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
_SOBEL_X = _SOBEL_Y.T


def block_a(channel):
    """17 features: 7 base stats, 4 GLCM, 6 Sobel."""
    ch = np.asarray(channel, dtype=np.float64)

    feats = compute_base_stats(ch)
    feats += graycoprops_matlab(graycomatrix_matlab(ch))

    gx = ndimage.correlate(ch, _SOBEL_X, mode="nearest")
    gy = ndimage.correlate(ch, _SOBEL_Y, mode="nearest")
    gmag = np.hypot(gx, gy)

    feats += [float(gx.mean()), float(gx.var(ddof=1)),
              float(gy.mean()), float(gy.var(ddof=1)),
              float(gmag.mean()), float(gmag.var(ddof=1))]
    return feats


def block_b(channel):
    """49 features: base stats for cA2, cH2, cV2, cD2, cH1, cV1, cD1."""
    ch = np.asarray(channel, dtype=np.float64)
    # MATLAB's dwtmode default is 'sym' (half-point symmetric extension).
    coeffs = pywt.wavedec2(ch, WAVELET, mode="symmetric", level=WAVE_LEVEL)
    cA2 = coeffs[0]
    cH2, cV2, cD2 = coeffs[1]
    cH1, cV1, cD1 = coeffs[2]

    feats = []
    for sub in (cA2, cH2, cV2, cD2, cH1, cV1, cD1):
        feats += compute_base_stats(sub)
    return feats


def block_c(gray, n_rings=NUM_RINGS):
    """20 features: mean FFT magnitude in concentric rings, centre outward."""
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    rows, cols = mag.shape

    cy = rows // 2          # where fftshift puts DC
    cx = cols // 2
    yy, xx = np.mgrid[0:rows, 0:cols]
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    r_max = radius.max()
    idx = np.floor(radius / (r_max + np.spacing(r_max)) * n_rings).astype(int)
    idx = np.clip(idx, 0, n_rings - 1)

    totals = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=n_rings)
    counts = np.bincount(idx.ravel(), minlength=n_rings)
    return list(np.where(counts > 0, totals / np.maximum(counts, 1), 0.0))


def _residual_stats(r):
    x = np.asarray(r, dtype=np.float64).ravel()
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    _, ku = higher_moments(x, mu)
    return [mu, sd, ku]


def block_d(gray):
    """9 features: mean/std/kurtosis of three high-pass residuals."""
    lap_kernel = np.array([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
    res_lap = ndimage.correlate(gray, lap_kernel, mode="nearest")

    # MATLAB medfilt2 'symmetric' padding == scipy's 'reflect'.
    res_med = gray - ndimage.median_filter(gray, size=3, mode="reflect")

    # MATLAB imgaussfilt(x, 1) uses a 2*ceil(2*sigma)+1 kernel and replicates.
    res_gau = gray - ndimage.gaussian_filter(gray, sigma=1.0, mode="nearest",
                                             truncate=2.0)

    return _residual_stats(res_lap) + _residual_stats(res_med) + _residual_stats(res_gau)


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0


def block_e(img):
    """3 features: Pearson correlation for R-G, R-B, G-B."""
    r = img[:, :, 0].astype(np.float64).ravel()
    g = img[:, :, 1].astype(np.float64).ravel()
    b = img[:, :, 2].astype(np.float64).ravel()
    return [_pearson(r, g), _pearson(r, b), _pearson(g, b)]


# ----------------------------------------------------------------------- public
def extract_features(source, mode="crop"):
    """Return the 230-feature vector for one image, in feature_extractor.m order."""
    img = standardise_size(load_rgb_uint8(source), mode=mode)
    gray = to_luminance(img)

    feats = []
    for c in range(3):
        feats += block_a(img[:, :, c])
    for c in range(3):
        feats += block_b(img[:, :, c])
    feats += block_c(gray)
    feats += block_d(gray)
    feats += block_e(img)

    v = np.asarray(feats, dtype=np.float64)
    if v.size != 230:
        raise RuntimeError(f"expected 230 features, built {v.size}")
    v[~np.isfinite(v)] = 0.0
    return v


# ------------------------------------------------------------- naming features
_CHANNELS  = ("red", "green", "blue")
_BASE_STATS = ("mean", "std", "variance", "energy", "entropy", "skewness", "kurtosis")
_GLCM_STATS = ("GLCM contrast", "GLCM correlation", "GLCM energy", "GLCM homogeneity")
_EDGE_STATS = ("mean Gx", "var Gx", "mean Gy", "var Gy", "mean |G|", "var |G|")
_SUBBANDS   = ("cA2", "cH2", "cV2", "cD2", "cH1", "cV1", "cD1")
_RESIDUALS  = ("Laplacian residual", "median residual", "Gaussian residual")
_RES_STATS  = ("mean", "std", "kurtosis")
_PAIRS      = ("R-G", "R-B", "G-B")


def describe_feature(col0):
    """Turn a 0-based column index into what it actually measures."""
    if col0 < 51:                                   # Block A
        ch, k = divmod(col0, 17)
        if k < 7:
            what = f"{_BASE_STATS[k]}"
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
    for name, lo, hi in (("A spatial", 0, 51), ("B wavelet", 51, 198),
                         ("C fft", 198, 218), ("D residual", 218, 227),
                         ("E corr", 227, 230)):
        if lo <= col0 < hi:
            return name
    raise IndexError(col0)


def size_warning(source, target=IMG_SIZE):
    """Warn when an image is too small to crop from, or None when it is fine.

    Below the crop size the image has to be scaled up first, and two things go
    wrong at once: MATLAB's imresize and PIL's bicubic do not agree, so the
    features drift away from the ones the model was trained on; and the model
    never saw an upscaled image during training - every dataset image was at
    least 450px - so such a vector is out of distribution regardless. Treat any
    prediction on a small image as unreliable rather than trusting it.
    """
    img = load_rgb_uint8(source)
    h, w = img.shape[:2]
    if h < target or w < target:
        return (f"Image is {w}x{h}, smaller than the {target}x{target} analysis "
                f"window, so it was scaled up before measurement. The model was "
                f"trained only on images large enough to crop, so this result is "
                f"unreliable.")
    return None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    folder, out_csv = sys.argv[1], sys.argv[2]
    label = float(sys.argv[3]) if len(sys.argv) > 3 else None

    files = list_images(folder)
    if not files:
        sys.exit(f"No images found in {folder}")

    rows, kept, skipped = [], [], 0
    for i, f in enumerate(files, 1):
        try:
            v = extract_features(f)
            rows.append(v if label is None else np.append(v, label))
            kept.append(str(f))
        except Exception as exc:
            skipped += 1
            print(f"SKIPPED: {f}\n         {exc}", file=sys.stderr)
        if i % 100 == 0:
            print(f"Processed {i} / {len(files)} images ({skipped} skipped)")

    matrix = np.vstack(rows)
    np.savetxt(out_csv, matrix, delimiter=",", fmt="%.15g")
    Path(out_csv).with_suffix(".filenames.txt").write_text("\n".join(kept) + "\n")

    print(f"\nFound {len(files)}, processed {len(rows)}, skipped {skipped}")
    print(f"Wrote {matrix.shape[0]} x {matrix.shape[1]} to {out_csv}")


if __name__ == "__main__":
    main()
