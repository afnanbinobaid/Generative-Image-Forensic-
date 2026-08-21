"""
demo.py - live AI-image detector for the presentation

Drop in an image, get a verdict, and see the actual DSP measurements behind it:
the analysis crop, its Fourier spectrum, the finest wavelet detail band, and the
high-pass noise residual. Those four panels are the point - they show the
classifier is reading signal-processing evidence, not a learned texture.

    python train_model.py     (once, produces model.joblib)
    python demo.py            (opens in a browser)

Add --share to get a temporary public URL, useful when presenting from a
machine that is not the one driving the projector.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pywt
import gradio as gr

from extract_features import (extract_features, load_rgb_uint8, standardise_size,
                              to_luminance, size_warning, describe_feature,
                              WAVELET, WAVE_LEVEL)

MODEL_PATH = Path("model.joblib")

# Gradio 6 moved `theme` from the Blocks constructor to launch(); 5 and earlier
# want it on Blocks. Support whichever is installed rather than pinning.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_KWARGS = {} if _GRADIO_MAJOR >= 6 else {"theme": gr.themes.Soft()}
_LAUNCH_KWARGS = {"theme": gr.themes.Soft()} if _GRADIO_MAJOR >= 6 else {}

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
REAL_COLOUR, AI_COLOUR = "#2a78d6", "#eb6834"


def load_bundle():
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: {MODEL_PATH.resolve()} not found.\n"
                 f"Run  python train_model.py  first.")
    return joblib.load(MODEL_PATH)


BUNDLE = load_bundle()


# ------------------------------------------------------------------ DSP panels
def dsp_panels(img_rgb):
    """The four signal-processing views the verdict is actually based on."""
    crop = standardise_size(load_rgb_uint8(img_rgb))
    gray = to_luminance(crop)

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

    coeffs = pywt.wavedec2(gray, WAVELET, mode="symmetric", level=WAVE_LEVEL)
    cD1 = coeffs[2][2]

    from scipy import ndimage
    residual = gray - ndimage.gaussian_filter(gray, sigma=1.0, mode="nearest",
                                              truncate=2.0)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), facecolor=SURFACE)
    panels = [
        (crop, None, "Analysis crop", "256x256 at native scale, no resampling"),
        (spectrum, "magma", "Fourier spectrum", "log magnitude, centre = low frequency"),
        (np.abs(cD1), "magma", "Wavelet cD1", "finest diagonal detail - where the signal is"),
        (residual, "coolwarm", "Noise residual", "image minus its Gaussian blur"),
    ]
    for ax, (data, cmap, title, sub) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        elif cmap == "coolwarm":
            lim = max(np.abs(data).max(), 1e-9)
            ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=9)
        ax.set_xlabel(sub, color=INK_MUTED, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)

    fig.tight_layout()
    return fig


def score_gauge(prob, threshold):
    """Where this image's score falls relative to the decision threshold."""
    fig, ax = plt.subplots(figsize=(7.6, 1.75), facecolor=SURFACE)

    ax.axhspan(0, 1, xmin=0, xmax=threshold, color=REAL_COLOUR, alpha=0.16)
    ax.axhspan(0, 1, xmin=threshold, xmax=1, color=AI_COLOUR, alpha=0.16)
    ax.axvline(threshold, color=INK, linewidth=2)
    ax.text(threshold, 1.14, f"threshold {threshold:.2f}", color=INK, fontsize=9,
            fontweight="bold", ha="center")

    colour = AI_COLOUR if prob >= threshold else REAL_COLOUR
    ax.plot([prob], [0.5], marker="o", markersize=17, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=5)
    ax.text(prob, -0.42, f"{prob:.3f}", color=colour, fontsize=12,
            fontweight="bold", ha="center")

    ax.text(0.01, 0.5, "REAL", color=REAL_COLOUR, fontsize=10, fontweight="bold",
            va="center", ha="left")
    ax.text(0.99, 0.5, "AI", color=AI_COLOUR, fontsize=10, fontweight="bold",
            va="center", ha="right")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig


def evidence_table(vec):
    """This image's key measurements against typical real and typical AI values."""
    top = BUNDLE["top_features"]
    m_real, m_ai, d = BUNDLE["mean_real"], BUNDLE["mean_ai"], BUNDLE["cohens_d"]

    rows = []
    for c in top[:8]:
        v, r, a = vec[c], m_real[c], m_ai[c]
        # which class this single measurement sits closer to
        leans = "real" if abs(v - r) < abs(v - a) else "AI"
        rows.append([describe_feature(int(c)), f"{v:.4g}", f"{r:.4g}",
                     f"{a:.4g}", leans, f"{d[c]:.2f}"])
    return rows


def analyse(image):
    # Gradio has already rendered whatever the previous call returned, so the
    # old figures can go. Without this every upload leaks two figures and a
    # long demo session eventually exhausts matplotlib.
    plt.close("all")

    if image is None:
        return "### Drop an image above to analyse it", None, None, []

    warning = size_warning(image)
    vec = extract_features(image)
    prob = float(BUNDLE["model"].predict_proba(vec.reshape(1, -1))[0, 1])
    threshold = float(BUNDLE["threshold"])

    is_ai = prob >= threshold
    # distance from the threshold, normalised to the room available on that side
    room = (1 - threshold) if is_ai else threshold
    margin = abs(prob - threshold) / max(room, 1e-9)
    strength = "strong" if margin > 0.6 else "moderate" if margin > 0.25 else "weak"

    verdict = "AI-GENERATED" if is_ai else "REAL PHOTOGRAPH"
    colour = AI_COLOUR if is_ai else REAL_COLOUR

    md = (f"<div style='padding:18px 20px;border-radius:8px;"
          f"border-left:5px solid {colour};background:rgba(0,0,0,.03)'>"
          f"<div style='font-size:1.9rem;font-weight:700;color:{colour};"
          f"letter-spacing:-.02em'>{verdict}</div>"
          f"<div style='margin-top:6px;color:#52514e'>score {prob:.3f} vs "
          f"threshold {threshold:.3f} &nbsp;·&nbsp; <b>{strength}</b> separation"
          f"</div></div>")

    if warning:
        md += (f"\n\n<div style='padding:12px 16px;margin-top:12px;border-radius:6px;"
               f"border-left:4px solid #C24A12;background:rgba(194,74,18,.08)'>"
               f"<b>Unreliable:</b> {warning}</div>")

    return md, score_gauge(prob, threshold), dsp_panels(image), evidence_table(vec)


# ------------------------------------------------------------------------- UI
def build():
    gens = ", ".join(BUNDLE.get("generators", [])) or "the training generators"
    subtitle = (f"230 classical DSP features · no neural network · "
                f"held-out AUC {BUNDLE['roc_auc']:.3f}, "
                f"accuracy {BUNDLE['accuracy']:.1%} · trained on {gens}")

    with gr.Blocks(title="AI Image Detector", **_BLOCKS_KWARGS) as app:
        gr.Markdown(f"# Detecting AI-generated images with signal processing\n{subtitle}")

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="numpy", label="Image to analyse", height=340)
                run = gr.Button("Analyse", variant="primary", size="lg")
            with gr.Column(scale=1):
                verdict = gr.Markdown("### Drop an image to the left to analyse it")
                gauge = gr.Plot(label="Where the score falls")

        gr.Markdown("### What the detector actually measured")
        panels = gr.Plot(label="")

        gr.Markdown("### The evidence, feature by feature\n"
                    "Each row is one DSP measurement: this image's value, "
                    "against the dataset average for real and for AI images. "
                    "Cohen's *d* is how strongly that feature separates the two "
                    "classes overall.")
        table = gr.Dataframe(
            headers=["measurement", "this image", "typical real", "typical AI",
                     "leans", "Cohen's d"],
            datatype=["str"] * 6, interactive=False, wrap=True)

        outputs = [verdict, gauge, panels, table]
        run.click(analyse, inputs=image, outputs=outputs)
        image.change(analyse, inputs=image, outputs=outputs)

    return app


if __name__ == "__main__":
    build().launch(share="--share" in sys.argv, inbrowser=True,
                   **_LAUNCH_KWARGS)
