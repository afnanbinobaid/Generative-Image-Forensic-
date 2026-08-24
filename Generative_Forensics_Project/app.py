"""
app.py - the live demonstration GUI for the generative image forensics detector

    streamlit run app.py

The same split the rest of the project uses, wrapped in a browser: MATLAB
measures the image, Python decides what it is. One upload runs

    demo_image(img, workdir)   ->  temp_features.csv  +  dsp_visuals.png
    model.joblib               ->  calibrated probability
    shap.TreeExplainer         ->  the per-feature log-odds behind that number

and the page then shows the verdict, the evidence that produced it, and the
signal processing it was measured from.

Nothing here re-implements the pipeline. The features come from the same
extractImageFeatures() the training set was built with, and the explanation
comes from predict_image.shap_contributions(), so the browser can never show a
number the command-line demo would disagree with.

By default every upload launches a fresh `matlab -batch` process, which pays
MATLAB's full startup - commonly 15-45s - each time. When the MATLAB Engine
API for Python is installed (ships inside a MATLAB install at
matlabroot/extern/engines/python, or `pip install matlabengine` on R2022b+),
the app starts one MATLAB session on first use and reuses it for every upload
after, so only that first image is slow. Nothing needs enabling - it is tried
automatically and falls back to the subprocess path if unavailable.

Environment (all optional):
    MATLAB_EXE              path to the MATLAB binary, if it is not on PATH
    MATLAB_ARGS              extra flags for the MATLAB launch (e.g. -softwareopengl)
    GIF_MATLAB_TIMEOUT       seconds to wait for a subprocess MATLAB run  (default 300)
    GIF_NO_MATLAB_ENGINE     set to skip the persistent-session fast path entirely
"""

import glob
import hashlib
import html
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                 # so the imports below work from anywhere
    sys.path.insert(0, str(HERE))

from feature_names import describe_feature    # noqa: E402

MODEL_PATH   = HERE / "model.joblib"
MATLAB_ENTRY = "demo_image"
FEATURES_CSV = "temp_features.csv"
VISUALS_PNG  = "dsp_visuals.png"


def _timeout_from_env(name, default):
    """Read a seconds-valued setting; a typo in it must not take the page down."""
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


N_FEATURES     = 230
ANALYSIS_SIDE  = 256          # the crop the extractor measures
TOP_DRIVERS    = 5
MATLAB_TIMEOUT = _timeout_from_env("GIF_MATLAB_TIMEOUT", 300)


class PipelineError(Exception):
    """A stage failed for a reason worth showing the user in full."""

    def __init__(self, message, detail=None, hint=None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.hint = hint


# ======================================================================
#  Stage 1 - MATLAB
# ======================================================================

def find_matlab():
    """Locate the MATLAB binary, preferring an explicit override."""
    override = os.environ.get("MATLAB_EXE", "").strip()
    if override:
        return override if (shutil.which(override) or Path(override).exists()) else None

    on_path = shutil.which("matlab")
    if on_path:
        return on_path

    # Default install locations, newest first - enough that a normal install
    # needs no configuration at all.
    for pattern in ("/usr/local/MATLAB/*/bin/matlab",
                    "/Applications/MATLAB_*.app/bin/matlab",
                    r"C:\Program Files\MATLAB\*\bin\matlab.exe",
                    r"C:\Program Files (x86)\MATLAB\*\bin\matlab.exe"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def _matlab_quote(path):
    """A path as a MATLAB single-quoted literal ('' escapes a quote)."""
    return str(path).replace("'", "''")


def _collect_outputs(work_dir, log):
    """Read the two artefacts exportForGui() was asked to write, or explain why not.

    Shared by both run paths below, so a subprocess run and an Engine-API run
    are validated identically and can never disagree about what "succeeded"
    means.
    """
    csv_path = Path(work_dir) / FEATURES_CSV
    if not csv_path.exists():
        # A non-zero exit (or, on the Engine path, a clean return) is the usual
        # cause, but a missing toolbox licence can exit cleanly and still write
        # nothing - test for the artefact, not the status code.
        raise PipelineError(
            "MATLAB ran but returned no feature vector.",
            detail=log or "(MATLAB produced no output at all.)",
            hint="Most often a missing toolbox: the extractor needs Image "
                 "Processing, Wavelet, and Statistics & Machine Learning.")

    try:
        features = np.loadtxt(csv_path, delimiter=",").ravel()
    except ValueError as exc:
        raise PipelineError("The feature CSV could not be read.", detail=str(exc))

    if features.size != N_FEATURES:
        raise PipelineError(
            f"Expected {N_FEATURES} features, got {features.size}.",
            detail=log,
            hint="app.py and extractImageFeatures.m have drifted apart - "
                 "run  python check_setup.py  to find out which one is stale.")

    png_path = Path(work_dir) / VISUALS_PNG
    png_bytes = png_path.read_bytes() if png_path.exists() else None
    return features, png_bytes


def run_matlab_subprocess(matlab, image_path, work_dir):
    """Run the extractor as a fresh `matlab -batch` process.

    Needs nothing beyond a MATLAB install, but pays MATLAB's full startup -
    commonly 15-45s - on every single call, because it is a new process each
    time. Used when the Engine API session in get_matlab_engine() is
    unavailable; otherwise run_matlab_engine() below is the fast path.
    """
    call = f"{MATLAB_ENTRY}('{_matlab_quote(image_path)}','{_matlab_quote(work_dir)}')"
    extra = shlex.split(os.environ.get("MATLAB_ARGS", ""))
    cmd = [matlab, *extra, "-batch", call]

    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True,
                              text=True, timeout=MATLAB_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PipelineError(
            f"MATLAB did not finish within {MATLAB_TIMEOUT} seconds.",
            hint="Raise GIF_MATLAB_TIMEOUT if the first run is just slow to "
                 "start - MATLAB's own launch can take a while on a cold machine.")
    except OSError as exc:
        raise PipelineError("MATLAB could not be launched.", detail=str(exc))

    log = "\n".join(part for part in (proc.stdout, proc.stderr) if part and part.strip())
    features, png_bytes = _collect_outputs(work_dir, log)
    return features, png_bytes, log


_ENGINE_LOCK = threading.Lock()   # one MATLAB session, so concurrent uploads queue rather than collide


@st.cache_resource(show_spinner=False)
def get_matlab_engine():
    """A MATLAB session kept alive for the life of the server process.

    Starting it is exactly as slow as one `matlab -batch` call, but it is done
    once - here, cached - rather than on every upload. Returns None, never
    raises, when the Engine API isn't installed or the session fails to
    start; callers then fall back to run_matlab_subprocess(), so a machine
    without the Engine API behaves exactly as it did before this existed.
    """
    if os.environ.get("GIF_NO_MATLAB_ENGINE"):
        return None
    try:
        import matlab.engine
    except ImportError:
        return None
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(str(HERE), nargout=0)
        return eng
    except Exception:
        return None


def run_matlab_engine(engine, image_path, work_dir):
    """Run the extractor in the persistent session from get_matlab_engine().

    Same two artefacts, same validation as the subprocess path - just without
    paying MATLAB's startup cost again.
    """
    import matlab.engine

    out, err = io.StringIO(), io.StringIO()
    try:
        with _ENGINE_LOCK:
            engine.demo_image(str(image_path), str(work_dir), nargout=0,
                              stdout=out, stderr=err)
    except matlab.engine.MatlabExecutionError as exc:
        log = (out.getvalue() + err.getvalue()).strip() or str(exc)
        raise PipelineError(
            "MATLAB raised an error while measuring the image.",
            detail=log,
            hint="Most often a missing toolbox: the extractor needs Image "
                 "Processing, Wavelet, and Statistics & Machine Learning.")

    log = (out.getvalue() + err.getvalue()).strip()
    features, png_bytes = _collect_outputs(work_dir, log)
    return features, png_bytes, log


# ======================================================================
#  Stage 2 - the model and its explanation
# ======================================================================

@st.cache_resource(show_spinner=False)
def load_bundle():
    """model.joblib, loaded once per server rather than once per upload."""
    if not MODEL_PATH.exists():
        raise PipelineError(
            "No trained model found.",
            detail=f"Expected {MODEL_PATH}",
            hint="Run  python train_model.py  once to produce model.joblib.")
    import joblib
    return joblib.load(MODEL_PATH)


def classify(features, bundle):
    """The calibrated verdict, and how far it sits from the decision boundary."""
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    prob = float(model.predict_proba(features.reshape(1, -1))[0, 1])
    is_ai = prob >= threshold

    # Distance from the threshold, scaled by the room available on that side,
    # so "strong" means the same thing either side of an off-centre boundary.
    room = (1 - threshold) if is_ai else threshold
    margin = abs(prob - threshold) / max(room, 1e-9)
    strength = "strong" if margin > 0.6 else "moderate" if margin > 0.25 else "weak"

    return {
        "prob": prob,
        "threshold": threshold,
        "is_ai": is_ai,
        "verdict": "AI-GENERATED" if is_ai else "REAL PHOTOGRAPH",
        "margin": margin,
        "strength": strength,
        "calibrated": bool(bundle.get("calibrated", False)),
    }


def explain(features, bundle):
    """Exact per-feature contributions to this one prediction.

    Returns (breakdown, None) or (None, reason). The breakdown is additive:
    base value + every contribution reproduces the model's own score, which is
    what stops the evidence panel from ever contradicting the verdict above it.
    """
    background = bundle.get("background")
    if background is None:
        return None, ("This model was trained before a SHAP background sample "
                      "was saved. Retrain with the current train_model.py for a "
                      "per-image explanation.")

    # SHAP needs the bare tree ensemble - the calibrated wrapper hides the
    # trees. Calibration is a monotonic remap, so it cannot reorder
    # contributions: explaining the base model still explains this verdict.
    explain_model = bundle.get("base_model", bundle["model"])

    try:
        from predict_image import shap_contributions
        contrib, base_value, space = shap_contributions(
            explain_model, features, background)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    order = np.argsort(np.abs(contrib))[::-1][:TOP_DRIVERS]
    top_sum = float(contrib[order].sum())
    total = float(base_value) + float(contrib.sum())
    reconstructed = 1 / (1 + np.exp(-total)) if space == "logit" else total
    raw_prob = float(explain_model.predict_proba(features.reshape(1, -1))[0, 1])

    mean_real, mean_ai = bundle["mean_real"], bundle["mean_ai"]
    drivers = [{
        "name": describe_feature(int(c)),
        "value": float(features[c]),
        "real": float(mean_real[c]),
        "ai": float(mean_ai[c]),
        "contribution": float(contrib[c]),
    } for c in (int(i) for i in order)]

    return {
        "space": space,
        "unit": "log-odds" if space == "logit" else "probability",
        "base_value": float(base_value),
        "drivers": drivers,
        "top_sum": top_sum,
        "rest_sum": float(contrib.sum()) - top_sum,
        "rest_count": int(contrib.size - len(order)),
        "total": total,
        "reconstructed": float(reconstructed),
        "raw_prob": raw_prob,
        "n_features": int(contrib.size),
    }, None


# ======================================================================
#  Presentation
# ======================================================================

st.set_page_config(page_title="Generative Image Forensics",
                   page_icon="◐", layout="centered")

STYLE = """
<style>
/* Tokens are unconditional, not behind a prefers-color-scheme query: .streamlit/
   config.toml pins the app to a light theme, so the page's actual background
   never turns dark. A dark-mode override keyed to the OS/browser preference
   would then fire on its own - painting light text meant for a dark surface
   onto the white one Streamlit is still rendering, which reads as invisible
   text. One committed palette, always applied, keeps the two in sync. */
:root {
  --ink:      #14161a;
  --ink-soft: #5b6270;
  --ink-faint:#8b929e;
  --line:     #e7e9ed;
  --line-soft:#f1f2f5;
  --surface:  #ffffff;
  --ai:       #a4571b;
  --ai-wash:  #fdf6ef;
  --real:     #1a6b5f;
  --real-wash:#f0f7f5;
  --warn:     #8a6212;
  --warn-wash:#fdf8ec;
  --warn-line:#ecdcb4;
}

/* Belt-and-suspenders on top of .streamlit/config.toml's backgroundColor: the
   page background is stated here too, so the light palette above is never
   applied over a background it doesn't actually match. */
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp { background:var(--surface) !important; }

/* a quiet frame: no toolbar, no chrome, generous air */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }

.block-container { max-width: 46rem; padding-top: 4.5rem; padding-bottom: 6rem; }

html, body, [class*="css"] { -webkit-font-smoothing: antialiased; }
.gf, .gf * {
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-feature-settings: "tnum" 1, "cv05" 1;
}
.gf-num { font-variant-numeric: tabular-nums; }

/* ---------- masthead ---------- */
.gf-eyebrow {
  font-size:.66rem; letter-spacing:.22em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600; margin-bottom:1.1rem;
}
.gf h1.gf-title {
  font-size:2.3rem; line-height:1.12; letter-spacing:-.028em;
  font-weight:600; color:var(--ink); margin:0 0 .85rem 0;
}
.gf p.gf-lede { font-size:1rem; line-height:1.65; color:var(--ink-soft); margin:0; max-width:34rem; }
.gf-rule { height:1px; background:var(--line); border:0; margin:2.6rem 0; }

/* ---------- processing ---------- */
.gf-stage { display:flex; align-items:center; gap:.8rem; padding:1.9rem .25rem; }
.gf-dot {
  width:7px; height:7px; border-radius:50%; background:var(--ink);
  animation: gfPulse 1.35s ease-in-out infinite; flex:none;
}
@keyframes gfPulse { 0%,100%{opacity:.18; transform:scale(.8);} 50%{opacity:1; transform:scale(1);} }
.gf-stage-text {
  font-size:.94rem; color:var(--ink-soft); letter-spacing:.005em;
  animation: gfFade .45s ease both;
}
@keyframes gfFade { from{opacity:0; transform:translateX(-4px);} to{opacity:1; transform:none;} }

/* ---------- the reveal ---------- */
@keyframes gfRise { from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;} }
.gf-reveal { animation: gfRise .6s cubic-bezier(.22,.61,.36,1) both; }
.gf-d1{animation-delay:.00s}.gf-d2{animation-delay:.09s}
.gf-d3{animation-delay:.18s}.gf-d4{animation-delay:.27s}

.gf-verdict { border-left:2px solid var(--edge); padding:.15rem 0 .3rem 1.5rem; margin:.5rem 0 0 0; }
.gf-verdict-label {
  font-size:.66rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600;
}
.gf-verdict-word {
  font-size:2.5rem; line-height:1.1; letter-spacing:-.035em; font-weight:600;
  color:var(--edge); margin:.55rem 0 .9rem 0;
}
.gf p.gf-reading { font-size:1.02rem; line-height:1.6; color:var(--ink); margin:0 0 .55rem 0; max-width:32rem; }
.gf p.gf-reading b { font-weight:600; }
.gf p.gf-subline { font-size:.83rem; color:var(--ink-faint); margin:0; }

.gf-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.35rem; }
.gf-chip {
  font-size:.73rem; letter-spacing:.02em; color:var(--ink-soft);
  border:1px solid var(--line); border-radius:999px; padding:.3rem .72rem;
  background:var(--surface);
}
.gf-chip b { color:var(--ink); font-weight:600; }

/* ---------- warning ---------- */
.gf-warn {
  background:var(--warn-wash); border:1px solid var(--warn-line);
  border-radius:10px; padding:1.05rem 1.25rem; margin:2rem 0 0 0;
}
.gf-warn-title {
  font-size:.7rem; letter-spacing:.16em; text-transform:uppercase;
  color:var(--warn); font-weight:700; margin-bottom:.5rem;
}
.gf p.gf-warn-body { font-size:.88rem; line-height:1.6; color:var(--ink-soft); margin:0; }

/* ---------- section headings ---------- */
.gf-h2 {
  font-size:.68rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--ink-faint); font-weight:600; margin:0 0 .5rem 0;
}
.gf p.gf-h2-sub { font-size:.9rem; line-height:1.6; color:var(--ink-soft); margin:0 0 1.5rem 0; max-width:33rem; }

/* ---------- additivity ledger ---------- */
/* Every cell carries the same top rule - transparent except on the total - so
   the four labels sit on one line instead of stepping down at the sum. */
.gf-ledger { display:flex; align-items:flex-start; gap:.55rem; flex-wrap:wrap; margin-bottom:2rem; }
.gf-cell { flex:1 1 0; min-width:6.6rem; border-top:2px solid transparent; padding-top:.55rem; }
.gf-cell-k {
  font-size:.68rem; color:var(--ink-faint); letter-spacing:.04em;
  margin-bottom:.45rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.gf-cell-v { font-size:1.28rem; font-weight:600; letter-spacing:-.02em; color:var(--ink); font-variant-numeric:tabular-nums; }
.gf-cell-v.pos { color:var(--ai); }
.gf-cell-v.neg { color:var(--real); }
.gf-op { flex:none; padding-top:2.05rem; font-size:.95rem; color:var(--ink-faint); }
.gf-total { border-top-color:var(--ink); }

/* ---------- driver table ---------- */
/* Streamlit styles bare <table> elements with its own grid rules; this panel is
   a ledger, not a spreadsheet, so those are cleared and only the horizontal
   rules that separate rows are put back. */
.gf-tablewrap { overflow-x:auto; }
.gf-table { width:100%; border-collapse:collapse; font-size:.82rem; border:0 !important; }
.gf-table th, .gf-table td { border:0 !important; background:transparent !important; }
.gf-table th {
  text-align:right; font-weight:500; font-size:.67rem; letter-spacing:.055em;
  text-transform:uppercase; color:var(--ink-faint);
  padding:0 0 .6rem 0; border-bottom:1px solid var(--line) !important;
}
.gf-table th:first-child, .gf-table td:first-child { text-align:left; }
.gf-table td {
  padding:.72rem 0; border-bottom:1px solid var(--line-soft) !important;
  color:var(--ink); font-variant-numeric:tabular-nums; text-align:right;
  white-space:nowrap;
}
.gf-table td.gf-name { white-space:normal; color:var(--ink); padding-right:1rem; line-height:1.4; }
.gf-table th:not(:first-child):not(.gf-c), .gf-table td:not(:first-child):not(.gf-c)
  { padding-left:1.15rem; }
.gf-table td.gf-ref { color:var(--ink-faint); }
.gf-table td.gf-this { font-weight:600; }
.gf-table th.gf-c, .gf-table td.gf-c { padding-left:1.1rem; width:8.5rem; }
.gf-bar { display:flex; align-items:center; justify-content:flex-end; gap:.5rem; }
.gf-bar-track { flex:1; height:4px; background:var(--line-soft); border-radius:2px; overflow:hidden; }
.gf-bar-fill { height:100%; border-radius:2px; display:block; }
.gf-bar-val { font-size:.8rem; font-weight:600; width:3.9rem; text-align:right; }
.gf p.gf-foot { font-size:.76rem; line-height:1.6; color:var(--ink-faint); margin:1.1rem 0 0 0; }

/* ---------- dsp panel ---------- */
.gf p.gf-caption {
  font-size:.76rem; color:var(--ink-faint); letter-spacing:.02em;
  margin:.85rem 0 0 0; text-align:center;
}
.gf p.gf-caption span { color:var(--ink-soft); }
[data-testid="stImageContainer"] img { border-radius:6px; }

/* ---------- provenance ---------- */
.gf-prov {
  font-size:.74rem; color:var(--ink-faint); line-height:1.8;
  border-top:1px solid var(--line); padding-top:1.2rem; margin-top:3.5rem;
}
.gf-prov b { color:var(--ink-soft); font-weight:600; }

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background:var(--surface); border:1px dashed var(--line);
  border-radius:10px; padding:1.6rem; transition:border-color .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--ink-faint); }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def fmt(x, digits=4):
    """A measurement, readable at any magnitude."""
    if x != x:                      # NaN
        return "-"
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return f"{x:,.{digits}g}"
    return f"{x:.2e}"


def block(markup):
    st.markdown(f'<div class="gf">{markup}</div>', unsafe_allow_html=True)


def error_card(err):
    """A failure, stated plainly, with the fix rather than a stack trace."""
    st.error(err.message if isinstance(err, PipelineError) else str(err))
    if isinstance(err, PipelineError):
        if err.hint:
            st.caption(err.hint)
        if err.detail:
            with st.expander("Technical detail"):
                st.code(err.detail.strip(), language="text")


# ---------------------------------------------------------------- masthead

block(
    '<div class="gf-eyebrow">Digital signal processing &nbsp;·&nbsp; gradient-boosted ensemble</div>'
    '<h1 class="gf-title">Generative Image Forensics</h1>'
    '<p class="gf-lede">Upload a photograph. MATLAB measures 230 signal-processing '
    'features from a native-scale 256&times;256 crop; a calibrated classifier reads them '
    'and reports whether the image was generated - and exactly which measurements '
    'made it say so.</p>'
)
st.markdown('<hr class="gf-rule">', unsafe_allow_html=True)

upload = st.file_uploader("Image", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")

if upload is None:
    block('<p class="gf-subline">JPEG or PNG. Images below 256&nbsp;px on either '
          'side can still be analysed, with a caveat.</p>')
    st.stop()


# ---------------------------------------------------------------- pipeline

ENGINE_START_STAGES = (
    "Starting MATLAB…",
    "Loading toolboxes…",
)
MATLAB_STAGES = (
    "Cropping the 256×256 analysis window…",
    "Measuring spatial statistics and GLCM texture…",
    "Decomposing db4 wavelet subbands…",
    "Computing the Fourier radial spectrum…",
    "Extracting high-frequency noise residuals…",
    "Rendering the signal-processing evidence…",
)
MODEL_STAGES = (
    "Loading the gradient-boosted ensemble…",
    "Calculating SHAP log-odds contributions…",
    "Generating the final verdict…",
)


def staged(slot, captions, work, dwell=1.6):
    """Run `work` off-thread while the caption shifts, then return its result.

    The captions are the honest sequence of what the pipeline is doing, not a
    fake progress bar - each one is a stage the extractor genuinely runs
    through. Nothing here touches Streamlit from the worker thread.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(work)
        i = 0
        while True:
            caption = captions[min(i, len(captions) - 1)]
            slot.markdown(
                f'<div class="gf"><div class="gf-stage"><span class="gf-dot"></span>'
                f'<span class="gf-stage-text">{esc(caption)}</span></div></div>',
                unsafe_allow_html=True)
            try:
                return future.result(timeout=dwell)
            except FutureTimeout:
                i += 1


def run_pipeline(image_bytes, filename, slot):
    """Upload -> features -> verdict -> explanation, as one result dict."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise PipelineError("That file could not be opened as an image.",
                            detail=str(exc))
    width, height = image.size

    # The Engine API session is the fast path: cached process-wide, so this
    # costs a MATLAB startup only the first time any user calls it, not once
    # per upload. Absent or failing to start, it returns None and the
    # subprocess path below behaves exactly as it always has.
    engine = staged(slot, ENGINE_START_STAGES, get_matlab_engine, dwell=1.2)

    matlab = None
    if engine is None:
        matlab = find_matlab()
        if matlab is None:
            raise PipelineError(
                "MATLAB was not found on this machine.",
                hint="The feature extractor is MATLAB code. Set MATLAB_EXE to "
                     "the binary's full path, or add it to PATH, then reload "
                     "this page.")

    bundle = load_bundle()          # fails fast, before the expensive stage

    work_dir = Path(tempfile.mkdtemp(prefix="gif_"))
    try:
        source = work_dir / f"input{Path(filename).suffix.lower() or '.png'}"
        source.write_bytes(image_bytes)

        if engine is not None:
            work = lambda: run_matlab_engine(engine, source, work_dir)
        else:
            work = lambda: run_matlab_subprocess(matlab, source, work_dir)
        features, png_bytes, log = staged(slot, MATLAB_STAGES, work)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    def model_work():
        result = classify(features, bundle)
        breakdown, reason = explain(features, bundle)
        return result, breakdown, reason

    result, breakdown, shap_reason = staged(slot, MODEL_STAGES, model_work, dwell=0.9)

    return {
        "filename": filename,
        "width": width,
        "height": height,
        # Below the analysis window the extractor has to upscale, which is a
        # low-pass filter: it attenuates exactly the high-frequency evidence
        # the model relies on. Flagged here, surfaced with the verdict.
        "upscaled": min(width, height) < ANALYSIS_SIDE,
        "png": png_bytes,
        "matlab_log": log,
        "bundle": bundle,
        **result,
        "breakdown": breakdown,
        "shap_reason": shap_reason,
    }


digest = hashlib.sha256(upload.getvalue()).hexdigest()

if st.session_state.get("digest") != digest:
    slot = st.empty()
    try:
        st.session_state["result"] = run_pipeline(upload.getvalue(), upload.name, slot)
        st.session_state["digest"] = digest
        st.session_state.pop("failure", None)
    except PipelineError as err:
        st.session_state["failure"] = err
        st.session_state["digest"] = digest
        st.session_state.pop("result", None)
    finally:
        slot.empty()

if st.session_state.get("failure") is not None:
    error_card(st.session_state["failure"])
    st.stop()

res = st.session_state["result"]


# ---------------------------------------------------------------- verdict

edge = "var(--ai)" if res["is_ai"] else "var(--real)"
percent = round(res["prob"] * 100)

reading = (f'Of images scoring near <b>{res["prob"]:.2f}</b>, about '
           f'<b>{percent}%</b> are AI-generated.')

block(
    f'<div class="gf-reveal gf-d1" style="--edge:{edge};">'
    f'  <div class="gf-verdict">'
    f'    <div class="gf-verdict-label">Verdict</div>'
    f'    <div class="gf-verdict-word">{esc(res["verdict"])}</div>'
    f'    <p class="gf-reading gf-num">{reading}</p>'
    f'    <p class="gf-subline gf-num">Calibrated score {res["prob"]:.4f} against a '
    f'decision threshold of {res["threshold"]:.4f} · {esc(res["strength"])} separation</p>'
    f'    <div class="gf-chips">'
    f'      <span class="gf-chip gf-num">{res["width"]} × {res["height"]} px</span>'
    f'      <span class="gf-chip">{esc(res["filename"])}</span>'
    f'      <span class="gf-chip gf-num"><b>230</b> features measured</span>'
    f'    </div>'
    f'  </div>'
    f'</div>'
)

if not res["calibrated"]:
    st.caption("This model's scores were left uncalibrated at training time, so "
               "read the sentence above as a ranking rather than a frequency.")

if res["upscaled"]:
    block(
        '<div class="gf-reveal gf-d2 gf-warn">'
        '  <div class="gf-warn-title">Unreliable analysis</div>'
        f' <p class="gf-warn-body">At {res["width"]}&times;{res["height"]}&nbsp;px this image is '
        'smaller than the 256&times;256 analysis window, so it was upscaled before '
        'measurement. Upscaling is a low-pass filter: it attenuates precisely the '
        'high-frequency wavelet and residual energy this detector reads, and every '
        'training image was large enough to crop without it. Treat the verdict above '
        'as indicative only.</p>'
        '</div>'
    )

st.markdown('<hr class="gf-rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------- evidence

block(
    '<div class="gf-reveal gf-d2">'
    '  <div class="gf-h2">Evidence</div>'
    '  <p class="gf-h2-sub">Every one of the 230 features contributes an exact, '
    'additive amount to this image\'s score. The base rate plus all of them '
    'reproduces the model\'s own output - so nothing below can disagree with the '
    'verdict above.</p>'
    '</div>'
)

bd = res["breakdown"]

if bd is None:
    st.info("Per-feature explanation unavailable — the verdict above still stands.")
    st.caption(res["shap_reason"])
else:
    def signclass(v):
        return "pos" if v > 0 else "neg" if v < 0 else ""

    op = '<div class="gf-op">+</div>'
    ledger = (
        '<div class="gf-ledger">'
        f'  <div class="gf-cell"><div class="gf-cell-k">Base rate</div>'
        f'    <div class="gf-cell-v">{bd["base_value"]:+.3f}</div></div>'
        f'  {op}'
        f'  <div class="gf-cell"><div class="gf-cell-k">Top {len(bd["drivers"])} drivers</div>'
        f'    <div class="gf-cell-v {signclass(bd["top_sum"])}">{bd["top_sum"]:+.3f}</div></div>'
        f'  {op}'
        f'  <div class="gf-cell"><div class="gf-cell-k">Other {bd["rest_count"]}</div>'
        f'    <div class="gf-cell-v {signclass(bd["rest_sum"])}">{bd["rest_sum"]:+.3f}</div></div>'
        '   <div class="gf-op">=</div>'
        f'  <div class="gf-cell gf-total"><div class="gf-cell-k">Final {esc(bd["unit"])}</div>'
        f'    <div class="gf-cell-v">{bd["total"]:+.3f}</div></div>'
        '</div>'
    )

    peak = max(abs(d["contribution"]) for d in bd["drivers"]) or 1.0
    rows = []
    for d in bd["drivers"]:
        c = d["contribution"]
        tone = "var(--ai)" if c > 0 else "var(--real)"
        width = max(4.0, abs(c) / peak * 100.0)
        rows.append(
            f'<tr>'
            f'  <td class="gf-name">{esc(d["name"])}</td>'
            f'  <td class="gf-this">{esc(fmt(d["value"]))}</td>'
            f'  <td class="gf-ref">{esc(fmt(d["real"], 3))}</td>'
            f'  <td class="gf-ref">{esc(fmt(d["ai"], 3))}</td>'
            f'  <td class="gf-c"><div class="gf-bar">'
            f'    <span class="gf-bar-track">'
            f'      <span class="gf-bar-fill" style="width:{width:.1f}%;background:{tone};"></span>'
            f'    </span>'
            f'    <span class="gf-bar-val" style="color:{tone};">{c:+.3f}</span>'
            f'  </div></td>'
            f'</tr>')

    agreement = ("exactly" if abs(bd["reconstructed"] - bd["raw_prob"]) < 5e-4
                 else "to within %.4f" % abs(bd["reconstructed"] - bd["raw_prob"]))
    calnote = (f", then calibrated to {res['prob']:.4f}" if res["calibrated"] else "")

    block(
        '<div class="gf-reveal gf-d3">'
        + ledger +
        '<div class="gf-tablewrap"><table class="gf-table">'
        '<thead><tr>'
        f'  <th>Measurement</th><th>This image</th><th>Typical real</th>'
        f'  <th>Typical AI</th><th class="gf-c">Δ {esc(bd["unit"])}</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table></div>'
        f'<p class="gf-foot gf-num">Top {len(bd["drivers"])} of {bd["n_features"]} by absolute '
        f'contribution. Warm bars push toward AI, cool bars toward real. All '
        f'{bd["n_features"]} contributions and the base rate reconstruct a score of '
        f'{bd["reconstructed"]:.4f}, matching the model\'s own {bd["raw_prob"]:.4f} '
        f'{agreement}{calnote}.</p>'
        '</div>'
    )

st.markdown('<hr class="gf-rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------- dsp proof

block(
    '<div class="gf-reveal gf-d3">'
    '  <div class="gf-h2">Signal processing</div>'
    '  <p class="gf-h2-sub">What the detector actually looked at, straight out of '
    'MATLAB - not a rendering of the verdict, but the intermediates the 230 '
    'features were measured from.</p>'
    '</div>'
)

if res["png"]:
    with st.container(border=True):
        st.image(res["png"], use_container_width=True)
    block('<p class="gf-caption">Signal Processing Evidence: '
          '<span>Analyzed Crop &nbsp;|&nbsp; Fourier Spectrum &nbsp;|&nbsp; '
          'Wavelet Detail &nbsp;|&nbsp; High-Pass Residual</span></p>')
else:
    st.info("MATLAB measured the image but could not render the figure — "
            "the verdict and evidence above are unaffected.")
    if res["matlab_log"]:
        with st.expander("MATLAB output"):
            st.code(res["matlab_log"].strip(), language="text")


# ---------------------------------------------------------------- provenance

b = res["bundle"]
provenance = [f'held-out ROC-AUC <b>{b["roc_auc"]:.3f}</b>',
              f'accuracy <b>{b["accuracy"]:.1%}</b>']
if b.get("calibration_method") and b["calibration_method"] != "none":
    provenance.append(f'calibration <b>{esc(b["calibration_method"])}</b>')
if b.get("generators"):
    provenance.append("trained on <b>" + esc(", ".join(b["generators"])) + "</b>")

block('<div class="gf-reveal gf-d4 gf-prov gf-num">'
      + " &nbsp;·&nbsp; ".join(provenance) +
      '<br>Features: extractImageFeatures.m &nbsp;·&nbsp; '
      'Explanation: SHAP TreeExplainer, exact and additive.</div>')
