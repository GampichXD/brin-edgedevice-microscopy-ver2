"""
Tile Stitching Pipeline -- SIFT + BFMatcher Variant
=====================================================
Feature extraction             : SIFT  (OpenCV)
Feature matching               : BFMatcher  (brute-force, Lowe's ratio test)
Translation estimation         : 2-DOF median of match displacements, with a
                                  stage-prior pre-filter and a nominal-shift
                                  fallback for pairs that fail registration
                                  entirely (see correction.txt) -- not RANSAC
                                  affine, since this rig has no rotation/scale.
Reprojection error             : calculate_reprojection_error (unchanged)
NCC evaluation                 : calculate_overlap_ncc (unchanged)
Overlap quality metrics        : PSNR, SSIM, RMSE, NCC (unchanged)
Feather blending               : blend_panorama (unchanged)
ROI crop                       : find_roi / extract_roi (unchanged)

Why SIFT + BFMatcher over SIFT/SuperPoint + LightGlue?
  - No learned matcher / GPU tensor stack required -- pure OpenCV, so it
    is the lightest-weight baseline of the three pipeline variants.
  - Lowe's ratio test on brute-force nearest neighbours is the classical
    reference point the LightGlue variants are benchmarked against.

Installation:  pip install -r requirements_BFMatcher.txt
Full guide:    see INSTALL.md in this directory
"""

import re
import csv
import traceback
import warnings
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

warnings.filterwarnings('ignore')

import os
import time
try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ============================================================
# PERFORMANCE & MEMORY TRACKER HELPER
# ============================================================
class PerformanceTracker:
    """
    Tracks runtime duration and memory consumption (RAM & GPU VRAM).
    """
    def __init__(self, name="Pipeline Benchmark"):
        self.name = name
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_times = {}

        # Reset GPU peak stats if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def record_step(self, step_name):
        now = time.time()
        elapsed = now - self.last_step_time
        self.step_times[step_name] = elapsed
        self.last_step_time = now

    def _get_ram_usage(self):
        current_ram_mb = 0.0
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                current_ram_mb = process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        if resource is not None:
            peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        else:
            if HAS_PSUTIL:
                try:
                    process = psutil.Process(os.getpid())
                    peak_ram_mb = getattr(process.memory_info(), 'peak_wset', 0.0) / (1024 * 1024)
                except Exception:
                    peak_ram_mb = 0.0
            else:
                peak_ram_mb = 0.0
        return current_ram_mb, peak_ram_mb

    def _get_gpu_usage(self):
        current_gpu_mb = 0.0
        peak_gpu_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                current_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return current_gpu_mb, peak_gpu_mb

    def print_summary(self):
        total_time = time.time() - self.start_time
        curr_ram, peak_ram = self._get_ram_usage()
        curr_gpu, peak_gpu = self._get_gpu_usage()

        print("\n" + "=" * 65)
        print(f"  PERFORMANCE & MEMORY BENCHMARK: {self.name}")
        print("=" * 65)
        print(f"  Total Execution Time : {total_time:.3f} seconds ({total_time / 60:.2f} min)")
        print("-" * 65)
        print("  Hardware Acceleration:")
        print(f"    - SIFT Detector Device  : {DETECTOR_ACCEL}")
        print(f"    - BFMatcher Device      : {MATCHER_ACCEL}")
        print("-" * 65)
        print("  Execution Time Breakdown:")
        for step, duration in self.step_times.items():
            pct = (duration / total_time * 100) if total_time > 0 else 0.0
            print(f"    - {step:<38}: {duration:8.3f} s ({pct:5.1f}%)")
        print("-" * 65)
        print("  Memory Consumption:")
        if curr_ram > 0:
            print(f"    - Current RAM (RSS)   : {curr_ram:.2f} MB")
        print(f"    - Peak RAM (RSS)      : {peak_ram:.2f} MB ({peak_ram / 1024:.2f} GB)")
        if peak_gpu > 0 or curr_gpu > 0:
            print(f"    - Current GPU VRAM    : {curr_gpu:.2f} MB")
            print(f"    - Peak GPU VRAM       : {peak_gpu:.2f} MB ({peak_gpu / 1024:.2f} GB)")
        else:
            print("    - GPU VRAM            : N/A (Running on CPU)")
        print("=" * 65 + "\n")


# Track acceleration methods used at runtime
DETECTOR_ACCEL = "CPU"
MATCHER_ACCEL = "CPU"


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # --- General ---
    'resize_factor':        0.5,
    'overlap_percentage':   0.4,
    'canvas_padding':       50,
    'registration_channel': 'b',
                                  # which single channel the shift search runs on
                                  # (blending/output always stays full RGB):
                                  #   'gray' -- 0.299R + 0.587G + 0.114B (luma)
                                  #   'r' / 'g' / 'b' -- one raw colour channel
                                  #   'auto' -- per tile, pick the channel with
                                  #             the largest post-highpass detail
                                  #    banding at tile pitch, which is why
    # the solve is per channel.         (std of x - GaussianBlur(x))
                                  # Use a raw channel when the tiles have a
                                  # strong colour cast that buries the texture

    # --- Dish / empty-tile masking ---
    # A circular dish inside a rectangular grid leaves tiles outside it as
    # pure background -- worse than useless for registration, since they are
    # exactly the low-texture pairs that fail matching and fragment the
    # transformation graph. Detect and drop them in load_image(), before
    # calculate_overlap() ever forms a pair with them.
    'skip_empty_tiles':    False,  # off by default: the threshold below needs
                                   # calibrating per dataset/exposure before
                                   # it's safe to trust (see empty_tile_min_std).
    'empty_tile_min_std':  8.0,    # a tile is dropped if its high-pass std is
                                   # <= this. Calibrate against one tile you
                                   # KNOW is empty background before enabling
                                   # skip_empty_tiles on a new dataset -- the
                                   # floor depends on exposure/gain, not just
                                   # content. [DISH-MASK] in the load_image
                                   # log lists every tile it dropped.
    'empty_tile_sigma':    12.0,   # Gaussian sigma (px) for the high-pass used
                                   # by the empty-tile check (same technique as
                                   # the 'highpass' registration preprocessing).

    # --- Downsampled preview ---
    # A full-resolution mosaic (hundreds of MP on a large grid) is not
    # something anyone views at 100% anyway -- for a human sanity check or a
    # thesis figure, a small preview costs almost nothing next to the full
    # blend that already ran. Purely additive: writes one extra small file
    # alongside the existing full-resolution result, changes nothing else.
    'save_preview':   False,   # write a downsampled copy next to the full-res result
    'preview_scale':  0.125,  # e.g. a 362 MP mosaic -> ~5.7 MP preview

    # --- Lazy tile loading ---
    # OFF by default -- this is a straight toggle, not a replacement: with it
    # False, behaviour is byte-for-byte the same as before this option
    # existed. Turn it on only when tile-RAM (image_data holding all N tiles)
    # is the thing that doesn't fit, e.g. a large grid on a memory-limited
    # device. Trades that RAM for re-decoding a tile from disk on every
    # access instead of once -- worth it only when storage is fast (per
    # correction.txt). See _LazyTile.
    'lazy_tile_loading': False,

    # --- Photometric correction (see STEP 0.5) ---
    # Ported from brute_force.py after A/B testing on real captures
    # (whole_petri_fr, 20x34) established this combination as the best
    # result: flat-field alone left a measurable halo, +gain compensation
    # reduced tile-pitch colour banding further (row-banding power roughly
    # halved). Both default ON here -- unlike brute_force.py, where they
    # default off to keep that file an A/B harness, this script has no
    # "unmodified baseline" to stay comparable to.
    'flatfield_correction': True,  # divide out smooth vignetting (median-stack
                                   # per-pixel estimate across sampled tiles)
    'flatfield_samples':    60,
    'flatfield_sigma':      25.0,   # heavy blur -- keeps ONLY low frequencies.

    # --- Per-tile gain compensation (STEP 5.6) ---
    # Flat-field is ONE spatial profile shared by every tile, so it cannot
    # fix a PER-TILE scalar (auto-exposure/auto-white-balance drift over a
    # long scan). Measured on whole_petri_fr after flat-field: R/G ratio
    # drifted 37.6% relative std across tiles vs 3.8% for luminance -- i.e.
    # the residual halo reads as COLOUR banding at tile pitch, which is why
    # the solve is per channel.
    'gain_compensation':    True,
    'gain_per_channel':     True,   # 3 independent solves. A single luminance
                                    # gain leaves the dominant R/G term untouched.
    'gain_reg_lambda':      0.3,    # pull every log-gain toward 0. NOT optional:
                                    # unregularised the solve "explains" real
                                    # scene differences as exposure and produced
                                    # gains spanning 15x on whole_petri_fr, which
                                    # wrecks the image. At 0.3 the fit is barely
                                    # worse with gains held to a sane range.
    'gain_max_logratio':    0.35,   # reject a pair whose two strips differ by
                                    # more than this in log space (~1.42x) --
                                    # that is content disagreement or
                                    # misregistration, not exposure.
    'gain_min_median':      15.0,   # skip a strip this dark; a ratio between two
                                    # near-black strips is noise.

    # --- Overlap-driven shading correction (STEP 5.7) ---
    # Solves ONE shading field shared by all tiles from the tiles' own
    # overlap disagreement -- see solve_overlap_shading()'s docstring for the
    # model and why it is a no-op on tiles cut from a single source image
    # (unlike flatfield_correction, which always applies something).
    # Complementary to gain_compensation, not a replacement: this handles the
    # WITHIN-tile spatial term, gain handles the per-tile scalar term.
    'overlap_shading':       False,   # OFF by default like the other photometric toggles
    'shading_grid':          (12, 20),# (rows, cols) control points. Shading is
                                      # low-frequency, so coarse is correct --
                                      # a fine grid just fits noise.
    'shading_sample_step':   6,       # sample every Nth pixel of each overlap
    'shading_smooth_lambda': 4.0,     # neighbouring control points should agree
    'shading_zero_lambda':   0.05,    # pull log S toward 0. This is what makes
                                      # "no evidence" produce "no correction",
                                      # and makes the normal matrix PD.
    'shading_max_logratio':  0.35,    # reject a pixel pair differing by >~1.42x
                                      # -- content mismatch / misregistration
    'shading_min_value':     12.0,    # skip near-black pixels; their ratio is noise
    'shading_max_logamp':    0.5,     # hard clamp on the solved field (~1.65x)
    'x_axis_descending':    True,  # see load_image()'s comment on unique_x --
                                    # flip to False for grid-index filenames
                                    # (tile_r_c / embedded _r_c_) if horizontal
                                    # registration fails near-totally while
                                    # vertical partially works.
    'display_feather_distance':  30,  # soft feather for human viewing
    'analysis_feather_distance': 0,   # hard seam for anaw-gainlysis/training copy
    'feather_distance':     30,       # legacy alias -- used by blend_panorama
    'normalization_method': 'skimage_histogram_match',
    'min_overlap_area':     500,

    # --- SIFT (feature detector) ---
    'feature_method':      'sift',    # Focused only on SIFT
    'sift_max_keypoints':  1024,      # Maximum keypoints per ROI crop (reduced for edge devices)
    'sift_peak_threshold': 0.01,      # DoG peak threshold; lower = more keypoints
    'sift_edge_threshold': 10,        # Harris edge threshold

    # --- BFMatcher ---
    'lowe_ratio': 0.75,

    # --- Translation estimation (see calculate_homography below) ---
    # This rig only translates the stage in X/Y -- no rotation or scale --
    # so registration is a 2-DOF median-of-displacements estimate, not a
    # 4-DOF RANSAC affine fit. See correction.txt: RANSAC's minimal sample
    # for a 4-DOF model is 2 points, so a "successful" 2-inlier fit is
    # unverified noise, not a real estimate.
    'reproj_thresh':    4.0,   # px -- inlier tolerance around the median displacement
    'min_inlier_ratio': 0.15,   # now enforced: pairs below this ratio are rejected
    'min_inlier_count': 10,    # absolute floor -- a ratio is meaningless when the
                                # denominator is ~13; catches the 2-inlier/15 case.
    'stage_prior_window': 50.0,  # px: matches whose displacement is farther than
                                # this from the geometrically-expected shift (derived
                                # from the pair's own overlap region) are excluded from
                                # the median seed -- discards matches that are
                                # geometrically impossible before they can bias the
                                # estimate, which Lowe's ratio test alone won't catch
                                # on repetitive/self-similar microscopy texture.
    'use_nominal_fallback': True,  # if a geometrically-expected adjacent pair fails
                                # registration entirely, substitute a nominal shift
                                # derived from its overlap-region geometry instead of
                                # dropping the edge -- keeps the transformation graph
                                # connected instead of fragmenting into a handful of
                                # reachable tiles.

    # --- Canvas safety ---
    'max_canvas_megapixels': 100,  # calculate_optimal_canvas() raises instead of
                                    # allocating past this -- a real N-tile grid never
                                    # gets close; hitting it means a bad accumulated
                                    # transform, not a genuinely huge mosaic.

    # --- Global tile-position adjustment (see solve_global_translations below) ---
    'use_global_adjustment': True,  # position every tile with ONE least-squares
                                    # solve over all pairwise translation
                                    # constraints instead of chaining transforms
                                    # along a BFS path from the reference tile.
                                    # Chaining accumulates per-pair error with
                                    # distance from the reference (staircase
                                    # drift at the mosaic corners on large
                                    # grids); the global solve spreads it
                                    # evenly. False -> old calculate_all_transforms.
    'grid_adjust_outlier_tol': 'adaptive', # before the solve, drop any pair whose
                                    # translation deviates from its row/column
                                    # median by more than this (fixed XY stage
                                    # => all horizontal pairs share a dx, all
                                    # vertical pairs a dy). Guards the L2 solve
                                    # against a bad pair smearing error grid-wide.
                                    # 'adaptive' -> derive the tolerance from the
                                    # measured per-direction spread itself (MAD)
                                    # instead of a fixed guess -- a fixed px value
                                    # increasingly mis-rejects as grid extent (and
                                    # the true spread) changes with grid size (see
                                    # correction.txt). A number pins it; None skips
                                    # the pre-filter entirely.
    'nominal_fallback_weight': 0.1, # weight (vs. 1.0 for a measured edge) given to
                                    # nominal-fallback edges (see
                                    # add_nominal_fallback_edges) in the
                                    # least-squares solve -- keeps the graph
                                    # connected without letting an assumed shift
                                    # pull real tiles' positions as hard as an
                                    # actual measurement would.

    # --- Device ---
    'use_gpu': False,                # Enable CUDA BFMatcher acceleration if available
    'debug': False,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': True,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
    'enable_benchmark': True,       # Toggle Performance & Memory Tracker benchmarking
    'verbose_pair_metrics': False,  # Print PSNR/SSIM/NCC/RepError per pair (off = final summary only)
}


# ============================================================
# STEP 1 : Image Loading & Coordinate Extraction
# ============================================================

def _tile_is_empty(gray, min_std=8.0, sigma=12.0):
    """
    True if *gray* looks like featureless background (e.g. outside a
    circular dish on a large grid) rather than real tissue/colony content.

    Same signal as the registration preprocessing's 'highpass' step: subtract
    a heavily blurred copy and look at what's left. A tile with real content
    has texture at this scale; empty background is flat, so its high-pass std
    stays near the sensor noise floor. CALIBRATE min_std against one tile you
    know is empty before relying on this on a new dataset -- the floor
    depends on exposure/gain, not just content. Empty tiles are worse than
    useless for registration: they are exactly the low-texture pairs that
    fail matching and fragment the transformation graph, so drop them before
    calculate_overlap() ever sees them (see CONFIG['skip_empty_tiles']).
    """
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    return float(hp.std()) <= min_std


def _registration_gray(img_rgb, channel):
    """
    Single-channel uint8 image the shift search runs on. The colour image in
    'image' is untouched -- blending and all output stay full RGB.

    'gray' is luma (0.299R + 0.587G + 0.114B); 'r'/'g'/'b' take one raw
    channel; 'auto' keeps whichever channel carries the most high-frequency
    detail (std of x - GaussianBlur(x)) -- useful when a strong colour cast
    buries the texture in luma, as in the blue-cast 10x10_1 tiles.
    """
    channel = (channel or 'gray').lower()
    if channel == 'gray':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    idx = {'r': 0, 'g': 1, 'b': 2}
    if channel in idx:
        return np.ascontiguousarray(img_rgb[:, :, idx[channel]])
    if channel == 'auto':
        best_ch, best_detail = None, -1.0
        for c in range(3):
            x = img_rgb[:, :, c].astype(np.float32)
            detail = float((x - cv2.GaussianBlur(x, (0, 0), sigmaX=3, sigmaY=3)).std())
            if detail > best_detail:
                best_ch, best_detail = c, detail
        return np.ascontiguousarray(img_rgb[:, :, best_ch])
    raise ValueError(f"Unknown registration_channel: '{channel}'")


# ============================================================
# STEP 0.5 : Photometric correction  (flat-field)
# ============================================================
# Smooth VIGNETTING -- a low-frequency radial/gradient falloff -- measured
# on whole_petri_fr. Clearly visible on background tiles (e.g. r0_c0,
# outside the dish: a clean gradient, no texture). Correct by dividing
# out a HEAVILY blurred (sigma ~51) per-pixel median across many tiles:
# real content varies tile to tile and washes out of the median, the
# illumination profile does not.


def estimate_flatfield(file_paths, resize_factor=1.0, samples=60,
                       sigma=51.0, verbose=True):
    """
    Per-pixel median across a random sample of tiles, heavily blurred.

    Background dominates each tile, so the median approximates the
    illumination profile; the blur then strips whatever real content
    leaked through, leaving only the low-frequency shading. Normalised to
    mean 1.0 so dividing by it preserves overall brightness.
    """
    import random as _random
    paths = list(file_paths)
    if not paths:
        return None
    picks = _random.sample(paths, min(int(samples), len(paths)))
    stack = []
    for p in picks:
        raw = cv2.imread(str(p))
        if raw is None:
            continue
        if resize_factor != 1.0:
            h, w = raw.shape[:2]
            raw = cv2.resize(raw, (int(w * resize_factor), int(h * resize_factor)))
        stack.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32))
    if not stack:
        return None
    # Tiles are not guaranteed to be pixel-identical in size -- measured
    # 20x (92,91) and 5x (92,92) in Euglena_tiles/5x5_ecoli, which makes a
    # naive np.stack raise. Crop every sample to the common minimum before
    # stacking; _photometric_correct() then resizes the result to whatever
    # each individual tile actually is.
    mh = min(a.shape[0] for a in stack)
    mw = min(a.shape[1] for a in stack)
    stack = [a[:mh, :mw] for a in stack]
    ff = np.median(np.stack(stack), axis=0)
    ff = cv2.GaussianBlur(ff, (0, 0), float(sigma))

    # Normalise EACH CHANNEL by its OWN mean, not by one scalar mean across
    # all three. These tiles are heavily colour-cast (measured B~170, G~128,
    # R~20), so a global mean (~106) makes the red plane of ff ~0.19, and
    # dividing red by that boosts it ~5x -- a violent red cast, not a
    # correction. Per-channel normalisation removes only the SPATIAL
    # variation and leaves the colour balance alone, which is the whole
    # point of a flat field.
    for c in range(ff.shape[2]):
        m = float(ff[:, :, c].mean())
        ff[:, :, c] /= max(m, 1e-6)

    if verbose:
        print(f"[FLATFIELD] estimated from {len(stack)} tiles "
              f"(blur sigma={sigma}); per-channel range "
              f"{ff.min():.3f}..{ff.max():.3f}")
    return ff.astype(np.float32)


# Module-level state so both the eager load path and _LazyTile._decode()
# apply exactly the same correction (a lazy re-decode must not silently
# skip it -- that would make lazy and eager runs disagree).
_FLATFIELD = None    # HxWx3 float32, mean-normalised, or None
_TILE_GAINS = None   # {coord: np.ndarray(3,) float32} once solved, or None
_SHADING_CORR = None # np.ndarray (h, w, 3) float32 multiplicative correction, or None


def _photometric_correct(img_rgb):
    """Apply the flat-field correction if it is enabled+prepared."""
    out = img_rgb
    if _FLATFIELD is not None:
        ff = _FLATFIELD
        if ff.shape[:2] != out.shape[:2]:
            # the field is smooth by construction, so resampling it to this
            # tile's exact size is lossless in practice
            ff = cv2.resize(ff, (out.shape[1], out.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        out = np.round(np.clip(out.astype(np.float32) / ff, 0, 255)).astype(np.uint8)
    return out


def prepare_photometric_correction(file_paths, resize_factor=1.0):
    """Build the flat-field once, before the load loop."""
    global _FLATFIELD
    _FLATFIELD = None
    paths = list(file_paths)
    if not paths:
        return

    if CONFIG.get('flatfield_correction', False):
        _FLATFIELD = estimate_flatfield(
            paths, resize_factor,
            samples=CONFIG.get('flatfield_samples', 60),
            sigma=CONFIG.get('flatfield_sigma', 51.0))


# STEP 5.6 : Per-tile gain compensation  (photometric)
# ============================================================

def solve_tile_gains(image_data, overlap_pairs, verbose=True):
    """
    One multiplicative gain per tile (per channel), solved over the whole
    overlap graph at once.

    Why this exists even though flat-field correction already runs: a flat
    field is ONE spatial profile shared by every tile. It removes the
    within-tile shading (measured: edge/centre 1.089 -> 1.018 on
    whole_petri_fr) but is mathematically incapable of fixing a per-tile
    scalar -- and that scalar is what is left of the halo. Measured on
    whole_petri_fr AFTER flat-field, interior row r8 c6..c18:

        G (luminance)  3.8% relative std across tiles
        B/G ratio      5.8%
        R/G ratio     37.6%

    Auto-exposure/auto-white-balance drift over a 680-tile scan. Since R/G
    drifts ~10x more than luminance, it shows up as colour banding at tile
    pitch -- hence per-channel by default.

    Model (Brown & Lowe style gain compensation, in log space so it stays a
    linear least-squares): every neighbour pair (i, j) sharing an overlap
    strip contributes

        log g_i - log g_j = log(median_j) - log(median_i)

    stacked into an over-determined system and solved with lstsq -- the same
    shape as solve_global_translations() does for positions, over the same
    pair graph.

    Two guards, both load-bearing. Without them the solve happily explains
    real scene differences as exposure: unregularised on whole_petri_fr it
    produced gains spanning 0.178..2.654 (a 15x range) which would wreck the
    image rather than fix it.
      * gain_reg_lambda   -- pulls every log-gain toward 0.
      * gain_max_logratio -- drops a pair whose strips disagree too much to
                             be exposure (content mismatch/misregistration).
      * gain_min_median   -- skips strips too dark for a ratio to mean
                             anything.
    Measured at lambda=0.3, cut=0.35: overlap mismatch down 51% (B) / 35%
    (G) / 49% (R), gains held to 0.72..1.46.

    Returns {coord: np.ndarray(3,) float32}, or None if nothing was solvable.
    """
    if not overlap_pairs:
        return None

    lam     = float(CONFIG.get('gain_reg_lambda', 0.3))
    max_lr  = float(CONFIG.get('gain_max_logratio', 0.35))
    min_med = float(CONFIG.get('gain_min_median', 15.0))
    per_ch  = bool(CONFIG.get('gain_per_channel', True))

    # --- measure each pair's two overlap strips once -------------------
    # Subsampled 2x on both axes before the median: a median over ~45k
    # samples is already stable well under the 1% we care about, and it
    # keeps this affordable at ~1300 pairs (and cheaper under lazy loading,
    # where each access re-decodes the tile).
    obs = []
    for pair in overlap_pairs:
        c1, c2 = pair['coord1'], pair['coord2']
        r1, r2 = pair['region1'], pair['region2']
        try:
            a = image_data[c1]['image'][r1['y']:r1['y'] + r1['height'],
                                        r1['x']:r1['x'] + r1['width']]
            b = image_data[c2]['image'][r2['y']:r2['y'] + r2['height'],
                                        r2['x']:r2['x'] + r2['width']]
        except Exception as e:
            print(f"[GAIN] skipped {c1}<->{c2}: {e}")
            continue
        if a.size == 0 or b.size == 0:
            continue
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h:2, :w:2].reshape(-1, 3).astype(np.float32)
        b = b[:h:2, :w:2].reshape(-1, 3).astype(np.float32)
        if a.size == 0 or b.size == 0:
            continue
        obs.append((c1, c2, np.median(a, axis=0), np.median(b, axis=0)))

    if not obs:
        print("[GAIN] no usable overlap strips -- gain compensation skipped")
        return None

    nodes = sorted({c for o in obs for c in (o[0], o[1])})
    idx   = {c: i for i, c in enumerate(nodes)}
    n     = len(nodes)
    gains = np.ones((n, 3), dtype=np.float64)

    print(f"\n[SUMMARY] GAIN COMPENSATION (lambda={lam}, "
          f"|log ratio| cut={max_lr}, per_channel={per_ch}):")

    n_ch = 3 if per_ch else 1
    for ch in range(n_ch):
        rows, rhs, wts = [], [], []
        kept = rej = 0
        for c1, c2, m1, m2 in obs:
            v1 = float(m1[ch]) if per_ch else float(m1.mean())
            v2 = float(m2[ch]) if per_ch else float(m2.mean())
            if v1 < min_med or v2 < min_med:
                rej += 1
                continue
            lr = float(np.log(v2) - np.log(v1))
            if abs(lr) > max_lr:
                rej += 1
                continue
            row = np.zeros(n)
            row[idx[c1]] = 1.0
            row[idx[c2]] = -1.0
            rows.append(row)
            rhs.append(lr)
            wts.append(1.0)
            kept += 1

        if kept == 0:
            print(f"   ch{ch}: no usable pairs (all rejected) -- gains left at 1.0")
            continue

        data_rows = np.asarray(rows)
        data_rhs  = np.asarray(rhs)

        # regularisation rows: lam * log g_i = 0 for every tile
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(0.0)
            wts.append(lam)

        A = np.asarray(rows) * np.asarray(wts)[:, None]
        b = np.asarray(rhs) * np.asarray(wts)
        lg, *_ = np.linalg.lstsq(A, b, rcond=None)

        before = float(np.abs(data_rhs).mean())
        after  = float(np.abs(data_rows @ lg - data_rhs).mean())
        g = np.exp(lg)
        gains[:, ch] = g
        pct = (1.0 - after / before) * 100.0 if before > 0 else 0.0
        print(f"   ch{ch}: pairs kept={kept} rejected={rej} | "
              f"|log ratio| {before:.4f} -> {after:.4f} ({pct:.0f}% better) | "
              f"gains {g.min():.3f}..{g.max():.3f}")

    if not per_ch:
        gains[:, 1] = gains[:, 0]
        gains[:, 2] = gains[:, 0]

    return {c: gains[idx[c]].astype(np.float32) for c in nodes}


# ============================================================
# STEP 5.7 : Overlap-Driven Shading Correction
# ============================================================
# Estimates the shading field S(u,v) from the tiles' own disagreement in
# their overlaps, instead of from a median stack of tile content
# (estimate_flatfield / CONFIG['flatfield_correction']).
#
# WHY THIS EXISTS -- measured on the real data:
#   * The median-stack flat field recovers a genuine, reproducible optical
#     profile on real captures: centre/corner came out 0.925 / 0.920 / 0.919
#     on 10x10_1 / whole_petri_fr / 3x3 -- same camera, same answer.
#   * On tiles CUT FROM ONE IMAGE (7x7, 10x10_euglena_red, 5x5_ecoli, ...)
#     the same estimator returns 1.000 / 1.015 / 0.953 -- no shared optical
#     profile exists, so np.median across tiles returns the median of the
#     CONTENT. Dividing by that injects a spurious, content-shaped
#     correction, which is why the photometric pipeline slightly degrades
#     those datasets while clearly helping real ones.
#   * A per-tile SCALAR gain (STEP 5.6) cannot fix the halo either: after
#     removing the scalar component from the averaged overlap ratio map, a
#     4.2% spatial residual remained with a systematic +7% left->right
#     gradient. That gradient IS the banding, and it is a within-tile
#     spatial term by construction.
#
# THE MODEL
#   Observed tile i:   O_i(u,v) = T(p) * S(u,v)
#   where T is the true scene at canvas point p and S is one shading field
#   shared by every tile (a property of the optics, not of the tile).
#   Taking logs, for a canvas point p seen by BOTH tile i and tile j:
#
#       log S(u_i,v_i) - log S(u_j,v_j) = log O_i(u_i,v_i) - log O_j(u_j,v_j)
#
#   T cancels -- that is the whole trick. Each overlapping pixel gives one
#   linear constraint on the DIFFERENCE of log S at two tile-local
#   positions, with no reference to scene content at all.
#
# WHY IT IS SAFE ON SYNTHETIC DATA
#   The right-hand side is measured disagreement. Tiles cut from one image
#   agree in their overlaps, so every RHS is ~0, and with the zero-pull
#   regulariser the solution is log S = 0 -- a flat field, a NO-OP. The
#   correction is applied only in proportion to evidence that one is needed,
#   unlike a median stack which always outputs something.
#
# SOLVING IT
#   log S is solved on a coarse control grid (shading is low-frequency, so
#   ~12x20 is the right basis) and upsampled. Constraints accumulate
#   straight into the normal equations A^T A / A^T b -- A itself would be
#   hundreds of thousands of rows and is never materialised. The system is a
#   few hundred unknowns and solves instantly.

def _shading_cell_index(u, v, h, w, gh, gw):
    """Map tile-local pixel coords to a flat coarse-grid control-point index."""
    gy = np.clip((v.astype(np.int64) * gh) // h, 0, gh - 1)
    gx = np.clip((u.astype(np.int64) * gw) // w, 0, gw - 1)
    return gy * gw + gx


def solve_overlap_shading(image_data, transforms, overlap_pairs, verbose=True):
    """
    Solve one shading field shared by all tiles, from overlap disagreement.

    Returns an (h, w, 3) float32 multiplicative correction (already 1/S and
    mean-normalised to 1), or None if there was not enough evidence.
    """
    gh, gw   = CONFIG.get('shading_grid', (12, 20))
    step     = int(CONFIG.get('shading_sample_step', 6))
    lam_s    = float(CONFIG.get('shading_smooth_lambda', 4.0))
    lam_0    = float(CONFIG.get('shading_zero_lambda', 0.05))
    max_lr   = float(CONFIG.get('shading_max_logratio', 0.35))
    min_v    = float(CONFIG.get('shading_min_value', 12.0))
    max_amp  = float(CONFIG.get('shading_max_logamp', 0.5))
    n_cells  = gh * gw

    usable = [p for p in overlap_pairs
              if p['coord1'] in transforms and p['coord2'] in transforms]
    if not usable:
        if verbose:
            print("[SHADING] no registered overlap pairs -- skipped")
        return None

    ref_h, ref_w = image_data[usable[0]['coord1']]['image'].shape[:2]

    AtA = [np.zeros((n_cells, n_cells), dtype=np.float64) for _ in range(3)]
    Atb = [np.zeros(n_cells, dtype=np.float64) for _ in range(3)]
    kept = [0, 0, 0]
    rejected = [0, 0, 0]
    before = [[], [], []]

    for pair in usable:
        c1, c2 = pair['coord1'], pair['coord2']
        im1 = image_data[c1]['image']
        im2 = image_data[c2]['image']
        if im1.shape[:2] != (ref_h, ref_w) or im2.shape[:2] != (ref_h, ref_w):
            continue                  # ragged edge tile -- skip, keeps indexing simple
        o1 = transforms[c1][:2, 2]
        o2 = transforms[c2][:2, 2]

        # overlapping rectangle of the two tiles, in the reference frame
        x0 = max(o1[0], o2[0]); x1 = min(o1[0] + ref_w, o2[0] + ref_w)
        y0 = max(o1[1], o2[1]); y1 = min(o1[1] + ref_h, o2[1] + ref_h)
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        xs = np.arange(int(np.ceil(x0)), int(np.floor(x1)), step)
        ys = np.arange(int(np.ceil(y0)), int(np.floor(y1)), step)
        if xs.size < 2 or ys.size < 2:
            continue
        X, Y = np.meshgrid(xs, ys)
        u1 = np.round(X - o1[0]).astype(np.int64); v1 = np.round(Y - o1[1]).astype(np.int64)
        u2 = np.round(X - o2[0]).astype(np.int64); v2 = np.round(Y - o2[1]).astype(np.int64)
        ok = ((u1 >= 0) & (u1 < ref_w) & (v1 >= 0) & (v1 < ref_h) &
              (u2 >= 0) & (u2 < ref_w) & (v2 >= 0) & (v2 < ref_h))
        if not ok.any():
            continue
        u1, v1, u2, v2 = u1[ok], v1[ok], u2[ok], v2[ok]

        ca_all = _shading_cell_index(u1, v1, ref_h, ref_w, gh, gw)
        cb_all = _shading_cell_index(u2, v2, ref_h, ref_w, gh, gw)

        for ch in range(3):
            a = im1[v1, u1, ch].astype(np.float64)
            b = im2[v2, u2, ch].astype(np.float64)
            good = (a > min_v) & (b > min_v)
            if not good.any():
                continue
            r = np.log(a[good]) - np.log(b[good])
            ca = ca_all[good]
            cb = cb_all[good]
            keep = np.abs(r) <= max_lr      # reject content / misregistration outliers
            rejected[ch] += int((~keep).sum())
            r = r[keep]
            ca = ca[keep]
            cb = cb[keep]
            if r.size == 0:
                continue
            kept[ch] += int(r.size)
            before[ch].append(np.abs(r))

            # accumulate (e_ca - e_cb)(e_ca - e_cb)^T  and  (e_ca - e_cb) * r
            M = AtA[ch]
            np.add.at(M, (ca, ca), 1.0)
            np.add.at(M, (cb, cb), 1.0)
            np.add.at(M, (ca, cb), -1.0)
            np.add.at(M, (cb, ca), -1.0)
            np.add.at(Atb[ch], ca, r)
            np.add.at(Atb[ch], cb, -r)

    if sum(kept) == 0:
        if verbose:
            print("[SHADING] no usable constraints -- skipped")
        return None

    # --- regularisation -------------------------------------------------
    # smoothness: neighbouring control points should agree (shading is
    # low-frequency). zero-pull: with no evidence the answer is log S = 0,
    # which is what makes this a no-op on data that does not need it, and
    # also makes the normal-equation matrix positive definite.
    Rs = np.zeros((n_cells, n_cells), dtype=np.float64)
    for gy in range(gh):
        for gx in range(gw):
            i = gy * gw + gx
            for dy, dx in ((0, 1), (1, 0)):
                ny, nx = gy + dy, gx + dx
                if ny >= gh or nx >= gw:
                    continue
                j = ny * gw + nx
                Rs[i, i] += 1.0
                Rs[j, j] += 1.0
                Rs[i, j] -= 1.0
                Rs[j, i] -= 1.0
    reg = (lam_s ** 2) * Rs + (lam_0 ** 2) * np.eye(n_cells)

    fields, stats = [], []
    for ch in range(3):
        x = np.linalg.solve(AtA[ch] + reg, Atb[ch])
        x -= x.mean()                                 # gauge: shading is relative
        x = np.clip(x, -max_amp, max_amp)
        logS = x.reshape(gh, gw).astype(np.float32)
        S = np.exp(cv2.resize(logS, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC))
        S /= max(float(S.mean()), 1e-6)               # preserve overall brightness
        fields.append((1.0 / S).astype(np.float32))   # correction = 1/S
        stats.append((float(np.abs(x).max()), float(S.min()), float(S.max())))

    corr = np.stack(fields, axis=-1)

    if verbose:
        print(f"\n[SUMMARY] OVERLAP-DRIVEN SHADING (grid={gh}x{gw}, "
              f"smooth={lam_s}, zero-pull={lam_0}):")
        for ch in range(3):
            b = np.concatenate(before[ch]) if before[ch] else np.zeros(1)
            amp, smin, smax = stats[ch]
            print(f"   ch{ch}: constraints kept={kept[ch]:7d} rejected={rejected[ch]:6d} | "
                  f"mean |log ratio| in overlaps={b.mean():.4f} | "
                  f"shading {smin:.3f}..{smax:.3f} (max |log S|={amp:.4f})")
        flat = max(s[0] for s in stats)
        if flat < 0.02:
            print(f"   -> solved field is essentially FLAT (max |log S| = {flat:.4f}); "
                  f"the tiles already agree in their overlaps, so this is a "
                  f"no-op. Expected on tiles cut from a single source image.")
    return corr


class _LazyTile:
    """
    Drop-in replacement for the eager {'image', 'image_gray', 'filename'}
    dict `load_image()` normally stores per tile. Supports the same
    `entry['image']` / `entry['image_gray']` / `entry['filename']` access
    every existing call site (blending, metrics, feature extraction,
    visualization) already uses -- nothing downstream needs to change -- but
    re-decodes the tile from disk on every access instead of holding pixels
    in RAM. Gated by CONFIG['lazy_tile_loading'] (off by default).

    Deliberately does NOT cache the decoded array: caching would let the
    working set creep back up toward "every tile touched so far stays in
    RAM," defeating the point on a large grid (e.g. hundreds of tiles) where
    the goal is O(1)-tile peak memory, not O(N). The tradeoff is re-decoding
    a tile once per access (a handful of times per tile across registration
    + blending) instead of once ever -- worth it only when RAM, not I/O, is
    the constraint (a memory-limited device with fast onboard storage, per
    correction.txt's "trades a little I/O for a lot of headroom").
    """
    __slots__ = ('path', 'filename', 'resize_factor', 'registration_channel')

    def __init__(self, path, filename, resize_factor, registration_channel):
        self.path = path
        self.filename = filename
        self.resize_factor = resize_factor
        self.registration_channel = registration_channel

    def _decode(self):
        raw = cv2.imread(str(self.path))
        if raw is None:
            raise FileNotFoundError(f"[LAZY-LOAD] Gagal memuat ulang: {self.path}")
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if self.resize_factor != 1.0:
            h, w = img_rgb.shape[:2]
            img_rgb = cv2.resize(img_rgb, (int(w * self.resize_factor), int(h * self.resize_factor)))
        # same photometric correction the eager path applies, so a lazy
        # re-decode can never disagree with an eager load
        return _photometric_correct(img_rgb)

    def __getitem__(self, key):
        if key == 'filename':
            return self.filename
        if key == 'image':
            return self._decode()
        if key == 'image_gray':
            return _registration_gray(self._decode(), self.registration_channel)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_image(folder_path, resize_factor=1.0, registration_channel=None):
    """
    Load all tile images from *folder_path*.

    Supported filename patterns
    ---------------------------
    * ``Focused_<x>_<y>.jpg``    -- real-stage coordinates
    * ``tile_r<row>_c<col>.jpg`` -- grid indices (col->x, row->y)
    * ``..._r<row>_c<col>_....jpg`` -- grid indices embedded anywhere in a
      longer filename carrying extra metadata (matched with `.search()`).
    * ``IMG_r<row>_c<col>.jpg``  -- the date-stripped short form
      (as in Euglena_tiles/E_Coli_5x5 after renaming).

    *registration_channel* (default CONFIG['registration_channel']) chooses
    which single channel goes into 'image_gray', the array the shift search
    consumes; 'image' always keeps the full RGB tile.

    Returns
    -------
    image_data : dict  {(x, y): {'image', 'image_gray', 'filename'}}
    grid_info  : dict  {'dimensions', 'x_range', 'y_range', 'unique_x', 'unique_y'}
    """
    if registration_channel is None:
        registration_channel = CONFIG.get('registration_channel', 'gray')
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' tidak ditemukan")

    focused_pattern  = re.compile(r'Focused_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.jpg')
    tile_pattern     = re.compile(r'tile_r(\d+)_c(\d+)\.jpg')
    embedded_pattern = re.compile(r'_r(\d+)_c(\d+)(?:_|\.)', re.IGNORECASE)

    # Build the flat-field ONCE from a sample of real tiles,
    # before any tile is stored (default on for this script; see CONFIG
    # comments -- no-op if a dataset does not need it, per
    # solve_overlap_shading()'s sibling reasoning in brute_force.py).
    def _parses_as_tile(p):
        name = p.name
        if focused_pattern.match(name):
            return True
        if tile_pattern.match(name):
            return True
        if embedded_pattern.search(name):
            return True
        return False
    prepare_photometric_correction(
        [p for p in folder_path.glob("*.jpg") if _parses_as_tile(p)], resize_factor)

    image_data = {}
    skipped_empty = []
    for file_path in folder_path.glob("*.jpg"):
        name = file_path.name
        coords = None

        m = focused_pattern.match(name)
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
        else:
            m = tile_pattern.match(name)
            if m:
                coords = (float(m.group(2)), float(m.group(1)))   # col -> x, row -> y
            else:
                m = embedded_pattern.search(name)
                if m:
                    coords = (float(m.group(2)), float(m.group(1)))   # col -> x, row -> y

        if coords is None:
            continue

        try:
            raw = cv2.imread(str(file_path))
            if raw is None:
                print(f"[WARN] Gagal memuat (None): {name}")
                continue
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            if resize_factor != 1.0:
                h, w = img_rgb.shape[:2]
                img_rgb = cv2.resize(img_rgb, (int(w * resize_factor), int(h * resize_factor)))
            img_rgb = _photometric_correct(img_rgb)
            if CONFIG.get('skip_empty_tiles', False):
                check_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                if _tile_is_empty(check_gray, CONFIG.get('empty_tile_min_std', 8.0),
                                  CONFIG.get('empty_tile_sigma', 12.0)):
                    skipped_empty.append((coords, name))
                    continue

            if CONFIG.get('lazy_tile_loading', False):
                # Discard the pixels just decoded above (only needed to
                # validate the file and run the empty-tile check) and store a
                # handle that reloads from disk on first real access instead.
                # Peak RAM here is O(1) tile during this loop, and O(1) tile
                # again whenever something later touches
                # image_data[coord]['image'/'image_gray'] -- never O(N tiles)
                # at once, unlike the eager dict below.
                image_data[coords] = _LazyTile(file_path, name, resize_factor,
                                               registration_channel)
            else:
                image_data[coords] = {
                    'image':      img_rgb,
                    'image_gray': _registration_gray(img_rgb, registration_channel),
                    'filename':   name,
                }
        except Exception as e:
            print(f"[ERROR] Gagal memuat {name}: {e}")
            traceback.print_exc()

    if skipped_empty:
        print(f"[DISH-MASK] Skipped {len(skipped_empty)} empty/background tile(s) "
              f"(high-pass std <= {CONFIG.get('empty_tile_min_std', 8.0)}, "
              f"sigma={CONFIG.get('empty_tile_sigma', 12.0)}): "
              f"{[c for c, _ in skipped_empty][:10]}"
              f"{' ...' if len(skipped_empty) > 10 else ''}")

    if not image_data:
        raise ValueError(
            "Tidak ada gambar yang ditemukan di folder."
            + (" Semua tile terdeteksi kosong -- periksa "
               "CONFIG['empty_tile_min_std'] (mungkin terlalu tinggi)."
               if skipped_empty else "")
        )

    x_coords = [c[0] for c in image_data]
    y_coords = [c[1] for c in image_data]
    # X sorts descending by default (unlike Y, which is ascending) --
    # historically tuned for the Focused_<x>_<y> real-stage-coordinate
    # convention, where the physical X axis can genuinely run opposite to
    # pixel/column-index direction on some rigs. For grid-index filenames
    # (tile_r_c / embedded _r_c_, where column count increases in the same
    # direction as pixels), this asymmetry makes calculate_overlap() pair
    # every horizontal neighbour with the wrong physical edge -- confirmed
    # on a real dataset where 0/504 horizontal pairs registered while
    # vertical (which has no such asymmetry) partially did. Flip
    # CONFIG['x_axis_descending'] to False for column-index datasets if
    # horizontal registration is failing near-totally.
    unique_x = sorted(set(x_coords), reverse=CONFIG.get('x_axis_descending', True))
    unique_y = sorted(set(y_coords))

    grid_info = {
        'dimensions': (len(unique_x), len(unique_y)),
        'x_range':    (min(x_coords), max(x_coords)),
        'y_range':    (min(y_coords), max(y_coords)),
        'unique_x':   unique_x,
        'unique_y':   unique_y,
    }
    print(f"{len(image_data)} gambar dimuat dari {folder_path}")
    print(f"Registrasi pada kanal : {registration_channel}")
    print(f"Dimensi Grid  : {grid_info['dimensions'][0]} x {grid_info['dimensions'][1]}")
    print(f"Rentang X     : {grid_info['x_range']}")
    print(f"Rentang Y     : {grid_info['y_range']}")
    return image_data, grid_info


def _normalise_axes(axes, rows, cols):
    """Guarantee axes is always a list-of-lists, regardless of grid shape."""
    if rows == 1 and cols == 1:
        return [[axes]]
    if rows == 1:
        return [list(axes)]
    if cols == 1:
        return [[ax] for ax in axes]
    return [list(row) for row in axes]


def visualize_grid_preview(image_data, grid_info):
    if not image_data or not grid_info:
        raise ValueError("Data gambar tidak valid")

    grid_w, grid_h = grid_info['dimensions']
    unique_x, unique_y = grid_info['unique_x'], grid_info['unique_y']

    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3 + 1))
    axes = _normalise_axes(axes, grid_h, grid_w)

    for row in axes:
        for ax in row:
            ax.axis('off')

    for coords, data in image_data.items():
        x_idx   = unique_x.index(coords[0])
        y_idx   = unique_y.index(coords[1])
        row_idx = grid_h - 1 - y_idx
        axes[row_idx][x_idx].imshow(data['image'])
        axes[row_idx][x_idx].set_title(f'({coords[0]},{coords[1]})', fontsize=16)

    plt.suptitle(f'Image Grid: {grid_w}x{grid_h} | {len(image_data)} images loaded', fontsize=20)
    plt.tight_layout()
    return fig


# ============================================================
# STEP 2 : Overlap Region Definition
# ============================================================

def calculate_overlap(image_data, grid_info, overlap_percentage=0.5):
    overlap_pairs = []
    unique_x, unique_y = grid_info['unique_x'], grid_info['unique_y']

    for coord, data in image_data.items():
        x, y = coord
        xi   = unique_x.index(x)
        yi   = unique_y.index(y)
        h, w = data['image'].shape[:2]

        # Right neighbour
        if xi < len(unique_x) - 1:
            right_coord = (unique_x[xi + 1], y)
            if right_coord in image_data:
                nb_h, nb_w = image_data[right_coord]['image'].shape[:2]
                ow = int(w * overlap_percentage)
                overlap_pairs.append({
                    'coord1':    coord,
                    'coord2':    right_coord,
                    'direction': 'horizontal',
                    'region1':   {'x': w - ow, 'y': 0, 'width': ow, 'height': h},
                    'region2':   {'x': 0, 'y': 0, 'width': int(nb_w * overlap_percentage), 'height': nb_h},
                })

        # Upper neighbour
        if yi < len(unique_y) - 1:
            upper_coord = (x, unique_y[yi + 1])
            if upper_coord in image_data:
                nb_h, nb_w = image_data[upper_coord]['image'].shape[:2]
                oh = int(h * overlap_percentage)
                overlap_pairs.append({
                    'coord1':    coord,
                    'coord2':    upper_coord,
                    'direction': 'vertical',
                    'region1':   {'x': 0, 'y': 0, 'width': w, 'height': oh},
                    'region2':   {'x': 0, 'y': nb_h - int(nb_h * overlap_percentage),
                                  'width': nb_w, 'height': int(nb_h * overlap_percentage)},
                })

    print(f"{len(overlap_pairs)} pasangan overlap ditemukan")
    return overlap_pairs


def visualize_overlap_regions(image_data, overlap_pairs, grid_info):
    grid_w, grid_h = grid_info['dimensions']
    unique_x, unique_y = grid_info['unique_x'], grid_info['unique_y']
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w*3, grid_h*3+1))
    axes = _normalise_axes(axes, grid_h, grid_w)
    for row in axes:
        for ax in row:
            ax.axis('off')

    coord_overlaps = defaultdict(list)
    for pair in overlap_pairs:
        c1, c2, d = pair['coord1'], pair['coord2'], pair['direction']
        coord_overlaps[c1].append({'region': pair['region1'], 'direction': d, 'role': 'source'})
        coord_overlaps[c2].append({'region': pair['region2'], 'direction': d, 'role': 'target'})

    for coords, data in image_data.items():
        xi      = unique_x.index(coords[0])
        yi      = unique_y.index(coords[1])
        row_idx = grid_h - 1 - yi
        ax      = axes[row_idx][xi]
        ax.imshow(data['image'])
        for ov in coord_overlaps.get(coords, []):
            r     = ov['region']
            color = 'lime' if ov['direction'] == 'horizontal' else 'red'
            ec    = ('darkgreen' if ov['role']=='source' else 'darkblue') \
                    if ov['direction']=='horizontal' \
                    else ('darkred' if ov['role']=='source' else 'darkorange')
            rect  = plt.Rectangle((r['x'], r['y']), r['width'], r['height'],
                                   lw=2, edgecolor=ec, facecolor=color, alpha=0.4)
            ax.add_patch(rect)
            lbl = ('H' if ov['direction']=='horizontal' else 'V') + \
                  ('S' if ov['role']=='source' else 'T')
            ax.text(r['x'] + r['width']//2, r['y'] + r['height']//2, lbl,
                    color='white', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        ax.set_title(f'{coords}', fontsize=16)

    plt.suptitle(f'Overlap Regions ({len(overlap_pairs)} pairs)', fontsize=20)
    plt.tight_layout()
    return fig


# ============================================================
# STEP 3 : Feature Extraction -- SIFT
# ============================================================

def select_descriptor(image, method='sift'):
    """
    Extract SIFT features via OpenCV. Always returns (keypoints, descriptors)
    as CPU types for keypoints and numpy arrays for descriptors. This exact
    cv2.SIFT_create() configuration is mirrored by SIFT_LG.py's
    _extract_region() so keypoint counts are directly comparable between
    the two pipeline variants.
    """
    global DETECTOR_ACCEL
    sift_max_kps = CONFIG.get('sift_max_keypoints', 1024)
    nfeatures = sift_max_kps if sift_max_kps != -1 else 0
    sift_peak_threshold = CONFIG['sift_peak_threshold']
    sift_edge_threshold = CONFIG['sift_edge_threshold']
    detector = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=sift_peak_threshold,
                               edgeThreshold=sift_edge_threshold, sigma=1.6)
    DETECTOR_ACCEL = "CPU"
    return detector.detectAndCompute(image, None)


def _adjust_keypoints(keypoints, dx, dy):
    return [
        cv2.KeyPoint(kp.pt[0] + dx, kp.pt[1] + dy,
                     kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ]


def extract_overlap_features(image_data, overlap_pairs, method='sift'):
    overlap_features = {}
    total_features   = 0

    for i, pair in enumerate(overlap_pairs, 1):
        coord1, coord2 = pair['coord1'], pair['coord2']
        r1, r2         = pair['region1'], pair['region2']
        direction      = pair['direction']

        try:
            gray1 = image_data[coord1]['image_gray']
            roi1  = gray1[r1['y']:r1['y'] + r1['height'], r1['x']:r1['x'] + r1['width']]
            # CLAHE on the ROI crop before detection, to raise keypoint density on
            # low-contrast agar.  Applied only to the crop, NOT to the stored
            # image_gray, so photometric metrics remain unaffected.
            #
            # KEEP THIS -- it is the measured best for a SIFT matcher, and it is
            # deliberately NOT what SP_LG.py does (that one feeds SuperPoint raw).
            # SIFT is already invariant to affine intensity change (gradient
            # orientations, L2-normalised descriptors), so it needs no photometric
            # normalisation; its failure mode here is keypoint STARVATION on flat
            # agar, which is exactly what CLAHE fixes. Measured on 40 real pairs of
            # 10x10 against the shipped brute_force run's shifts: CLAHE 71 px median
            # error / 13% of pairs within 10 px, versus raw 156 px / 3% and highpass
            # 110 px / 5%.  z-score was also tried and is a no-op (SIFT cancels it by
            # construction).  See PREPROCESSING_STUDY.md.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi1 = clahe.apply(roi1)
            kp1, feat1 = select_descriptor(roi1, method)
            kp1 = _adjust_keypoints(kp1, r1['x'], r1['y'])

            gray2 = image_data[coord2]['image_gray']
            roi2  = gray2[r2['y']:r2['y'] + r2['height'], r2['x']:r2['x'] + r2['width']]
            roi2 = clahe.apply(roi2)
            kp2, feat2 = select_descriptor(roi2, method)
            kp2 = _adjust_keypoints(kp2, r2['x'], r2['y'])

            overlap_features[(coord1, coord2)] = {
                'coord1': coord1, 'coord2': coord2, 'direction': direction,
                'keypoints1': kp1, 'keypoints2': kp2,
                'features1':  feat1, 'features2':  feat2,
                'region1':    r1, 'region2': r2,
            }
            n = len(kp1) + len(kp2)
            total_features += n
            print(f"Pair {i}/{len(overlap_pairs)}: {coord1} <-> {coord2} | Features: {n}")
        except Exception as e:
            print(f"[ERROR] Feature extraction failed {coord1} <-> {coord2}: {e}")
            traceback.print_exc()

    print(f"   Successful extractions : {len(overlap_features)}")
    print(f"   Total features         : {total_features}")
    avg = total_features / len(overlap_features) if overlap_features else 0
    print(f"   Avg features/pair      : {avg:.1f}")
    return overlap_features


# ============================================================
# STEP 4 : Feature Matching -- BFMatcher
# ============================================================

def keypoints_matching(feat1, feat2, method='sift', ratio=0.75):
    """
    Perform keypoint matching using SIFT descriptors.
    Uses CUDA-accelerated DescriptorMatcher if available,
    otherwise falls back to standard CPU BFMatcher.
    """
    global MATCHER_ACCEL
    use_gpu = CONFIG.get('use_gpu', True)

    # 1. Try CUDA BFMatcher (on Jetson / CUDA-enabled OpenCV builds)
    if use_gpu and 'cuda' in dir(cv2) and hasattr(cv2.cuda, 'DescriptorMatcher_createBFMatcher'):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
                gpu_feat1 = cv2.cuda_GpuMat()
                gpu_feat2 = cv2.cuda_GpuMat()
                gpu_feat1.upload(feat1)
                gpu_feat2.upload(feat2)
                raw = matcher.knnMatch(gpu_feat1, gpu_feat2, k=2)
                MATCHER_ACCEL = "GPU (CUDA BFMatcher)"
                return [m for m, n in raw if m.distance < n.distance * ratio]
        except Exception as e:
            print(f"[WARN] CUDA BFMatcher failed, falling back: {e}")

    # 2. CPU Fallback
    MATCHER_ACCEL = "CPU"
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(feat1, feat2, k=2)
    return [m for m, n in raw if m.distance < n.distance * ratio]


def match_overlap_features(overlap_features, method, ratio=0.75):
    match_result     = {}
    total_matches    = 0
    successful_pairs = 0

    for pair_key, fd in overlap_features.items():
        coord1, coord2 = pair_key
        feat1, feat2   = fd['features1'], fd['features2']

        if feat1 is None or feat2 is None:
            print(f"[SKIP] {pair_key}: missing features")
            continue

        try:
            matches = keypoints_matching(feat1, feat2, method=method, ratio=ratio)
            if not matches:
                print(f"[SKIP] {pair_key}: no matches after ratio test")
                continue

            min_feat = min(len(feat1), len(feat2))
            quality  = len(matches) / min_feat if min_feat > 0 else 0
            match_result[pair_key] = {
                'matches':      matches,
                'match_count':  len(matches),
                'direction':    fd['direction'],
                'quality':      quality,
                'keypoints1':   fd['keypoints1'],
                'keypoints2':   fd['keypoints2'],
                'region1':      fd['region1'],
                'region2':      fd['region2'],
            }
            total_matches    += len(matches)
            successful_pairs += 1
            print(f"{coord1} <-> {coord2} | Matches: {len(matches)} | Quality: {quality:.2f}")
        except Exception as e:
            print(f"[ERROR] Matching failed {pair_key}: {e}")
            traceback.print_exc()

    _print_matching_summary(match_result, len(overlap_features), successful_pairs, total_matches)
    return match_result


def _print_matching_summary(match_result, total_pairs, successful_pairs, total_matches):
    print(f"\n📊 OVERLAP MATCHING SUMMARY:")
    print(f"   Total pairs attempted  : {total_pairs}")
    print(f"   Successful matches     : {successful_pairs}")
    if successful_pairs:
        counts  = [r['match_count'] for r in match_result.values()]
        quality = [r['quality']     for r in match_result.values()]
        print(f"   Total matches found    : {sum(counts)}")
        print(f"   Avg matches/pair       : {np.mean(counts):.1f}")
        print(f"   Avg quality score      : {np.mean(quality):.2f}")


# ============================================================
# STEP 5 : Homography via RANSAC
# ============================================================

def calculate_homography(kp1, kp2, matches, reproj_thresh, region1=None, region2=None):
    """Estimate a pure 2-DOF translation from matched keypoints.

    This rig only translates the stage in X/Y -- there is no rotation or
    scale to estimate. estimateAffinePartial2D's 4-DOF model only needs 2
    point correspondences to solve exactly, so a RANSAC "success" backed by
    2-3 inliers is an unverified minimal sample, not a real estimate (see
    correction.txt). The median of per-match displacement vectors IS the
    robust estimator here: one correspondence determines the whole model,
    so no RANSAC sampling is needed, and there is no rotation/scale for
    outliers to be absorbed into.

    If *region1*/*region2* (the pair's own overlap regions from
    calculate_overlap()) are given, matches whose displacement is farther
    than CONFIG['stage_prior_window'] from the geometrically-expected shift
    are excluded from the median seed -- this discards matches that are
    geometrically impossible given the known tile layout, which Lowe's
    ratio test alone won't catch on repetitive/self-similar texture.
    """
    if len(matches) < 3:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    d = pts2 - pts1

    if region1 is not None and region2 is not None:
        expected = np.float32([region2['x'] - region1['x'],
                                region2['y'] - region1['y']])
        window = CONFIG.get('stage_prior_window', 50.0)
        plausible = np.abs(d - expected).max(axis=1) < window
    else:
        plausible = np.ones(len(d), dtype=bool)
    if plausible.sum() < 3:
        return None

    tx, ty = np.median(d[plausible], axis=0)
    # Final inlier accounting runs over ALL matches (not just the plausible
    # subset) against this seed, so real matches near the stage-prior window
    # edge still count -- the prior only protects the seed from corruption.
    inlier_mask = np.abs(d - [tx, ty]).max(axis=1) < reproj_thresh
    min_count = CONFIG.get('min_inlier_count', 10)
    if inlier_mask.sum() < min_count:
        return None
    tx, ty = d[inlier_mask].mean(axis=0)   # refine on inliers only

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    status = inlier_mask.astype(np.uint8).reshape(-1, 1)
    return (matches, H, status)


def calculate_homographies_batch(image_data, match_result, reproj_thresh=4):
    homography_results = {}
    success = failed = 0

    for (coord1, coord2), info in match_result.items():
        try:
            result = calculate_homography(
                info['keypoints1'], info['keypoints2'],
                info['matches'], reproj_thresh,
                info.get('region1'), info.get('region2')
            )
            if result is None:
                print(f"[SKIP] {coord1}↔{coord2}: too few matches or inconsistent displacement")
                failed += 1
                continue

            matches, H, status = result
            inliers = int(np.sum(status)) if status is not None else 0
            n       = len(matches)
            inlier_ratio = inliers / n if n else 0

            min_ratio = CONFIG.get('min_inlier_ratio', 0.3)
            min_count = CONFIG.get('min_inlier_count', 10)
            if inliers < min_count or inlier_ratio < min_ratio:
                print(f"[SKIP] {coord1}↔{coord2}: only {inliers}/{n} inliers "
                      f"({inlier_ratio:.2f}) -- below min_inlier_count={min_count} "
                      f"/ min_inlier_ratio={min_ratio}")
                failed += 1
                continue

            homography_results[(coord1, coord2)] = {
                'homography_matrix': H,
                'inliers':           inliers,
                'total_matches':     n,
                'inlier_ratio':      inlier_ratio,
                'direction':         info['direction'],
                'status':            status,
                'source':            'measured',
            }
            print(f"   OK {coord1}↔{coord2}: {inliers}/{n} inliers")
            success += 1
        except Exception as e:
            print(f"[ERROR] Homography failed {coord1}↔{coord2}: {e}")
            failed += 1

    total = success + failed
    print(f"\n📊 HOMOGRAPHY SUMMARY:")
    print(f"   Success : {success}")
    print(f"   Failed  : {failed}")
    if total:
        print(f"   Rate    : {success / total * 100:.1f}%")
    if homography_results:
        ratios = [r['inlier_ratio'] for r in homography_results.values()]
        print(f"   Avg inlier ratio: {np.mean(ratios):.3f} "
              f"(min {min(ratios):.3f}, max {max(ratios):.3f})")
    _print_translation_stats(homography_results)
    return homography_results


def _print_translation_stats(homography_results):
    """Print the measured tx/ty spread per direction (correction.txt step 1).

    A fixed-XY stage should report nearly the same translation for every
    horizontal pair, and the same for every vertical pair. A tight std here
    (roughly stage jitter, e.g. 8-15px) means registration is healthy and
    grid_adjust_outlier_tol / stage_prior_window can be tuned tightly around
    it. A std in the hundreds means registration itself is broken -- no
    amount of outlier-tolerance tuning will fix that; look at match quality
    (Root cause 2 in correction.txt) instead.
    """
    if not homography_results:
        return
    by_dir = defaultdict(list)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        by_dir[_ga_pair_direction(c1, c2)].append((float(H[0, 2]), float(H[1, 2])))

    print(f"\n📊 TRANSLATION SPREAD (per direction):")
    for d, rows in by_dir.items():
        tx = np.array([r[0] for r in rows])
        ty = np.array([r[1] for r in rows])
        print(f"   {d:10s}: tx med={np.median(tx):7.1f} std={tx.std():6.1f} | "
              f"ty med={np.median(ty):7.1f} std={ty.std():6.1f} | n={len(rows)}")


# ============================================================
# STEP 5.1 : Reprojection Error Calculation
# ============================================================

def calculate_reprojection_error(keypoints1, keypoints2, matches, homography_matrix):
    """
    Calculate reprojection error for matched keypoints using homography.
    Reprojects keypoints1 using H and compares with keypoints2.
    """
    if not matches:
        return {'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0}

    try:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float32)])
        proj = (homography_matrix @ pts1_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj - pts2, axis=1)
        return {
            'count': len(errors),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
        }
    except Exception as e:
        print(f"[WARNING] Error calculating reprojection error: {e}")
        return {'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0}


def evaluate_homography_reprojection(match_result, homography_results,
                                     output_csv="homography_reprojection_sift_bfm.csv"):
    results = []
    for (coord1, coord2), info in homography_results.items():
        match_info = match_result.get((coord1, coord2), {})
        kp1 = match_info.get('keypoints1')
        kp2 = match_info.get('keypoints2')
        matches = match_info.get('matches', [])
        if kp1 is None or kp2 is None or not matches:
            continue

        reproj = calculate_reprojection_error(kp1, kp2, matches, info['homography_matrix'])
        results.append({
            'tile1_coord': f"({coord1[0]},{coord1[1]})",
            'tile2_coord': f"({coord2[0]},{coord2[1]})",
            'match_count': info['total_matches'],
            'inliers': reproj['count'],
            'mean_error': round(reproj['mean'], 3),
            'median_error': round(reproj['median'], 3),
            'min_error': round(reproj['min'], 3),
            'max_error': round(reproj['max'], 3),
        })
        if CONFIG.get('verbose_pair_metrics', False):
            print(f"✅ {coord1}↔{coord2}: reprojection mean={reproj['mean']:.3f} "
                  f"median={reproj['median']:.3f} min={reproj['min']:.3f} max={reproj['max']:.3f}")

    if results:
        _save_csv(results, output_csv,
                  ['tile1_coord','tile2_coord','match_count','inliers','mean_error','median_error','min_error','max_error'])
    _print_reprojection_summary(results)
    return results


def _print_reprojection_summary(results):
    if not results:
        return
    print(f"\n📊 REPROJECTION ERROR SUMMARY:")
    for key in ('mean_error', 'median_error', 'min_error', 'max_error'):
        vals = [r[key] for r in results]
        print(f"{key.upper():12s} — mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")


# ============================================================
# STEP 5 (cont.) : Transformation Graph
# ============================================================

def add_nominal_fallback_edges(homography_results, overlap_pairs, verbose=True):
    """
    For every geometrically-expected adjacent pair (from calculate_overlap())
    that failed registration entirely, inject a nominal-shift homography
    derived from the pair's own overlap-region geometry, instead of just
    dropping the edge. A dropped edge fragments the transformation graph --
    exactly what turns "100 tiles" into "2 reachable tiles" once enough pairs
    fail. Nominal edges are tagged 'source': 'nominal_fallback' so callers
    can report how many were needed, distinguishing assumed from measured.
    """
    if not CONFIG.get('use_nominal_fallback', True):
        return homography_results

    added = 0
    for pair in overlap_pairs:
        key = (pair['coord1'], pair['coord2'])
        if key in homography_results:
            continue
        r1, r2 = pair['region1'], pair['region2']
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = r2['x'] - r1['x']
        H[1, 2] = r2['y'] - r1['y']
        homography_results[key] = {
            'homography_matrix': H,
            'inliers':           0,
            'total_matches':     0,
            'inlier_ratio':      0.0,
            'direction':         pair['direction'],
            'status':            None,
            'source':            'nominal_fallback',
        }
        added += 1

    if verbose and added:
        print(f"[FALLBACK] added {added} nominal-shift edge(s) for pair(s) that "
              f"failed registration, to keep the transformation graph connected")
    return homography_results


def build_transformation_graph(homography_results):
    graph = defaultdict(dict)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping singular homography for edge {c1}↔{c2}")
            continue
        graph[c1][c2] = H
        graph[c2][c1] = H_inv
    return dict(graph)


def _find_path_bfs(source, target, graph):
    """BFS shortest path. Uses deque for O(1) popleft (vs O(N) list.pop(0))."""
    if source == target:
        return [source]
    visited = {source}
    queue   = deque([(source, [source])])   # FIX: deque for O(1) dequeue
    while queue:
        node, path = queue.popleft()
        for nb in graph.get(node, {}):
            if nb == target:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def _calculate_transform_to_reference(source, reference, graph):
    path = _find_path_bfs(source, reference, graph)
    if not path or len(path) < 2:
        return None
    H = np.eye(3)
    for a, b in zip(path, path[1:]):
        if b not in graph.get(a, {}):
            return None
        H = graph[a][b] @ H
    return H


def select_reference_image(graph):
    """Select reference tile.

    Strategy: among tiles with the most connections (i.e. interior tiles),
    prefer the one closest to the geometric centre of the grid.  This
    minimises the worst-case BFS chain length and therefore reduces
    homography drift for large (e.g. 5×5+) mosaics.
    """
    counts   = {c: len(nb) for c, nb in graph.items()}
    max_conn = max(counts.values())
    # Candidates: all tiles sharing the maximum connection count
    candidates = [c for c, n in counts.items() if n == max_conn]
    # Geometric centre of the whole graph
    cx = sum(c[0] for c in graph) / len(graph)
    cy = sum(c[1] for c in graph) / len(graph)
    reference = min(candidates,
                    key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    print(f"📍 Reference image: {reference} "
          f"({counts[reference]} connections, closest to grid centre)")
    return reference


def calculate_all_transforms(image_data, homography_results):
    graph     = build_transformation_graph(homography_results)
    reference = select_reference_image(graph)

    transforms       = {}
    reachable_images = []

    for coord in image_data:
        if coord == reference:
            transforms[coord] = np.eye(3)
            reachable_images.append(coord)
        else:
            T = _calculate_transform_to_reference(coord, reference, graph)
            if T is not None:
                transforms[coord] = T
                reachable_images.append(coord)
                print(f"   ✅ {coord} → {reference}")
            else:
                print(f"   ❌ {coord} → {reference}: no path")

    print(f"📊 Reachable: {len(reachable_images)}/{len(image_data)}")
    return transforms, reference, reachable_images


# ============================================================
# STEP 5 (cont.) : Global Translation Adjustment  (self-contained)
# ============================================================
# Drop-in alternative to the BFS `calculate_all_transforms` above. Instead of
# chaining pairwise transforms along one path from a reference tile -- which
# lets each pair's small error ACCUMULATE with distance from the reference
# (the "staircase drift" at the mosaic corners on large grids) -- this solves
# every tile's translation into the reference frame AT ONCE by least squares.
# Each registered pair (c1, c2) with 3x3 transform H contributes one linear
# constraint per axis:
#
#         t_c1 - t_c2 = (H[0,2], H[1,2])
#
# The stacked system A t = b is over-determined; np.linalg.lstsq finds the
# tile positions that best satisfy ALL pairs simultaneously. For a spanning
# tree this reproduces the BFS chain exactly; every loop-closing edge adds a
# constraint that spreads the error evenly instead of trusting one chain.
# Only the TRANSLATION part of H is used -- the right model for a fixed XY
# stage, where any per-pair rotation/scale an estimator reports is noise.
#
# Gated by CONFIG['use_global_adjustment']; the pre-filter tolerance is
# CONFIG['grid_adjust_outlier_tol'].  (Previously in stitch_global_adjust.py,
# inlined here so each pipeline script stands alone.)

def _ga_pair_direction(c1, c2):
    """'horizontal' if the two tiles share a grid row (differ in x only),
    'vertical' if they share a column. Coords are grid indices."""
    if c1[1] == c2[1]:
        return 'horizontal'
    if c1[0] == c2[0]:
        return 'vertical'
    return 'other'


def _ga_connected_component(adj, start):
    seen = {start}
    q = deque([start])
    while q:
        n = q.popleft()
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                q.append(m)
    return seen


def _adaptive_tol(vals, floor=5.0, k=3.0):
    """Robust per-axis tolerance from the median absolute deviation, instead
    of a fixed guess. A fixed absolute tolerance rejects a growing fraction
    of legitimate pairs as grid extent (and thus the true per-pair spread)
    grows with it -- see correction.txt. 1.4826*MAD is a consistent estimator
    of std for Gaussian-like noise; `floor` keeps tight, low-noise spreads
    from being over-rejected on sub-pixel jitter."""
    vals = np.asarray(vals, dtype=np.float64)
    mad = float(np.median(np.abs(vals - np.median(vals))))
    return max(floor, k * 1.4826 * mad)


def reject_grid_outliers(homography_results, tol, verbose=True):
    """
    Fixed-stage prior: every horizontal pair should report the same
    translation, every vertical pair the same. Drop any MEASURED pair whose
    translation deviates from its direction's median by more than a
    tolerance, BEFORE the least-squares solve (plain L2 is not
    outlier-robust, so one bad pair would smear its error across the whole
    grid).

    *tol*: None -> no-op (skip filtering). 'adaptive' -> derive a robust
    per-direction, per-axis tolerance from the measured spread itself (see
    _adaptive_tol) instead of a fixed guess. A number -> that fixed px
    tolerance on both axes, as before.

    Two rules that matter when nominal-fallback edges are present (see
    add_nominal_fallback_edges):
      1. The reference median/tolerance is computed from 'measured' edges
         ONLY when any exist for that direction. Nominal-fallback edges are
         synthetic (generated from the same formula every time), so mixing
         them into the population would collapse the computed spread toward
         zero once they outnumber real measurements, backwards-rejecting
         genuinely good real measurements that legitimately disagree with
         the assumed grid geometry (this is not hypothetical: it is exactly
         what happened on a 540-tile run where 89% of pairs fell back).
      2. Nominal-fallback edges are NEVER rejected here -- they exist
         specifically to keep the transformation graph connected when real
         registration fails, and their low weight in the least-squares
         solve (CONFIG['nominal_fallback_weight']) already limits their
         influence. Rejecting them too would just re-fragment the graph.

    Returns (kept_dict, rejected_keys).
    """
    if tol is None or not homography_results:
        return homography_results, []

    by_dir = defaultdict(list)
    by_dir_measured = defaultdict(list)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        row = ((c1, c2), float(H[0, 2]), float(H[1, 2]))
        d = _ga_pair_direction(c1, c2)
        by_dir[d].append(row)
        if info.get('source', 'measured') == 'measured':
            by_dir_measured[d].append(row)

    stat_rows = {d: (by_dir_measured[d] if by_dir_measured.get(d) else rows)
                 for d, rows in by_dir.items()}

    median = {
        d: (float(np.median([r[1] for r in rows])),
            float(np.median([r[2] for r in rows])))
        for d, rows in stat_rows.items()
    }

    adaptive = (tol == 'adaptive')
    tol_xy = {}
    if adaptive:
        for d, rows in stat_rows.items():
            tx_tol = _adaptive_tol([r[1] for r in rows])
            ty_tol = _adaptive_tol([r[2] for r in rows])
            tol_xy[d] = (tx_tol, ty_tol)
            if verbose:
                n_meas = len(by_dir_measured.get(d, []))
                print(f"[GLOBAL-ADJ] adaptive tolerance ({d}): "
                      f"tx_tol={tx_tol:.1f}px ty_tol={ty_tol:.1f}px "
                      f"(from {n_meas} measured / {len(by_dir[d])} total)")

    kept, rejected = {}, []
    for (c1, c2), info in homography_results.items():
        if info.get('source') == 'nominal_fallback':
            kept[(c1, c2)] = info
            continue
        d = _ga_pair_direction(c1, c2)
        mx, my = median[d]
        tx_tol, ty_tol = tol_xy[d] if adaptive else (tol, tol)
        H = info['homography_matrix']
        if abs(float(H[0, 2]) - mx) > tx_tol or abs(float(H[1, 2]) - my) > ty_tol:
            rejected.append((c1, c2))
        else:
            kept[(c1, c2)] = info

    if verbose and rejected:
        print(f"[GLOBAL-ADJ] rejected {len(rejected)} grid-inconsistent "
              f"pair(s) from row/col median before solve")
    return kept, rejected


def _ga_select_reference(adj):
    """Most-connected tile, tie-broken by proximity to the grid centre --
    mirrors select_reference_image()."""
    counts = {c: len(nb) for c, nb in adj.items()}
    max_conn = max(counts.values())
    candidates = [c for c, n in counts.items() if n == max_conn]
    cx = sum(c[0] for c in adj) / len(adj)
    cy = sum(c[1] for c in adj) / len(adj)
    return min(candidates, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)


def solve_global_translations(homography_results, image_data,
                              reference=None, outlier_tol=None, verbose=True):
    """
    Least-squares solve for every tile's pure-translation transform into the
    reference frame. Drop-in replacement for calculate_all_transforms():

        transforms, reference, reachable_images = \
            solve_global_translations(homography_results, image_data, ...)

    transforms[coord] is a 3x3 translation matrix (identity for the
    reference). Tiles not connected to the reference are omitted, exactly
    like the BFS version.

    Edges tagged 'source': 'nominal_fallback' (see add_nominal_fallback_edges)
    are down-weighted (CONFIG['nominal_fallback_weight'], default 0.1)
    relative to measured edges (weight 1.0) via a standard weighted-least-
    squares trick: scale each constraint row/target by sqrt(weight) before
    the ordinary lstsq solve. They keep the graph connected but shouldn't be
    trusted as much as a real measurement (see correction.txt).
    """
    hr, _ = reject_grid_outliers(homography_results, outlier_tol, verbose)
    if not hr:                         # everything rejected -> fall back
        hr = homography_results

    adj = defaultdict(set)
    for (c1, c2) in hr:
        adj[c1].add(c2)
        adj[c2].add(c1)
    if not adj:
        return {}, reference, []

    if reference is None or reference not in adj:
        reference = _ga_select_reference(adj)

    component = _ga_connected_component(adj, reference)
    solve_nodes = [c for c in component if c != reference]
    col = {c: i for i, c in enumerate(solve_nodes)}
    m = len(solve_nodes)

    transforms = {reference: np.eye(3)}
    reachable = [reference]

    if m > 0:
        nominal_weight = CONFIG.get('nominal_fallback_weight', 0.1)
        rows, bx, by, weights = [], [], [], []
        for (c1, c2), info in hr.items():
            if c1 not in component or c2 not in component:
                continue
            H = info['homography_matrix']
            row = np.zeros(m)
            if c1 != reference:
                row[col[c1]] += 1.0
            if c2 != reference:
                row[col[c2]] -= 1.0
            rows.append(row)
            bx.append(float(H[0, 2]))
            by.append(float(H[1, 2]))
            weights.append(nominal_weight if info.get('source') == 'nominal_fallback' else 1.0)

        A = np.asarray(rows)
        bx = np.asarray(bx)
        by = np.asarray(by)
        sw = np.sqrt(np.asarray(weights))
        A_w  = A * sw[:, None]
        bx_w = bx * sw
        by_w = by * sw
        tx, *_ = np.linalg.lstsq(A_w, bx_w, rcond=None)
        ty, *_ = np.linalg.lstsq(A_w, by_w, rcond=None)

        if verbose:
            res = np.sqrt((A @ tx - bx) ** 2 + (A @ ty - by) ** 2)
            n_nominal = int(np.sum(sw < 1.0))
            print(f"[GLOBAL-ADJ] solved {A.shape[0]} constraints "
                  f"({n_nominal} nominal-fallback, weight={nominal_weight}) for "
                  f"{m + 1} tiles | per-pair residual px: "
                  f"mean={res.mean():.2f} median={np.median(res):.2f} "
                  f"max={res.max():.2f}")

        for c in solve_nodes:
            T = np.eye(3)
            T[0, 2] = tx[col[c]]
            T[1, 2] = ty[col[c]]
            transforms[c] = T
            reachable.append(c)

    if verbose:
        n_total = len(image_data) if image_data is not None else len(component)
        print(f"[GLOBAL-ADJ] reference={reference} | "
              f"reachable {len(reachable)}/{n_total}")
    return transforms, reference, reachable


def calculate_optimal_canvas(image_data, transforms):
    all_corners  = []
    corner_coord = []
    for coord, T in transforms.items():
        h, w = image_data[coord]['image'].shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_corners.extend(cv2.perspectiveTransform(corners, T).reshape(-1, 2))
        corner_coord.extend([coord] * 4)

    arr    = np.array(all_corners)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    pad    = CONFIG['canvas_padding']

    canvas_w = int(max_xy[0] - min_xy[0]) + 2 * pad
    canvas_h = int(max_xy[1] - min_xy[1]) + 2 * pad

    # Sanity cap: a real N-tile grid never gets remotely close to this. If we
    # hit it, it means a bad accumulated transform pushed a tile far off the
    # mosaic -- not that the mosaic is genuinely huge -- so fail fast with a
    # diagnostic pointing at the outlier tile(s) instead of trying to
    # allocate a multi-gigapixel canvas and dying with a raw MemoryError.
    max_mp = CONFIG.get('max_canvas_megapixels', 100)
    if canvas_w * canvas_h > max_mp * 1_000_000:
        outlier_idx = {int(arr[:, 0].argmin()), int(arr[:, 0].argmax()),
                       int(arr[:, 1].argmin()), int(arr[:, 1].argmax())}
        outliers = sorted({corner_coord[i] for i in outlier_idx})
        raise RuntimeError(
            f"calculate_optimal_canvas: would need {canvas_w}x{canvas_h} "
            f"({canvas_w * canvas_h / 1e6:.0f} MP), exceeding "
            f"CONFIG['max_canvas_megapixels']={max_mp}. This almost always "
            f"means at least one accumulated transform is wrong, not that "
            f"the mosaic is really this big. Suspect tile(s): {outliers} -- "
            f"check their pairwise homographies/inlier ratios (or set "
            f"CONFIG['use_global_adjustment']=True), or raise "
            f"max_canvas_megapixels if this size is genuinely expected."
        )

    offset_x = int(-min_xy[0]) + pad
    offset_y = int(-min_xy[1]) + pad

    print(f"   Canvas  : {canvas_w} × {canvas_h}")
    print(f"   Offset  : ({offset_x}, {offset_y})")
    return canvas_w, canvas_h, offset_x, offset_y


# ============================================================
# STEP 5.2 : NCC Calculation
# ============================================================

def _to_gray_for_ncc(img):
    if img is None:
        return None
    a = np.clip(img, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        return a[:, :, 0].astype(np.float32)
    return a.astype(np.float32)


def calculate_overlap_ncc(image_data, transforms, coord1, coord2,
                          canvas_w, canvas_h, offset_x, offset_y):
    """
    Calculate normalized cross-correlation (NCC) inside the overlap region
    between two warped images.
    """
    try:
        off = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
        img1 = image_data[coord1]['image'].astype(np.float32)
        img2 = image_data[coord2]['image'].astype(np.float32)
        w1 = cv2.warpPerspective(img1, off @ transforms[coord1], (canvas_w, canvas_h))
        w2 = cv2.warpPerspective(img2, off @ transforms[coord2], (canvas_w, canvas_h))

        mask1 = (w1.sum(axis=2) > 0)
        mask2 = (w2.sum(axis=2) > 0)
        overlap = mask1 & mask2
        n_overlap = int(overlap.sum())
        if n_overlap < 100:
            return None

        g1 = _to_gray_for_ncc(w1)
        g2 = _to_gray_for_ncc(w2)
        ov1 = g1[overlap]
        ov2 = g2[overlap]

        mean1, mean2 = np.mean(ov1), np.mean(ov2)
        std1, std2 = np.std(ov1), np.std(ov2)
        if std1 < 1e-6 or std2 < 1e-6:
            ncc = 0.0
        else:
            ncc = float(np.mean((ov1 - mean1) * (ov2 - mean2) / (std1 * std2)))
            ncc = np.clip(ncc, -1.0, 1.0)

        return {'ncc': ncc, 'overlap_pixels': n_overlap}
    except Exception as e:
        print(f"[WARNING] Error calculating NCC for {coord1}↔{coord2}: {e}")
        return None


# ============================================================
# STEP 5.5 : Overlap Quality Metrics
# ============================================================

def _to_gray(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img


def normalize_intensity(img1, img2, method='clahe'):
    """Normalize intensity of two images before metric computation.

    Note on histogram_match direction: when method='skimage_histogram_match',
    img2 is always matched to img1 (i.e. img2 is normalized to img1's
    histogram).  This means metrics are always computed with tile2 normalized
    to tile1.  The direction is deterministic within a run, but be aware that
    pair ordering from homography_results.keys() may not always place the same
    physical tile as coord1.
    """
    i1 = img1.copy().astype(np.float32)
    i2 = img2.copy().astype(np.float32)

    if method == 'none':
        pass
    elif method == 'mean_std':
        m1, s1 = np.mean(i1), np.std(i1)
        m2, s2 = np.mean(i2), np.std(i2)
        if s2 > 0:
            i2 = (i2 - m2) * (s1 / s2) + m1
        i2 = np.clip(i2, 0, 255)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if img1.ndim == 3:
            for c in range(img1.shape[2]):
                i1[:,:,c] = clahe.apply(i1[:,:,c].astype(np.uint8))
                i2[:,:,c] = clahe.apply(i2[:,:,c].astype(np.uint8))
        else:
            i1 = clahe.apply(i1.astype(np.uint8)).astype(np.float32)
            i2 = clahe.apply(i2.astype(np.uint8)).astype(np.float32)
    elif method == 'skimage_histogram_match':
        ch_axis = -1 if img1.ndim == 3 else None
        i2 = match_histograms(i2, i1, channel_axis=ch_axis).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return i1.astype(np.uint8), i2.astype(np.uint8)


def compute_psnr(img1, img2, norm_method='none'):
    a, b = normalize_intensity(img1, img2, norm_method)
    mse  = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2, norm_method='none'):
    a, b = normalize_intensity(img1, img2, norm_method)
    a, b = _to_gray(a), _to_gray(b)
    if a.size < 49:
        return 0.0
    return float(ssim(a, b, data_range=a.max() - a.min()))


def compute_rmse(img1, img2, norm_method='none'):
    a, b = normalize_intensity(img1, img2, norm_method)
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _tile_bbox(image_data, coord, T_adj, canvas_w, canvas_h, pad=0):
    """Warped bounding box of one tile under T_adj, clamped to the canvas.

    Warping a tile into this tile-sized box instead of the full canvas is
    the "windowed warp" from correction.txt -- cv2.warpPerspective's output
    size is the single biggest memory cost in detect_pairwise_overlap() and
    blend_panorama(), and a tile only ever covers a small fraction of a
    large mosaic's canvas (a 20x20 grid's per-tile transient is ~2.5GB under
    a whole-canvas warp).

    *pad*: extra margin (px) added around the tile's own footprint. Since
    every transform here is a pure translation, a warped tile is an
    axis-aligned rectangle that exactly fills its own tight bbox -- with
    pad=0 there is NO zero-background pixel anywhere in the window, which
    silently breaks blend_panorama()'s cv2.erode()/distanceTransform() (they
    need real zero-padding around the shape to compute the feather falloff
    correctly; without it cv2.erode's border handling treats the window
    edge as foreground). blend_panorama() must pass pad >= its feather
    reach; detect_pairwise_overlap() has no morphology step and is exact
    with pad=0.
    """
    h, w = image_data[coord]['image'].shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(corners, T_adj).reshape(-1, 2)
    x0 = max(0, int(np.floor(wc[:,0].min())) - pad)
    y0 = max(0, int(np.floor(wc[:,1].min())) - pad)
    x1 = min(canvas_w, int(np.ceil(wc[:,0].max())) + pad)
    y1 = min(canvas_h, int(np.ceil(wc[:,1].max())) + pad)
    return x0, y0, x1, y1


def detect_pairwise_overlap(image_data, transforms, coord1, coord2,
                             canvas_w, canvas_h, offset_x, offset_y):
    """Windowed warp: both tiles are warped into the union of their own
    bounding boxes, not the full canvas. Their overlap (intersection) is
    always fully contained within that union, so this is exact -- just
    cheaper than a full-canvas warp for two tiles on a large mosaic."""
    offset_matrix = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float32)

    img1 = image_data[coord1]['image'].astype(np.float32)
    img2 = image_data[coord2]['image'].astype(np.float32)
    T1   = offset_matrix @ transforms[coord1]
    T2   = offset_matrix @ transforms[coord2]

    bx0_1, by0_1, bx1_1, by1_1 = _tile_bbox(image_data, coord1, T1, canvas_w, canvas_h)
    bx0_2, by0_2, bx1_2, by1_2 = _tile_bbox(image_data, coord2, T2, canvas_w, canvas_h)
    x0, y0 = min(bx0_1, bx0_2), min(by0_1, by0_2)
    x1, y1 = max(bx1_1, bx1_2), max(by1_1, by1_2)
    if x1 <= x0 or y1 <= y0:
        return None, None, None, 0
    T_shift = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], dtype=np.float32)

    w1 = cv2.warpPerspective(img1, T_shift @ T1, (x1 - x0, y1 - y0))
    w2 = cv2.warpPerspective(img2, T_shift @ T2, (x1 - x0, y1 - y0))

    mask1 = (w1.sum(axis=2) > 0).astype(np.uint8)
    mask2 = (w2.sum(axis=2) > 0).astype(np.uint8)
    ov    = (mask1 > 0) & (mask2 > 0)

    if ov.sum() < 100:
        return None, None, None, 0

    ov_r1 = w1.copy()
    ov_r2 = w2.copy()
    for c in range(3):
        ov_r1[:,:,c] *= ov
        ov_r2[:,:,c] *= ov

    rows, cols    = np.where(ov)
    r0, r1_       = rows.min(), rows.max()
    c0, c1_       = cols.min(), cols.max()
    return (ov_r1[r0:r1_+1, c0:c1_+1],
            ov_r2[r0:r1_+1, c0:c1_+1],
            ov[r0:r1_+1, c0:c1_+1],
            int(ov.sum()))


def evaluate_overlap_metrics(image_data, transforms, reachable_images,
                              canvas_w, canvas_h, offset_x, offset_y,
                              homography_results=None,
                              output_csv="overlap_metrics_sift_bfm.csv",
                              norm_method=None,
                              visualize_sample=False):
    """Compute PSNR, SSIM, RMSE, NCC for overlapping tile pairs.

    Scalability fix
    ---------------
    When *homography_results* is provided (recommended), only the
    adjacent registered pairs are evaluated -- O(N) instead of the
    previous O(N²) double-loop over all reachable-image combinations.
    For a 5×5 grid this reduces from C(25,2)=300 pair-warpings down to
    ~40 (the actual adjacent pairs).  NCC is now computed inline from
    the already-warped overlap crops so there is no redundant re-warp.
    """
    norm_method = norm_method or CONFIG['normalization_method']
    results     = []

    # Build pair list: adjacent only (fast) or all combinations (legacy)
    if homography_results is not None:
        pairs = list(homography_results.keys())
    else:
        # Fallback -- O(N²), kept for backward-compat with old call sites
        pairs = [(coord1, coord2)
                 for i, coord1 in enumerate(reachable_images)
                 for j, coord2 in enumerate(reachable_images)
                 if i < j]

    for (coord1, coord2) in pairs:
        if coord1 not in transforms or coord2 not in transforms:
            continue
        try:
            ov1, ov2, ov_mask, n_pixels = detect_pairwise_overlap(
                image_data, transforms, coord1, coord2,
                canvas_w, canvas_h, offset_x, offset_y
            )
            if ov1 is None:
                continue

            ov1_u8 = np.clip(ov1, 0, 255).astype(np.uint8)
            ov2_u8 = np.clip(ov2, 0, 255).astype(np.uint8)

            psnr = compute_psnr(ov1_u8, ov2_u8, norm_method)
            s    = compute_ssim(ov1_u8, ov2_u8, norm_method)
            rmse = compute_rmse(ov1_u8, ov2_u8, norm_method)

            # NCC computed inline from already-warped crops -- no re-warp
            ncc = None
            if n_pixels >= 100:
                g1 = _to_gray_for_ncc(ov1_u8)
                g2 = _to_gray_for_ncc(ov2_u8)
                ov_flat1 = g1[ov_mask]
                ov_flat2 = g2[ov_mask]
                mean1, mean2 = ov_flat1.mean(), ov_flat2.mean()
                std1,  std2  = ov_flat1.std(),  ov_flat2.std()
                if std1 > 1e-6 and std2 > 1e-6:
                    ncc = float(np.clip(
                        np.mean((ov_flat1 - mean1) * (ov_flat2 - mean2)
                                / (std1 * std2)), -1.0, 1.0))
                else:
                    ncc = 0.0

            results.append({
                'tile1_coord':    f"({coord1[0]},{coord1[1]})",
                'tile2_coord':    f"({coord2[0]},{coord2[1]})",
                'overlap_pixels': n_pixels,
                'overlap_width':  ov1.shape[1],
                'overlap_height': ov1.shape[0],
                'psnr':           round(psnr, 3),
                'ssim':           round(s,    4),
                'rmse':           round(rmse, 3),
                'ncc':            round(ncc, 4) if ncc is not None else None,
            })
            if CONFIG.get('verbose_pair_metrics', False):
                print(f"✅ {coord1} ↔ {coord2}: PSNR={psnr:.2f} SSIM={s:.3f} RMSE={rmse:.2f} NCC={ncc if ncc is not None else 'N/A'} | {n_pixels}px")

        except Exception as e:
            print(f"[ERROR] {coord1} ↔ {coord2}: {e}")

    _save_csv(results, output_csv,
              ['tile1_coord','tile2_coord','overlap_pixels','overlap_width','overlap_height','psnr','ssim','rmse','ncc'])
    _print_metric_summary(results)
    return results


def _save_csv(results, path, fieldnames):
    if not results:
        print("⚠️  No results to save.")
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved: {path}")


def _print_metric_summary(results):
    if not results:
        return
    metric_keys = [k for k in results[0].keys()
                   if k not in ('tile1_coord', 'tile2_coord', 'overlap_pixels', 'overlap_width', 'overlap_height')]
    for key in metric_keys:
        vals = [r[key] for r in results if r[key] is not None]
        if not vals:
            continue
        print(f"{key.upper():4s} — mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")


# ============================================================
# STEP 6 : Feather Blending
# ============================================================

def blend_panorama(image_data, transforms, reachable_images,
                   canvas_w, canvas_h, offset_x, offset_y):
    """
    Warp and feather-blend all reachable tiles onto a single canvas.

    Windowed-warp strategy
    -----------------------------------
    Each tile is warped into its OWN bounding box (see _tile_bbox()), not
    the full canvas -- cv2.warpPerspective's output size is the single
    biggest memory cost here, and a tile only ever covers a small fraction
    of a large mosaic's canvas (see correction.txt: a 20x20 grid's per-tile
    transient alone is ~2.5GB under a whole-canvas warp). The persistent
    canvas/weight_map/overlap_count buffers are still canvas-sized --
    blending is inherently cumulative across tiles -- but the per-tile
    transient drops from O(canvas_size) to O(tile_bbox_size), which is what
    actually exhausts memory on large grids.

    Accumulate-then-normalise compositing (correction.txt)
    --------------------------------------------------------
    Every tile adds a weighted contribution to `canvas`/`weight_map`; the
    single normalising divide happens once, at the very end, in the same
    strip loop that replaces the old full-canvas copy chain. This replaces
    sequential alpha-over compositing with a running `max(ext_w, cur_w)`
    weight tracker, which is only correct for exactly two overlapping
    tiles -- wrong (and order-dependent) at any pixel touched by 3+ tiles,
    i.e. every interior grid intersection. That was a real correctness bug
    (a checkerboard-with-bright-intersections pattern), not just a memory
    one; this fixes both at once, since accumulate-then-normalise also
    removes the overlap_here/new_here branching and its overlap_count
    dependency entirely -- "first touch" and "Nth touch" are the same code
    path once canvas/weight_map start at zero.

    Pass 1 -- warp each tile into its own bbox to build overlap_count (kept
              for the debug/stats output only -- the blend itself no longer
              needs it), then discard the small warp immediately.
    Pass 2 -- re-warp each tile into its own bbox, accumulate its weighted
              contribution into the canvas, then discard the warp.

    Final conversion is done in row-strips, in-place: `canvas[valid]` /
    `np.clip(...)` / `.astype(...)` on the un-strip version each
    materialise a full canvas-sized copy (3 extra full buffers stacked on
    top of the already-resident ones, right at the end of the run -- the
    actual peak-memory moment on a large grid, not the loop). `np.unique`
    is likewise replaced with `np.bincount` for the overlap-count histogram
    (no full sort).

    Returns
    -------
    final      : np.uint8  (H, W, 3)  -- blended panorama
    debug_info : dict  -- overlap_count, weight_map, valid_pixels, overlap_stats
    """
    offset_matrix = np.array([[1, 0, offset_x],
                               [0, 1, offset_y],
                               [0, 0, 1]], dtype=np.float32)

    canvas        = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map    = np.zeros((canvas_h, canvas_w),    dtype=np.float32)
    overlap_count = np.zeros((canvas_h, canvas_w),    dtype=np.int32)

    # Every transform here is a pure translation, so a warped tile is an
    # axis-aligned rectangle that exactly fills its own tight bbox -- no
    # zero-background pixel would exist in the window without this margin,
    # which breaks the erode()/distanceTransform() feather calculation
    # below (see _tile_bbox()'s docstring). Must cover the erosion kernel's
    # reach (2 iterations of 5x5 =~ 4px) plus the feather falloff distance.
    feather_pad = int(CONFIG.get('feather_distance', 30)) + 10

    print(f"Pass 1 -- computing overlap map ({len(reachable_images)} tiles) ...")
    for coord in reachable_images:
        img   = image_data[coord]['image'].astype(np.float32)
        T_adj = offset_matrix @ transforms[coord]
        x0, y0, x1, y1 = _tile_bbox(image_data, coord, T_adj, canvas_w, canvas_h, pad=feather_pad)
        if x1 <= x0 or y1 <= y0:
            continue
        T_local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float32) @ T_adj
        small = cv2.warpPerspective(img, T_local, (x1 - x0, y1 - y0))
        overlap_count[y0:y1, x0:x1] += (small.sum(axis=2) > 0).astype(np.int32)
        del small

    kernel = np.ones((5, 5), np.uint8)
    print(f"Pass 2 -- blending ({len(reachable_images)} tiles) ...")
    for coord in reachable_images:
        img    = image_data[coord]['image'].astype(np.float32)
        if _TILE_GAINS is not None:
            # Per-tile photometric gain (STEP 5.6). Applied here in Pass 2
            # only: Pass 1 just builds overlap_count from `sum(axis=2) > 0`,
            # and a positive gain cannot change whether a pixel is non-zero.
            # Values may exceed 255 after scaling -- the strip conversion at
            # the end of this function clips.
            _g = _TILE_GAINS.get(coord)
            if _g is not None:
                img = img * _g
        if _SHADING_CORR is not None and img.shape == _SHADING_CORR.shape:
            # Overlap-driven shading correction (STEP 5.7). Same placement
            # rationale as the gain above: Pass 1 only tests `sum(axis=2) > 0`
            # and a positive multiplicative field cannot change that, so
            # applying it in Pass 2 alone is consistent.
            img = img * _SHADING_CORR
        T_adj  = offset_matrix @ transforms[coord]
        x0, y0, x1, y1 = _tile_bbox(image_data, coord, T_adj, canvas_w, canvas_h, pad=feather_pad)
        if x1 <= x0 or y1 <= y0:
            continue
        T_local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float32) @ T_adj
        warped = cv2.warpPerspective(img, T_local, (x1 - x0, y1 - y0))
        mask   = (warped.sum(axis=2) > 0).astype(np.uint8)

        inner        = cv2.erode(mask, kernel, iterations=2)
        dist         = cv2.distanceTransform(inner, cv2.DIST_L2, 5)
        max_dist     = dist.max()
        feather_zone = min(CONFIG['feather_distance'], max_dist * 0.3) if max_dist > 0 else 1
        feather      = np.minimum(dist / feather_zone, 1.0) * mask.astype(np.float32)

        # Accumulate-then-normalise: floor the feather so an edge pixel that
        # legitimately has zero distance-transform weight (right at the tile
        # boundary) still contributes instead of being silently dropped;
        # re-apply the mask afterward so the floor never leaks outside the
        # tile's real footprint.
        feather = np.maximum(feather, 1e-3) * mask.astype(np.float32)

        # Views into the canvas-sized buffers, sliced to this tile's bbox --
        # basic slicing returns a view, so in-place writes below apply
        # straight through to canvas/weight_map with no explicit writeback.
        canvas_slice = canvas[y0:y1, x0:x1]
        weight_slice = weight_map[y0:y1, x0:x1]

        for c in range(3):
            canvas_slice[:, :, c] += warped[:, :, c] * feather
        weight_slice += feather

        del warped

    # canvas currently holds a WEIGHTED SUM (accumulate, not average, in the
    # loop above), so this pass also does the single normalising divide --
    # one division per pixel instead of one per tile-pair -- and it folds
    # into the same strip loop that used to be the memory-spike site.
    STRIP = 512
    final = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    for y0s in range(0, canvas_h, STRIP):
        ys = slice(y0s, min(y0s + STRIP, canvas_h))
        w = weight_map[ys][:, :, None]
        np.divide(canvas[ys], w, out=canvas[ys], where=w > 1e-6)
        np.clip(canvas[ys], 0, 255, out=canvas[ys])
        final[ys] = canvas[ys].astype(np.uint8)
        final[ys][weight_map[ys] <= 0] = 0

    # np.unique(...) sorts the whole array (plus workspace) just to build a
    # histogram of a handful of small integer values -- np.bincount doesn't.
    counts = np.bincount(overlap_count.ravel())
    overlap_stats = {i: int(c) for i, c in enumerate(counts) if c}

    debug_info = {
        'overlap_count': overlap_count,
        'weight_map':    weight_map,
        'valid_pixels':  weight_map > 0,
        'overlap_stats': overlap_stats,
    }
    return final, debug_info


def visualize_blending_debug(debug_info, final_image):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    im1 = axes[0,0].imshow(debug_info['overlap_count'], cmap='jet')
    axes[0,0].set_title('Overlap Count Map')
    plt.colorbar(im1, ax=axes[0,0])
    im2 = axes[0,1].imshow(debug_info['weight_map'], cmap='viridis')
    axes[0,1].set_title('Weight Map')
    plt.colorbar(im2, ax=axes[0,1])
    axes[1,0].imshow(debug_info['valid_pixels'].astype(np.uint8)*255, cmap='gray')
    axes[1,0].set_title('Valid Pixels Mask')
    axes[1,1].imshow(final_image)
    axes[1,1].set_title('Blended Result')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
# STEP 6 (cont.) : ROI Cropping
# ============================================================

def _build_height_matrix_vectorized(valid_mask):
    h_matrix = np.zeros_like(valid_mask, dtype=np.int32)
    col_data = valid_mask.astype(np.int32)
    for row in range(valid_mask.shape[0]):
        if row == 0:
            h_matrix[row] = col_data[row]
        else:
            h_matrix[row] = (h_matrix[row - 1] + 1) * col_data[row]
    return h_matrix


def largest_rectangle_in_histogram(heights):
    stack, max_area, best = [], 0, (0, 0, 0)
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width  = i if not stack else i - stack[-1] - 1
            area   = height * width
            if area > max_area:
                max_area = area
                left     = 0 if not stack else stack[-1] + 1
                best     = (left, width, height)
        stack.append(i)
    while stack:
        height = heights[stack.pop()]
        width  = len(heights) if not stack else len(heights) - stack[-1] - 1
        area   = height * width
        if area > max_area:
            max_area = area
            left     = 0 if not stack else stack[-1] + 1
            best     = (left, width, height)
    return max_area, best


def find_roi(image, min_threshold=1, debug=True):
    """Find the largest axis-aligned rectangle free of black borders."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    valid_mask = (gray >= min_threshold).astype(np.int32)
    rows, cols = valid_mask.shape

    if debug:
        pct = valid_mask.sum() / (rows * cols) * 100
        print(f"Image: {cols}x{rows} | Valid pixels: {valid_mask.sum()} ({pct:.1f}%)")

    h_matrix = _build_height_matrix_vectorized(valid_mask)

    max_area, best_roi = 0, None
    for row in range(rows):
        area, (left, width, height) = largest_rectangle_in_histogram(h_matrix[row])
        if area > max_area:
            max_area = area
            best_roi = {'x': left, 'y': row - height + 1,
                        'width': width, 'height': height, 'area': area}

    if best_roi and debug:
        eff = best_roi['area'] / (rows * cols) * 100
        ar  = best_roi['width'] / best_roi['height'] if best_roi['height'] else 0
        print(f"ROI: pos=({best_roi['x']},{best_roi['y']}) "
              f"size={best_roi['width']}x{best_roi['height']} "
              f"eff={eff:.1f}% ar={ar:.2f}")
    return best_roi


def extract_roi(image, roi_info, padding=5):
    """Crop ROI from *image* with optional *padding* on all sides."""
    if roi_info is None:
        return image, {}
    x  = max(0, roi_info['x'] - padding)
    y  = max(0, roi_info['y'] - padding)
    x2 = min(image.shape[1], roi_info['x'] + roi_info['width']  + padding)
    y2 = min(image.shape[0], roi_info['y'] + roi_info['height'] + padding)
    roi_img = image[y:y2, x:x2]
    stats = {
        'original_size':      (image.shape[1], image.shape[0]),
        'roi_size':           (x2 - x, y2 - y),
        'roi_bbox':           (x, y, x2 - x, y2 - y),
        'area_efficiency':    (x2 - x) * (y2 - y) / (image.shape[0] * image.shape[1]) * 100,
        'content_efficiency': roi_info['area'] / ((x2 - x) * (y2 - y)) * 100,
    }
    return roi_img, stats


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser(description="SIFT + BFMatcher tile stitching")
    _parser.add_argument('--path', default="/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/10x10_euglena_red",
                         help="Folder of tile images to stitch")
    folder_path = _parser.parse_args().path

    # Initialize Performance & Memory Tracker
    tracker = None
    if CONFIG.get('enable_benchmark', True):
        tracker = PerformanceTracker(f"SIFT + BFMatcher Pipeline ({CONFIG['feature_method'].upper()})")

    # Step 1 -- Load images & build grid
    image_data, grid_info = load_image(folder_path, CONFIG['resize_factor'])
    if CONFIG.get('debug', False):
        visualize_grid_preview(image_data, grid_info)
        plt.show()
    if tracker:
        tracker.record_step("1. Image Loading & Grid Preview")

    # Step 2 -- Compute overlap zones
    overlap_pairs = calculate_overlap(image_data, grid_info, CONFIG['overlap_percentage'])
    if CONFIG.get('debug', False):
        visualize_overlap_regions(image_data, overlap_pairs, grid_info)
        plt.show()
    if tracker:
        tracker.record_step("2. Overlap Region Setup")

    # Step 3 -- SIFT extraction on each ROI
    overlap_features = extract_overlap_features(image_data, overlap_pairs, CONFIG['feature_method'])
    if tracker:
        tracker.record_step("3. SIFT Feature Extraction")

    # Step 4 -- BFMatcher matching
    match_result = match_overlap_features(overlap_features, CONFIG['feature_method'], CONFIG['lowe_ratio'])
    if tracker:
        tracker.record_step("4. BFMatcher Feature Matching")

    # Step 5 -- Translation estimation (2-DOF median, stage-prior filtered)
    homography_results = calculate_homographies_batch(image_data, match_result, CONFIG['reproj_thresh'])
    homography_results = add_nominal_fallback_edges(homography_results, overlap_pairs)
    if tracker:
        tracker.record_step("5. Translation Estimation")

    # Step 5.1 -- Reprojection error report
    if CONFIG.get('evaluate_metrics', False):
        evaluate_homography_reprojection(match_result, homography_results,
                                         output_csv="homography_reprojection_sift_bfm.csv")

    # Step 5.5 -- Overlap quality metrics (PSNR, SSIM, RMSE, NCC)
    if CONFIG.get('use_global_adjustment', True):
        transforms, reference, reachable_images = solve_global_translations(
            homography_results, image_data,
            outlier_tol=CONFIG.get('grid_adjust_outlier_tol', 'adaptive'))
    else:
        transforms, reference, reachable_images = calculate_all_transforms(image_data, homography_results)
    if len(reachable_images) < 0.9 * len(image_data):
        print(f"[WARNING] Only {len(reachable_images)}/{len(image_data)} tiles reachable "
              f"({len(reachable_images) / len(image_data) * 100:.1f}%) -- the mosaic is "
              f"missing a significant fraction of tiles. Check registration quality "
              f"(inlier ratios, TRANSLATION SPREAD) before trusting this result.")
    canvas_w, canvas_h, offset_x, offset_y  = calculate_optimal_canvas(image_data, transforms)
    if CONFIG.get('evaluate_metrics', False):
        evaluate_overlap_metrics(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            homography_results=homography_results,   # adjacent-only -- O(N) not O(N²)
            output_csv="overlap_metrics_sift_bfm.csv",
            visualize_sample=False,
        )
    if tracker and CONFIG.get('evaluate_metrics', False):
        tracker.record_step("5.5 Overlap Quality Metrics Evaluation")

    # Step 5.6 -- per-tile photometric gain compensation. Runs after
    # registration (it reuses the same overlap pair graph) and before
    # blending, which is the only consumer -- blend_panorama() reads
    # the module-level _TILE_GAINS set here.
    if CONFIG.get('gain_compensation', False):
        _TILE_GAINS = solve_tile_gains(image_data, overlap_pairs)
        if tracker:
            tracker.record_step("5.6 Per-Tile Gain Compensation")

    # Step 5.7 -- overlap-driven shading correction. Needs `transforms`
    # (it works in the registered reference frame), so it runs after the
    # global adjustment, and before blending, its only consumer.
    if CONFIG.get('overlap_shading', False):
        _SHADING_CORR = solve_overlap_shading(image_data, transforms, overlap_pairs)
        if tracker:
            tracker.record_step("5.7 Overlap-Driven Shading Correction")

    # Step 6 -- Feather blending
    blended, debug_info = blend_panorama(
        image_data, transforms, reachable_images,
        canvas_w, canvas_h, offset_x, offset_y
    )
    if CONFIG.get('debug', False):
        visualize_blending_debug(debug_info, blended)
        plt.show()

    # ROI crop -- remove black borders
    roi_info = find_roi(blended, min_threshold=1, debug=True)
    final_result, roi_stats = extract_roi(blended, roi_info, padding=10)

    # Save result
    save_path = str(Path(folder_path) / "result_sift_bfm.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {save_path}")
    if CONFIG.get('save_preview', True):
        scale   = CONFIG.get('preview_scale', 0.125)
        preview = cv2.resize(final_result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        preview_path = str(Path(save_path).with_name(Path(save_path).stem + "_preview" + Path(save_path).suffix))
        cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved preview ({scale:.3f}x, {preview.shape[1]}x{preview.shape[0]}): {preview_path}")
    if tracker:
        tracker.record_step("6. Feather Blending, ROI Crop & Saving")

    # Output Benchmark Summary Report
    if tracker:
        tracker.print_summary()

    if CONFIG.get('debug', False):
        plt.figure(figsize=(20, 10))
        plt.imshow(final_result)
        plt.axis('off')
        plt.title('SIFT + BFMatcher Tile Stitching Result', fontsize=16)
        plt.show()