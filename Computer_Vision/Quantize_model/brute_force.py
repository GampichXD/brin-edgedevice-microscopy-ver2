"""
Tile Stitching Pipeline -- Brute-Force Subtraction Variant
=========================================================
Feature extraction/matching    : N/A -- direct pixel-difference search.
                                  For each overlap region the two ROI crops
                                  are turned into float matrices and one is
                                  slid over the other across every integer
                                  (dx, dy) offset in a search window. At each
                                  offset the overlapping sub-matrices are
                                  scored (CONFIG['search_metric']):
                                    'sad' -- mean |a - b|   (subtraction, L1)
                                    'ssd' -- mean (a - b)^2 (subtraction, L2)
                                    'ncc' -- 1 - ZNCC(a, b) (zero-mean
                                             normalized cross-correlation
                                             *instead of* subtraction; each
                                             window is re-centred and re-
                                             scaled first, so a constant
                                             brightness/contrast gap between
                                             the two tiles is not counted as
                                             a mismatch)
                                  All three are costs where 0 == perfect and
                                  lower wins, so the offset with the lowest
                                  score is the estimated translational shift
                                  -- "if the tiles line up, the subtraction
                                  cancels out" (or the correlation peaks). No
                                  keypoints, descriptors, matcher, or FFT
                                  involved.
Homography                     : translation-only 3x3 matrix built directly
                                  from the measured (dx, dy) shift (identical
                                  interface to phase_correlation_stitching.py
                                  -- still a homography_results dict keyed by
                                  (coord1, coord2), still fed into the same
                                  BFS transformation graph). No RANSAC: the
                                  search yields one global shift per pair.
Reprojection error              : N/A -- there are no discrete point
                                  correspondences to reproject. The minimum
                                  per-pixel difference (`error`, lower is
                                  better) and a derived distinctiveness score
                                  (`response`, in [0, 1], higher is better --
                                  how far the winning offset's error sits
                                  below the median error over the whole
                                  search surface) are the per-pair quality
                                  signals in its place; see STEP 5's summary.
NCC evaluation                  : calculate_overlap_ncc (unchanged)
Overlap quality metrics         : PSNR, SSIM, RMSE, NCC (unchanged)
Feather blending                : blend_panorama (unchanged)
ROI crop                        : find_roi / extract_roi (unchanged)

Why brute-force subtraction?
  - The simplest possible registration: no library beyond numpy is needed
    for the core step -- just crop, subtract, take the mean. It is trivially
    easy to reason about and to verify by eye (dump the difference matrix
    for the winning offset and confirm it is near-zero).
  - Like phase correlation it assumes tiles differ by translation only
    (fixed XY stage, negligible rotation/scale/illumination drift). When
    that holds it is exact to +-1 px (or to sub-pixel with the parabolic
    refinement below).
  - Cost is O(search_area * overlap_area) per pair -- much heavier than
    phase correlation's two FFTs. Keep `search_radius` tight (it only needs
    to cover the stage's step-repeatability error, not the whole overlap)
    and use `search_downsample` > 1 for a fast coarse pass. Equivalent in
    spirit to cv2.matchTemplate(..., cv2.TM_SQDIFF), which is the same
    slide-and-subtract done in optimised C -- swap it in if you want speed
    and don't need the explicit error surface.
  - Breaks down under rotation, scale change, or strong illumination drift
    between tiles -- `response` exists to flag when that assumption is being
    violated for a given pair (a flat error surface => low response =>
    the winning offset is barely better than any other => shift is likely
    garbage). Revisit low-confidence pairs with SIFT_BFM.py / SIFT_LG.py /
    SP_LG.py.

Reused verbatim from phase_correlation_stitching.py
--------------------------------------------------
PerformanceTracker, load_image, calculate_overlap, _translation_homography,
the transformation-graph / BFS reference-tile selection /
calculate_all_transforms / calculate_optimal_canvas, the NCC + PSNR/SSIM/
RMSE overlap-metric evaluation (with CSV export), blend_panorama, and the
ROI crop step are IDENTICAL -- everything past STEP 5 only consumes a
`homography_results` dict of 3x3 matrices, agnostic to how those matrices
were estimated.

Installation:  pip install opencv-python numpy scikit-image psutil
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
    Tracks runtime duration and memory consumption (RAM).
    Same shape/API as phase_correlation_stitching.py's tracker -- the
    brute-force subtraction search never touches CUDA/torch, so GPU-memory
    tracking would just be dead weight here.
    """
    def __init__(self, name="Pipeline Benchmark"):
        self.name = name
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_times = {}

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

    def print_summary(self):
        total_time = time.time() - self.start_time
        curr_ram, peak_ram = self._get_ram_usage()

        print("\n" + "=" * 65)
        print(f"  PERFORMANCE & MEMORY BENCHMARK: {self.name}")
        print("=" * 65)
        print(f"  Total Execution Time : {total_time:.3f} seconds ({total_time / 60:.2f} min)")
        print("-" * 65)
        print("  Hardware Acceleration:")
        print(f"    - Registration Method   : Brute-Force Subtraction (SAD/SSD search, CPU-only)")
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
        print("    - GPU VRAM            : N/A (brute-force subtraction runs on CPU)")
        print("=" * 65 + "\n")


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # --- General ---
    'resize_factor':        0.5,
    'overlap_percentage':   0.4,      # expected overlap from stage step size
    'canvas_padding':       50,

    # --- Canvas safety ---
    'max_canvas_megapixels': 100,  # calculate_optimal_canvas() raises instead of
                                    # allocating past this -- a real N-tile grid never
                                    # gets close; hitting it means a bad accumulated
                                    # transform, not a genuinely huge mosaic.

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
    # Toggleable so this file can still be A/B'd against
    # brute_force_subtraction_stitching.py.
    'flatfield_correction': True,  # divide out smooth vignetting
    'flatfield_samples':    60,     # tiles median-stacked to estimate it
    'flatfield_sigma':      25.0,   # heavy blur -- keeps ONLY low frequencies.

    # --- Per-tile gain compensation (STEP 5.6) ---
    # Flat-field removes the WITHIN-tile shading, but it is one spatial
    # profile shared by every tile, so it cannot touch a PER-TILE scalar.
    # Measured on whole_petri_fr (20x34) AFTER flat-field, interior row r8:
    #     G (luminance)  3.8% relative std across tiles
    #     B/G ratio      5.8%
    #     R/G ratio     37.6%   <-- dominant
    # i.e. auto-exposure/auto-white-balance drifted over the 680-tile scan.
    # Because R/G drifts ~10x more than luminance, the residual halo reads
    # as COLOUR banding at tile pitch, not brightness banding -- which is
    # why the solve is per channel.
    'gain_compensation':    True,   # measured best combo (with flatfield_correction):
                                    # ff+gain cut row-pitch banding power roughly in
                                    # half vs ff alone on whole_petri_fr (2.62 vs
                                    # 3.51). Was OFF while this file stayed an A/B
                                    # harness; now the default.
    'gain_per_channel':     True,   # 3 independent solves. A single luminance
                                    # gain leaves the dominant R/G term untouched.
    'gain_reg_lambda':      0.3,    # pull every log-gain toward 0. NOT optional:
                                    # unregularised the solve "explains" real
                                    # scene differences as exposure and produced
                                    # gains spanning 0.178..2.654 (15x) on
                                    # whole_petri_fr, which wrecks the image.
                                    # At 0.3 the fit is barely worse (51 vs 55%
                                    # mismatch reduction) with gains held to
                                    # 0.72..1.46.
    'gain_max_logratio':    0.35,   # reject a pair whose two strips differ by
                                    # more than this in log space (~1.42x) --
                                    # that is content disagreement or
                                    # misregistration, not exposure.
    'gain_min_median':      15.0,   # skip a strip this dark; a ratio between two
                                    # near-black strips is noise, and the red
                                    # channel here means ~20.

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
    'analysis_feather_distance': 0,   # hard seam for analysis/training copy
    'feather_distance':     30,       # legacy alias -- used by blend_panorama
    'normalization_method': 'skimage_histogram_match',
    'min_overlap_area':     500,

    # --- Brute-force subtraction search ---
    # The search runs coarse-to-fine (an image pyramid): a wide, cheap pass on
    # ROIs shrunk by 'coarse_downsample' locates the shift to within a few px,
    # then a tight full-resolution pass refines it. This keeps a large
    # 'search_radius' affordable -- a single-scale O(r^2) sweep at r=150 would
    # be ~50x slower than the r=20 the nominal overlap alignment alone needs.
    'search_radius':        160,  # coarse pass searches integer shifts in
                                  # [-r, +r] px on BOTH axes around the nominal
                                  # overlap alignment. Must cover how far the
                                  # TRUE overlap can deviate from
                                  # 'overlap_percentage' (not just stage
                                  # repeatability). Cost grows as O(r^2), but
                                  # it is paid on the downsampled ROIs.
    'coarse_downsample':    4,    # shrink both ROIs by this factor for the wide
                                  # coarse pass (shift scaled back up after).
                                  # coarse accuracy ~= this many px; the fine
                                  # pass then recovers exact + sub-pixel.
    'fine_radius':          6,    # full-resolution refine pass searches
                                  # [-fine_radius, +fine_radius] px around the
                                  # coarse estimate. Keep >= coarse_downsample.
    'search_step':          1,    # stride of the coarse integer search grid
                                  # (1 = every pixel on the downsampled ROI).
    'search_metric':        'ncc',  # how each candidate overlap is scored. All
                                    # three become a COST (0 == perfect, lower
                                    # wins) so the same search machinery serves
                                    # every one:
                                    #   'sad' = mean |a - b|     (subtraction, L1)
                                    #   'ssd' = mean (a - b)^2    (subtraction, L2,
                                    #           cv2.TM_SQDIFF objective)
                                    #   'ncc' = 1 - ZNCC(a, b)    (cross-correlation
                                    #           *instead of* subtraction: zero-means
                                    #           and unit-scales EACH window first,
                                    #           so per-window brightness/contrast
                                    #           gaps a raw subtraction leaves behind
                                    #           don't count as mismatch. Slower --
                                    #           two extra reductions per window.)
    'preprocess':           'highpass',
                                  # applied ONCE to each ROI before the slide-
                                  # and-subtract, so "aligned => difference ~ 0"
                                  # actually holds on real microscope tiles.
                                  #   'highpass' -- x - GaussianBlur(x); kills
                                  #                 vignetting / low-frequency
                                  #                 shading. THE measured best
                                  #                 for this matcher.
                                  #   'none'     -- raw grayscale, kept only as
                                  #                 the A/B baseline.
                                  # CLAHE and z-score used to be options here and
                                  # were removed after measurement (see
                                  # PREPROCESSING_STUDY.md):
                                  #   * z-score is a NO-OP under search_metric
                                  #     'ncc' -- _pair_cost() already zero-means
                                  #     and unit-scales EACH window, which is
                                  #     strictly better than one global rescale
                                  #     of the whole ROI. Measured divergence
                                  #     1.9e-05 px over 80 real pairs.
                                  #   * CLAHE scored worse than highpass on both
                                  #     a ground-truth test (p90 error 0.10 vs
                                  #     0.07 px) and on real pairs (response
                                  #     0.371 vs 0.402). It is nonlinear and
                                  #     tile-local, so it maps the SAME physical
                                  #     pixels differently in the two crops.
    'highpass_sigma':       12.0, # Gaussian sigma (px) for the 'highpass' step.
                                  # ~ the scale of the shading to remove; larger
                                  # keeps more structure, smaller flattens harder.
                                  # NOTE: applied AFTER 'resize_factor', so at
                                  # the default 0.5 this is ~24 px at capture
                                  # scale.
    'registration_channel': 'b',
                                  # which single channel the shift search runs on
                                  # (blending/output always stays full RGB):
                                  #   'gray' -- 0.299R + 0.587G + 0.114B (luma)
                                  #   'r' / 'g' / 'b' -- one raw colour channel
                                  #   'auto' -- per tile, pick the channel with
                                  #             the largest post-highpass detail
                                  #             (std of x - GaussianBlur(x))
                                  # Use a raw channel when the tiles have a
                                  # strong colour cast that buries the texture
                                  # in luma -- e.g. the blue-cast 10x10_1 set
                                  # registers best on 'b' with 'highpass'.
    'min_overlap_fraction': 0.5,  # at a candidate shift, ignore it unless the
                                  # two ROIs still overlap over >= this fraction
                                  # of each axis -- stops tiny-overlap offsets
                                  # from winning with a spuriously low mean error.
    'subpixel_refine':      True, # parabolic fit on the fine-pass error surface
                                  # around the integer minimum -> sub-pixel (dx, dy).
    'match_confidence_threshold': 0.02,  # below this `response`, treat the pair as
                                         # unreliable (flat error surface => the
                                         # winning offset is barely better than the
                                         # rest => rotation/illumination/focus drift)
                                         # and drop it. `response` is
                                         # (median_error - min_error) / median_error,
                                         # clipped to [0, 1]; set NEGATIVE to force
                                         # every pair through for debugging -- watch
                                         # for the warning register_overlap_pairs()
                                         # prints when responses are uniformly low.
    'shift_sign':           1,    # flip to -1 if the stitched mosaic comes out
                                  # systematically offset -- see _translation_homography()

    # --- Grid-consistency / outlier rejection (fixed XY stage prior) ---
    # A rigid stage means EVERY horizontal pair should report the same dx and
    # every vertical pair the same dy. These two options use that prior to
    # kill the failure modes the per-pair `response` score cannot catch:
    #   * a search that clamped at the edge of its window (true overlap far
    #     from 'overlap_percentage') -- returns a plausible-looking but
    #     hundreds-of-px-short shift, which then opens a gap / black band;
    #   * per-pair sub-pixel noise that accumulates into staircase drift as
    #     tiles get chained further from the reference along the BFS path.
    'reject_clamped_pairs':      True,  # drop any pair whose coarse search
                                        # winner sat on the +-search_radius
                                        # boundary (its true shift is outside
                                        # the searched range). Distinct from a
                                        # flat cost surface; the response score
                                        # alone can't tell them apart.
    'enforce_grid_consistency':  True,  # after registration, snap any pair whose
                                        # (dx, dy) deviates from its direction's
                                        # median by > 'grid_consistency_tol' px
                                        # to that median. Cheap stand-in for a
                                        # global bundle adjustment: removes
                                        # clamped/outlier pairs AND most of the
                                        # BFS-chain drift. Well justified for a
                                        # fixed-stage grid -- state it as such.
    'grid_consistency_tol':      'adaptive',   # px deviation from the per-direction
                                        # median tolerated before a pair's shift is
                                        # snapped to the median. Reverted from
                                        # 'adaptive' after a direct A/B on a real
                                        # 540-tile run: this rig's per-pair dx/dy
                                        # noise (std ~17-45px) is measurement noise,
                                        # not real stage variation -- 3*1.4826*MAD
                                        # on that spread computes a tolerance
                                        # (100-150px) almost as wide as the data's
                                        # entire range, so 'adaptive' snapped only
                                        # 19% of pairs (vs. 98% at tol=3.0) and let
                                        # the noise feed straight into the
                                        # least-squares solve: residual mean
                                        # 0.08px -> 22.23px, ROI efficiency
                                        # 95.4% -> 84.9%, visibly worse mosaic. The
                                        # rigid-stage prior this function is built
                                        # on wants an aggressive, tight tolerance,
                                        # not a permissive one -- 'adaptive' is
                                        # still available (a number or 'adaptive'
                                        # both work) if a specific rig's per-pair
                                        # jitter genuinely exceeds this.
    'use_global_adjustment':     True,  # position every tile with ONE least-
                                        # squares solve over all pairwise
                                        # constraints (see solve_global_translations
                                        # below) instead of chaining transforms along a
                                        # BFS path from the reference. Chaining
                                        # lets error accumulate with distance
                                        # from the reference (staircase drift at
                                        # the mosaic corners); the global solve
                                        # spreads it evenly. Set False for the
                                        # old BFS behaviour (calculate_all_transforms).
    'nominal_fallback_weight':   0.1,   # weight (vs. 1.0 for a measured edge) given
                                        # to nominal-fallback edges (see
                                        # add_nominal_fallback_edges) in the
                                        # least-squares solve.
    'use_nominal_fallback':      True,  # if a geometrically-expected adjacent pair
                                        # fails registration entirely (low response,
                                        # clamped search), substitute a nominal shift
                                        # derived from its overlap-region geometry
                                        # instead of dropping the edge -- keeps the
                                        # transformation graph connected instead of
                                        # fragmenting into a handful of reachable
                                        # tiles (see correction.txt: this is the
                                        # failure mode that turned 100 tiles into 2).

    # --- Device ---
    'debug': False,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': True,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
    'enable_benchmark': True,       # Toggle Performance & Memory Tracker benchmarking
    'verbose_pair_metrics': False,  # Print PSNR/SSIM/NCC per pair (off = final summary only)
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

    def _coords_of(name):
        """(x, y) grid/stage coords from a tile filename, or None."""
        m = focused_pattern.match(name)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        m = tile_pattern.match(name)
        if m:
            return (float(m.group(2)), float(m.group(1)))   # col -> x, row -> y
        m = embedded_pattern.search(name)
        if m:
            return (float(m.group(2)), float(m.group(1)))   # col -> x, row -> y
        return None

    # Resolve the tile list ONCE, up front. Filtering by "does the filename
    # parse as tile coordinates" matters beyond tidiness: this folder also
    # contains previously-saved result mosaics (result_*.jpg, thousands of px
    # on a side). A bare glob("*.jpg") pulls those into the flat-field median
    # stack -- which fails outright on the shape mismatch.
    tile_files = [(p, _coords_of(p.name)) for p in sorted(folder_path.glob("*.jpg"))]
    tile_files = [(p, c) for p, c in tile_files if c is not None]

    # Build the flat-field ONCE from a sample of real tiles,
    # before any tile is stored (no-op when the toggle is off).
    prepare_photometric_correction([p for p, _ in tile_files], resize_factor)

    image_data = {}
    skipped_empty = []
    for file_path, coords in tile_files:
        name = file_path.name

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

def calculate_overlap(image_data, grid_info, overlap_percentage=0.2):
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
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3 + 1))
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
            ec    = ('darkgreen' if ov['role'] == 'source' else 'darkblue') \
                    if ov['direction'] == 'horizontal' \
                    else ('darkred' if ov['role'] == 'source' else 'darkorange')
            rect  = plt.Rectangle((r['x'], r['y']), r['width'], r['height'],
                                   lw=2, edgecolor=ec, facecolor=color, alpha=0.4)
            ax.add_patch(rect)
            lbl = ('H' if ov['direction'] == 'horizontal' else 'V') + \
                  ('S' if ov['role'] == 'source' else 'T')
            ax.text(r['x'] + r['width'] // 2, r['y'] + r['height'] // 2, lbl,
                    color='white', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        ax.set_title(f'{coords}', fontsize=16)

    plt.suptitle(f'Overlap Regions ({len(overlap_pairs)} pairs)', fontsize=20)
    plt.tight_layout()
    return fig


# ============================================================
# STEP 3-4 : Brute-Force Subtraction Registration (combined extract+match stage)
# ============================================================
# There is no separate "extract features" / "match features" stage -- the
# two overlap crops are converted to float matrices and one is slid over the
# other; at every candidate (dx, dy) the overlapping sub-matrices are scored
# and the best offset wins.
#
# The score is always a COST (0 == perfect match, lower wins) so the same
# minimise-and-parabola machinery serves every metric:
#   'sad' / 'ssd' -- SUBTRACTION: mean |a - b| or mean (a - b)^2. The
#                    difference matrix goes to ~0 where the tiles line up.
#   'ncc'         -- CROSS-CORRELATION instead of subtraction: 1 - ZNCC(a, b),
#                    where ZNCC re-centres (subtract mean) and re-scales
#                    (divide by std) EACH candidate window before comparing.
#                    That makes it blind to per-window brightness/contrast
#                    differences a plain subtraction cannot cancel, at the
#                    cost of two extra passes over each window. 1 - ZNCC is
#                    0 for a perfect (positively correlated) match, 1 for
#                    uncorrelated, 2 for perfect anti-correlation.

def to_float_gray(img):
    """Single-channel float32 view of an ROI crop."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32)


def _preprocess_roi(img, method, sigma):
    """
    Condition an ROI crop ONCE so that the slide-and-score actually reaches
    its minimum where the tiles align.

      'highpass' -- subtract a Gaussian-blurred copy; removes microscope
                    vignetting / low-frequency shading. This is what the
                    flat-field step (STEP 0.5) cannot do: flat-field is ONE
                    profile shared by every tile, whereas the residual shading
                    inside a single overlap crop still tilts the cost surface.
      'none'     -- raw grayscale. Kept only so the A/B baseline is still
                    reachable; measurably worse (see PREPROCESSING_STUDY.md:
                    real-pair response 0.345 vs 0.402, and 5% of ground-truth
                    trials missing by >1 px instead of 0%).

    Without the highpass, two perfectly registered tiles can still differ by
    tens of grey levels across the overlap, leaving the cost surface flat and
    the search rudderless.
    """
    x = to_float_gray(img)
    if 'highpass' in method:
        x = x - cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return x


def _pair_cost(sub_a, sub_b, metric):
    """
    Score one aligned window pair as a COST (0 == perfect match, lower wins).

        'sad' -> mean |a - b|            (subtraction, L1, robust to hot pixels)
        'ssd' -> mean (a - b)^2          (subtraction, L2, cv2.TM_SQDIFF objective)
        'ncc' -> 1 - ZNCC(a, b)          (cross-correlation used *instead of*
                 subtraction: each window is zero-meaned and unit-scaled first,
                 so a constant brightness/contrast difference between the two
                 tiles does not register as a mismatch. 0 = perfect positive
                 correlation, 1 = uncorrelated, 2 = perfect anti-correlation.)

    A window that is perfectly flat (zero variance) has undefined ZNCC, so it
    is reported as the worst finite cost (1.0).
    """
    if metric == 'ncc':
        a0 = sub_a - sub_a.mean()
        b0 = sub_b - sub_b.mean()
        denom = np.sqrt(float(np.sum(a0 * a0)) * float(np.sum(b0 * b0)))
        if denom < 1e-12:
            return 1.0
        return 1.0 - float(np.sum(a0 * b0) / denom)
    if metric == 'ssd':
        d = sub_a - sub_b
        return float(np.mean(d * d))
    # 'sad'
    return float(np.mean(np.abs(sub_a - sub_b)))


def _sweep(a, b, cx, cy, radius, step, metric, min_overlap_fraction):
    """
    One single-scale sweep of *b* over *a* across every integer (sx, sy) in
    cx +- radius, cy +- radius (stride *step*). Each candidate overlap is
    scored by _pair_cost(..., metric) -- a subtraction ('sad'/'ssd') or a
    cross-correlation ('ncc'), both returned as a cost where lower is better.

    Returns (err_surface, best_sx, best_sy, best_err).
    """
    h, w = a.shape
    xs = range(cx - radius, cx + radius + 1, step)
    ys = range(cy - radius, cy + radius + 1, step)
    err_surface = {}
    best_sx = cx
    best_sy = cy
    best_err = np.inf

    for sy in ys:
        ay0, ay1 = max(0, sy), h + min(0, sy)
        by0, by1 = max(0, -sy), h + min(0, -sy)
        if (ay1 - ay0) < min_overlap_fraction * h:
            continue
        for sx in xs:
            ax0, ax1 = max(0, sx), w + min(0, sx)
            bx0, bx1 = max(0, -sx), w + min(0, -sx)
            if (ax1 - ax0) < min_overlap_fraction * w:
                continue

            err = _pair_cost(a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1], metric)

            err_surface[(sx, sy)] = err
            if err < best_err:
                best_err, best_sx, best_sy = err, sx, sy

    return err_surface, best_sx, best_sy, best_err


def _brute_force_shift(roi1, roi2, search_radius, coarse_downsample, fine_radius,
                       search_step, metric, min_overlap_fraction, subpixel,
                       preprocess='highpass', highpass_sigma=12.0):
    """
    Coarse-to-fine registration of *roi2* against *roi1* -- slide-and-score,
    where the score is a subtraction ('sad'/'ssd') or a cross-correlation
    ('ncc') per CONFIG['search_metric']; see _pair_cost().

    1. COARSE : shrink both ROIs by *coarse_downsample*, sweep +-search_radius
                (in full-res px) around the nominal overlap alignment. Cheap
                because it runs on the small ROIs; locates the shift to
                ~coarse_downsample px.
    2. FINE   : full resolution, sweep +-fine_radius px around the coarse
                estimate to pin the exact integer shift.
    3. SUBPIX : parabola through the fine-pass error minimum and its
                neighbours -> sub-pixel (dx, dy).

    Returns (dx, dy, error, response, clamped)
      error    : the winning offset's match cost -- mean |a - b| ('sad'),
                 mean (a - b)^2 ('ssd'), or 1 - ZNCC ('ncc'). 0 == identical;
                 a small positive number for a good real-world match.
      response : (median_error - min_error) / median_error over the FINE
                 surface, clipped to [0, 1]. High => the winning offset stands
                 out sharply from neighbouring offsets (confident). Low =>
                 flat cost surface, the shift is unreliable.
      clamped  : True if the COARSE winner sat on the +-search_radius edge --
                 the true shift is outside the searched range (usually
                 'overlap_percentage' is wrong), so (dx, dy) is short by
                 however far past the edge the real minimum lies. `response`
                 does NOT catch this on its own.
    """
    a = _preprocess_roi(roi1, preprocess, highpass_sigma)
    b = _preprocess_roi(roi2, preprocess, highpass_sigma)
    step = max(1, int(search_step))

    # --- 1. coarse pass on downsampled ROIs -------------------------------
    ds = max(1, int(coarse_downsample))
    a_c = a[::ds, ::ds] if ds > 1 else a
    b_c = b[::ds, ::ds] if ds > 1 else b
    radius_c = max(1, int(round(search_radius / ds)))
    _, csx, csy, _ = _sweep(a_c, b_c, 0, 0, radius_c, step,
                            metric, min_overlap_fraction)
    coarse_sx, coarse_sy = csx * ds, csy * ds

    # coarse winner pinned to the window edge -> the real minimum is further
    # out than the search could reach (see 'reject_clamped_pairs').
    clamped = (abs(csx) >= radius_c - step) or (abs(csy) >= radius_c - step)

    # --- 2. fine pass at full resolution --------------------------------
    fr = max(int(fine_radius), ds)
    err_surface, best_sx, best_sy, best_err = _sweep(
        a, b, coarse_sx, coarse_sy, fr, 1, metric, min_overlap_fraction)

    if not err_surface:
        return 0.0, 0.0, float('inf'), 0.0, True

    errs = np.fromiter(err_surface.values(), dtype=np.float64)
    med  = float(np.median(errs))
    response = float(np.clip((med - best_err) / (med + 1e-9), 0.0, 1.0))

    dx, dy = float(best_sx), float(best_sy)
    if subpixel:
        dx += _parabolic_offset(err_surface, best_sx, best_sy, 1, axis='x')
        dy += _parabolic_offset(err_surface, best_sx, best_sy, 1, axis='y')

    return dx, dy, best_err, response, clamped


def _parabolic_offset(err_surface, sx, sy, step, axis):
    """Sub-pixel correction from a parabola through the error minimum and its
    two neighbours along *axis*. Returns 0 if a neighbour is missing (edge of
    the search window) or the parabola is degenerate."""
    if axis == 'x':
        em = err_surface.get((sx - step, sy))
        e0 = err_surface.get((sx, sy))
        ep = err_surface.get((sx + step, sy))
    else:
        em = err_surface.get((sx, sy - step))
        e0 = err_surface.get((sx, sy))
        ep = err_surface.get((sx, sy + step))
    if em is None or e0 is None or ep is None:
        return 0.0
    denom = em - 2.0 * e0 + ep
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (em - ep) / denom, -1.0, 1.0)) * step


def register_overlap_pairs(image_data, overlap_pairs):
    """
    Brute-force-subtract every overlap ROI crop against its counterpart.

    Returns
    -------
    match_result : dict
        {(coord1, coord2): {
            'dx', 'dy', 'error', 'response', 'clamped',
            'direction', 'region1', 'region2'
        }}
    """
    match_result    = {}
    low_confidence  = []
    threshold       = CONFIG['match_confidence_threshold']

    for i, pair in enumerate(overlap_pairs, 1):
        coord1, coord2 = pair['coord1'], pair['coord2']
        r1, r2         = pair['region1'], pair['region2']
        direction      = pair['direction']

        try:
            gray1 = image_data[coord1]['image_gray']
            gray2 = image_data[coord2]['image_gray']
            roi1 = gray1[r1['y']:r1['y'] + r1['height'], r1['x']:r1['x'] + r1['width']]
            roi2 = gray2[r2['y']:r2['y'] + r2['height'], r2['x']:r2['x'] + r2['width']]

            if roi1.size == 0 or roi2.size == 0:
                print(f"[SKIP] {coord1}<->{coord2}: empty overlap ROI")
                continue

            if roi1.shape != roi2.shape:
                # the slide-and-subtract needs identically-shaped inputs;
                # crop both to their smaller common shape (can happen at
                # grid edges with unevenly sized tiles).
                h = min(roi1.shape[0], roi2.shape[0])
                w = min(roi1.shape[1], roi2.shape[1])
                roi1 = roi1[:h, :w]
                roi2 = roi2[:h, :w]

            dx, dy, error, response, clamped = _brute_force_shift(
                roi1, roi2,
                search_radius=CONFIG['search_radius'],
                coarse_downsample=CONFIG['coarse_downsample'],
                fine_radius=CONFIG['fine_radius'],
                search_step=CONFIG['search_step'],
                metric=CONFIG['search_metric'],
                min_overlap_fraction=CONFIG['min_overlap_fraction'],
                subpixel=CONFIG['subpixel_refine'],
                preprocess=CONFIG['preprocess'],
                highpass_sigma=CONFIG['highpass_sigma'],
            )
            if response < threshold:
                low_confidence.append((coord1, coord2, response))

            match_result[(coord1, coord2)] = {
                'dx': dx, 'dy': dy, 'error': error, 'response': response,
                'clamped': bool(clamped),
                'direction': direction,
                'region1': r1, 'region2': r2,
            }
            print(f"Pair {i}/{len(overlap_pairs)}: {coord1} <-> {coord2} "
                  f"| shift=({dx:.2f},{dy:.2f}) error={error:.3f} "
                  f"response={response:.3f}{' [CLAMPED]' if clamped else ''}")
        except Exception as e:
            print(f"[ERROR] Brute-force subtraction failed {coord1} <-> {coord2}: {e}")
            traceback.print_exc()

    _print_registration_summary(match_result, len(overlap_pairs), low_confidence)
    return match_result


def _print_registration_summary(match_result, total_pairs, low_confidence):
    print(f"\n[SUMMARY] BRUTE-FORCE SUBTRACTION REGISTRATION:")
    print(f"   Total pairs attempted   : {total_pairs}")
    print(f"   Successful pairs        : {len(match_result)}")
    if match_result:
        responses = [r['response'] for r in match_result.values()]
        errors    = [r['error'] for r in match_result.values()]
        metric    = CONFIG['search_metric']
        cost_label = "1 - ZNCC" if metric == 'ncc' else f"{metric} diff"
        print(f"   Metric used             : {metric}")
        print(f"   Avg response (confidence): {np.mean(responses):.3f}")
        print(f"   Avg min cost ({cost_label:8s}): {np.mean(errors):.3f}")

        # Per-direction shift distribution. On a fixed XY stage every
        # horizontal pair should return nearly the same dx and every vertical
        # pair nearly the same dy -- a large std, or values piled up at the
        # search-window edge, points straight at the failing pairs.
        print("   Shift distribution (fixed-stage prior: std should be small):")
        for direction, axis in (('horizontal', 'dx'), ('vertical', 'dy')):
            vals = np.array([r[axis] for r in match_result.values()
                             if r['direction'] == direction], dtype=np.float64)
            if vals.size:
                print(f"     {direction:10s} {axis}: median={np.median(vals):7.1f} "
                      f"std={vals.std():6.1f} min={vals.min():7.1f} max={vals.max():7.1f} "
                      f"(n={vals.size})")
        n_clamped = sum(1 for r in match_result.values() if r.get('clamped'))
        if n_clamped:
            print(f"   [WARNING] {n_clamped} pair(s) CLAMPED at the +-"
                  f"{CONFIG['search_radius']}px search-window edge -- the true "
                  f"shift is outside the searched range, so those shifts are "
                  f"short by hundreds of px (=> gaps / black bands). "
                  f"'overlap_percentage' ({CONFIG['overlap_percentage']}) is "
                  f"almost certainly wrong; measure the real overlap and set it.")
    if low_confidence:
        threshold = CONFIG['match_confidence_threshold']
        print(f"   [WARNING] {len(low_confidence)} pairs had low confidence "
              f"(response < {threshold}) -- the error surface is nearly flat, "
              f"so the winning offset is barely better than any other. Likely "
              f"violates the pure-translation assumption (rotation/illumination/"
              f"focus drift). Consider SIFT_BFM.py / SIFT_LG.py / SP_LG.py for "
              f"these pairs.")
        for c1, c2, resp in low_confidence[:10]:
            print(f"    - {c1}<->{c2}: response={resp:.3f}")


def enforce_grid_consistency(match_result, tol=None):
    """
    Fixed-stage prior: every horizontal pair should report the same dx and
    every vertical pair the same dy. Take the per-direction median (dx, dy)
    and snap any pair deviating from it by more than a tolerance to that
    median.

    *tol*: a fixed px tolerance, 'adaptive' (default, via CONFIG) to derive a
    robust per-direction, per-axis tolerance from the measured spread itself
    (median absolute deviation -- see _adaptive_tol) instead of a fixed
    guess, or a number to pin it. A fixed tolerance increasingly mis-snaps as
    grid extent (and the true per-pair variation) changes with grid size --
    see correction.txt.

    This is a cheap, well-justified stand-in for a global bundle adjustment:
      * it removes clamped / outlier pairs (a shift hundreds of px off is far
        past the tolerance, so it gets replaced by the good median) without
        dropping the edge from the transformation graph;
      * it strips the per-pair sub-pixel noise that would otherwise
        accumulate into staircase drift along the BFS chain.

    Mutates *match_result* in place (adds 'dx_raw'/'dy_raw'/'snapped_to_median'
    to changed pairs, clears their 'clamped' flag since the median is now the
    trusted value) and returns it.
    """
    if not match_result:
        return match_result
    tol = tol if tol is not None else CONFIG.get('grid_consistency_tol', 'adaptive')

    by_dir = defaultdict(list)
    for key, info in match_result.items():
        by_dir[info['direction']].append(key)

    n_snapped = 0
    print(f"\n[SUMMARY] GRID CONSISTENCY (fixed-stage prior, tol={tol}):")
    for direction, keys in by_dir.items():
        dxs = [match_result[k]['dx'] for k in keys]
        dys = [match_result[k]['dy'] for k in keys]
        med_dx = float(np.median(dxs))
        med_dy = float(np.median(dys))
        if tol == 'adaptive':
            dx_tol = _adaptive_tol(dxs)
            dy_tol = _adaptive_tol(dys)
        else:
            dx_tol = dy_tol = float(tol)
        snapped_here = 0
        for k in keys:
            info = match_result[k]
            if abs(info['dx'] - med_dx) > dx_tol or abs(info['dy'] - med_dy) > dy_tol:
                info['dx_raw'], info['dy_raw'] = info['dx'], info['dy']
                info['dx'], info['dy'] = med_dx, med_dy
                info['snapped_to_median'] = True
                info['clamped'] = False
                snapped_here += 1
        n_snapped += snapped_here
        print(f"   {direction:10s}: median shift=({med_dx:7.2f},{med_dy:7.2f}) "
              f"tol=({dx_tol:.1f},{dy_tol:.1f})px from {len(keys)} pairs, "
              f"{snapped_here} snapped")
    print(f"   Total snapped to median : {n_snapped}/{len(match_result)}")
    return match_result


# ============================================================
# ============================================================
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


# STEP 5 : Homography from Brute-Force Shift  (no RANSAC)
# ============================================================

def _translation_homography(region1, region2, dx, dy):
    """
    Build the 3x3 translation-only matrix mapping a pixel in coord1's full
    tile image to the corresponding pixel in coord2's full tile image, from
    the measured brute-force shift between their two ROI crops.

    Sign convention: assumes the search returns (dx, dy) such that content at
    local position q in roi1 appears at (q - dx, q - dy) in roi2 (i.e. roi2
    shifted by (dx, dy) lines up with roi1). If the stitched mosaic comes out
    systematically offset (tiles overlapping too much, or gapping apart,
    consistently in the same direction), flip CONFIG['shift_sign'] to -1.
    """
    sign = CONFIG.get('shift_sign', 1)
    tx = (region2['x'] - region1['x']) - sign * dx
    ty = (region2['y'] - region1['y']) - sign * dy
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


def calculate_homographies_batch(image_data, match_result, response_threshold=None):
    """
    Build a translation-only 3x3 transform directly from each pair's measured
    (dx, dy) shift -- no RANSAC needed, since the brute-force search already
    produces a single global estimate per pair rather than a point cloud to
    filter outliers from. `response` (how far the winning offset's error sits
    below the median error over the whole search surface, in [0, 1]) is used
    as the accept/reject signal in its place.
    """
    response_threshold = response_threshold if response_threshold is not None else CONFIG['match_confidence_threshold']
    reject_clamped = CONFIG.get('reject_clamped_pairs', True)
    homography_results = {}
    success = failed = 0

    for (coord1, coord2), info in match_result.items():
        try:
            if info['response'] < response_threshold:
                print(f"[SKIP] {coord1}<->{coord2}: response {info['response']:.3f} < threshold")
                failed += 1
                continue

            if reject_clamped and info.get('clamped'):
                print(f"[SKIP] {coord1}<->{coord2}: search clamped at window edge "
                      f"(true shift outside +-{CONFIG['search_radius']}px)")
                failed += 1
                continue

            H = _translation_homography(info['region1'], info['region2'],
                                        info['dx'], info['dy'])
            homography_results[(coord1, coord2)] = {
                'homography_matrix': H,
                'inliers':           1,     # one global estimate per pair,
                'total_matches':     1,     # not a point cloud
                'inlier_ratio':      1.0,
                'response':          info['response'],  # brute-force quality signal
                'error':             info['error'],     # winning offset's per-pixel diff
                'direction':         info['direction'],
                'status':            None,
                'source':            'measured',
                'region1':           info['region1'],
                'region2':           info['region2'],
            }
            print(f"   OK {coord1}<>{coord2}: shift=({info['dx']:.2f},{info['dy']:.2f}) "
                  f"error={info['error']:.3f} response={info['response']:.3f}")
            success += 1
        except Exception as e:
            print(f"[ERROR] Homography failed {coord1}<>{coord2}: {e}")
            failed += 1

    total = success + failed
    print(f"\n[SUMMARY] HOMOGRAPHY (Brute-Force Subtraction):")
    print(f"   Success : {success}")
    print(f"   Failed  : {failed}")
    if total:
        print(f"   Rate    : {success / total * 100:.1f}%")
    if homography_results:
        resp = [r['response'] for r in homography_results.values()]
        print(f"   Avg response: {np.mean(resp):.3f} (min {min(resp):.3f}, max {max(resp):.3f})")
    _print_translation_stats(homography_results)
    return homography_results


def _print_translation_stats(homography_results):
    """Print the measured tx/ty spread per direction (correction.txt).

    A fixed-XY stage should report nearly the same translation for every
    horizontal pair, and the same for every vertical pair. A tight std here
    (roughly stage jitter, e.g. 8-15px) means registration is healthy. A std
    in the hundreds means registration itself is broken -- look at `response`
    and clamped-pair counts rather than re-tuning tolerances.
    """
    if not homography_results:
        return
    by_dir = defaultdict(list)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        by_dir[_ga_pair_direction(c1, c2)].append((float(H[0, 2]), float(H[1, 2])))

    print(f"\n[SUMMARY] TRANSLATION SPREAD (per direction):")
    for d, rows in by_dir.items():
        tx = np.array([r[0] for r in rows])
        ty = np.array([r[1] for r in rows])
        print(f"   {d:10s}: tx med={np.median(tx):7.1f} std={tx.std():6.1f} | "
              f"ty med={np.median(ty):7.1f} std={ty.std():6.1f} | n={len(rows)}")


def add_nominal_fallback_edges(homography_results, overlap_pairs, verbose=True):
    """
    For every geometrically-expected adjacent pair (from calculate_overlap())
    that failed registration entirely (low response, clamped search window),
    inject a nominal-shift homography derived from the pair's own
    overlap-region geometry, instead of just dropping the edge. A dropped
    edge fragments the transformation graph -- exactly what turns "100
    tiles" into "2 reachable tiles" once enough pairs fail (see
    correction.txt: graph percolation collapses superlinearly with grid
    size). Nominal edges are tagged 'source': 'nominal_fallback' so callers
    can report how many were needed, and so solve_global_translations
    weights them lower than measured edges in the least-squares solve.
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
            'response':          0.0,
            'error':             float('inf'),
            'direction':         pair['direction'],
            'status':            None,
            'source':            'nominal_fallback',
            'region1':           r1,
            'region2':           r2,
        }
        added += 1

    if verbose and added:
        print(f"[FALLBACK] added {added} nominal-shift edge(s) for pair(s) that "
              f"failed registration, to keep the transformation graph connected")
    return homography_results


# NOTE -- Reprojection error (STEP 5.1 in the feature-based pipelines) has no
# equivalent here: the brute-force search doesn't produce discrete point
# correspondences to reproject through the estimated H, only a single global
# shift per pair. `response` / `error`, reported in the homography summary
# above, are this pipeline's quality signals instead.


# ============================================================
# STEP 5 (cont.) : Transformation Graph  (unchanged)
# ============================================================

def build_transformation_graph(homography_results):
    graph = defaultdict(dict)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        graph[c1][c2] = H
        graph[c2][c1] = np.linalg.inv(H)
    return dict(graph)


def _find_path_bfs(source, target, graph):
    """BFS shortest path. Uses deque for O(1) popleft (vs O(N) list.pop(0))."""
    if source == target:
        return [source]
    visited = {source}
    queue   = deque([(source, [source])])
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
    prefer the one closest to the geometric centre of the grid. This
    minimises the worst-case BFS chain length and therefore reduces
    homography drift for large (e.g. 5x5+) mosaics.
    """
    counts   = {c: len(nb) for c, nb in graph.items()}
    max_conn = max(counts.values())
    candidates = [c for c, n in counts.items() if n == max_conn]
    cx = sum(c[0] for c in graph) / len(graph)
    cy = sum(c[1] for c in graph) / len(graph)
    reference = min(candidates,
                    key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    print(f"Reference image: {reference} "
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
                print(f"   OK {coord} -> {reference}")
            else:
                print(f"   FAIL {coord} -> {reference}: no path")

    print(f"Reachable: {len(reachable_images)}/{len(image_data)}")
    return transforms, reference, reachable_images


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

_SHADING_CORR = None   # np.ndarray (h, w, 3) float32 multiplicative correction, or None


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
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
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

    print(f"   Canvas  : {canvas_w} x {canvas_h}")
    print(f"   Offset  : ({offset_x}, {offset_y})")
    return canvas_w, canvas_h, offset_x, offset_y


# ============================================================
# STEP 5.2 : NCC Calculation  (unchanged)
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
    """Normalized cross-correlation (NCC) inside the overlap region between
    two warped images."""
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
        print(f"[WARNING] Error calculating NCC for {coord1}<>{coord2}: {e}")
        return None


# ============================================================
# STEP 5.5 : Overlap Quality Metrics  (unchanged)
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
    img2 is always matched to img1's histogram. The direction is deterministic
    within a run, but pair ordering from homography_results.keys() may not
    always place the same physical tile as coord1.
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
                i1[:, :, c] = clahe.apply(i1[:, :, c].astype(np.uint8))
                i2[:, :, c] = clahe.apply(i2[:, :, c].astype(np.uint8))
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
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, T_adj).reshape(-1, 2)
    x0 = max(0, int(np.floor(wc[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(wc[:, 1].min())) - pad)
    x1 = min(canvas_w, int(np.ceil(wc[:, 0].max())) + pad)
    y1 = min(canvas_h, int(np.ceil(wc[:, 1].max())) + pad)
    return x0, y0, x1, y1


def detect_pairwise_overlap(image_data, transforms, coord1, coord2,
                             canvas_w, canvas_h, offset_x, offset_y):
    """Windowed warp: both tiles are warped into the union of their own
    bounding boxes, not the full canvas. Their overlap (intersection) is
    always fully contained within that union, so this is exact -- just
    cheaper than a full-canvas warp for two tiles on a large mosaic."""
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

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
    T_shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float32)

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
        ov_r1[:, :, c] *= ov
        ov_r2[:, :, c] *= ov

    rows, cols = np.where(ov)
    r0, r1_    = rows.min(), rows.max()
    c0, c1_    = cols.min(), cols.max()
    return (ov_r1[r0:r1_ + 1, c0:c1_ + 1],
            ov_r2[r0:r1_ + 1, c0:c1_ + 1],
            ov[r0:r1_ + 1, c0:c1_ + 1],
            int(ov.sum()))


def evaluate_overlap_metrics(image_data, transforms, reachable_images,
                              canvas_w, canvas_h, offset_x, offset_y,
                              homography_results=None,
                              output_csv="overlap_metrics_brute_force.csv",
                              norm_method=None,
                              visualize_sample=False):
    """Compute PSNR, SSIM, RMSE, NCC for overlapping tile pairs.

    When *homography_results* is provided (recommended), only the adjacent
    registered pairs are evaluated -- O(N) instead of O(N^2) over all
    reachable-image combinations.
    """
    norm_method = norm_method or CONFIG['normalization_method']
    results     = []

    if homography_results is not None:
        pairs = list(homography_results.keys())
    else:
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
                print(f"OK {coord1} <> {coord2}: PSNR={psnr:.2f} SSIM={s:.3f} "
                      f"RMSE={rmse:.2f} NCC={ncc if ncc is not None else 'N/A'} | {n_pixels}px")

        except Exception as e:
            print(f"[ERROR] {coord1} <> {coord2}: {e}")

    _save_csv(results, output_csv,
              ['tile1_coord', 'tile2_coord', 'overlap_pixels', 'overlap_width',
               'overlap_height', 'psnr', 'ssim', 'rmse', 'ncc'])
    _print_metric_summary(results)
    return results


def _save_csv(results, path, fieldnames):
    if not results:
        print("[WARN] No results to save.")
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
                   if k not in ('tile1_coord', 'tile2_coord', 'overlap_pixels',
                                'overlap_width', 'overlap_height')]
    for key in metric_keys:
        vals = [r[key] for r in results if r[key] is not None]
        if not vals:
            continue
        print(f"{key.upper():4s} -- mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")


# ============================================================
# STEP 6 : Feather Blending  (unchanged)
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

    Pass 1 -- warp each tile into its own bbox to build overlap_count, then
              discard the small warp immediately.
    Pass 2 -- re-warp each tile into its own bbox and blend that slice into
              the canvas immediately, then discard the warp.
    """
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

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

        # [FIX 1 -- accumulate-then-normalise, replacing sequential alpha
        # compositing] Floor the feather so an edge pixel that legitimately
        # has zero distance-transform weight (right at the tile boundary)
        # still contributes instead of being silently dropped; re-apply the
        # mask afterward so the floor never leaks outside the tile's real
        # footprint. Every tile then just adds its weighted contribution --
        # "first touch" and "Nth touch" are the same code path, and the
        # order tiles are processed in no longer matters.
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

    # [FIX 2 -- strip-wise in-place conversion, replacing the full-canvas
    # boolean-mask copy chain] canvas currently holds a WEIGHTED SUM (fix 1
    # accumulates rather than averages inline), so this pass also does the
    # single normalising divide -- one division per pixel instead of one
    # per tile-pair, and it folds into the same strip loop that used to be
    # the memory-spike site (canvas[valid] / np.clip / .astype each
    # materialised a full extra canvas-sized buffer here).
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
    im1 = axes[0, 0].imshow(debug_info['overlap_count'], cmap='jet')
    axes[0, 0].set_title('Overlap Count Map')
    plt.colorbar(im1, ax=axes[0, 0])
    im2 = axes[0, 1].imshow(debug_info['weight_map'], cmap='viridis')
    axes[0, 1].set_title('Weight Map')
    plt.colorbar(im2, ax=axes[0, 1])
    axes[1, 0].imshow(debug_info['valid_pixels'].astype(np.uint8) * 255, cmap='gray')
    axes[1, 0].set_title('Valid Pixels Mask')
    axes[1, 1].imshow(final_image)
    axes[1, 1].set_title('Blended Result')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
# STEP 6 (cont.) : ROI Cropping  (unchanged)
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
    _parser = argparse.ArgumentParser(description="Brute-force subtraction tile stitching")
    _parser.add_argument('--path', default="c:/path/to/your/tiles",
                         help="Folder of tile images to stitch")
    folder_path = _parser.parse_args().path

    # Initialize Performance & Memory Tracker
    tracker = None
    if CONFIG.get('enable_benchmark', True):
        tracker = PerformanceTracker("Brute-Force Subtraction Pipeline")

    # Step 1 -- Load images & build grid
    image_data, grid_info = load_image(folder_path, CONFIG['resize_factor'],
                                       CONFIG['registration_channel'])
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

    # Step 3-4 -- Brute-force subtraction registration (combined extract+match stage)
    match_result = register_overlap_pairs(image_data, overlap_pairs)

    # Step 4.5 -- fixed-stage grid-consistency snap (outlier + drift control)
    if CONFIG.get('enforce_grid_consistency', True):
        match_result = enforce_grid_consistency(match_result)
    if tracker:
        tracker.record_step("3-4. Brute-Force Subtraction Registration")

    # Step 5 -- Translation-only homography from measured shifts (no RANSAC)
    homography_results = calculate_homographies_batch(image_data, match_result)
    homography_results = add_nominal_fallback_edges(homography_results, overlap_pairs)
    if tracker:
        tracker.record_step("5. Homography from Brute-Force Shift")

    # Step 5.1 -- N/A for brute-force subtraction; see note above STEP 5 (cont.)

    # Step 5.5 -- Overlap quality metrics (PSNR, SSIM, RMSE, NCC)
    if CONFIG.get('use_global_adjustment', True):
        # grid consistency is already enforced upstream (Step 4.5), so no
        # extra outlier rejection needed here -> outlier_tol=None
        transforms, reference, reachable_images = solve_global_translations(
            homography_results, image_data, outlier_tol=None)
    else:
        transforms, reference, reachable_images = calculate_all_transforms(image_data, homography_results)
    n_total = len(image_data)
    if len(reachable_images) < 0.9 * n_total:
        print(f"[WARNING] Only {len(reachable_images)}/{n_total} tiles reachable "
              f"({len(reachable_images) / n_total * 100:.1f}%) -- the mosaic is "
              f"missing a significant fraction of tiles. Check registration quality "
              f"(response scores, TRANSLATION SPREAD) before trusting this result.")
    canvas_w, canvas_h, offset_x, offset_y  = calculate_optimal_canvas(image_data, transforms)
    if CONFIG.get('evaluate_metrics', False):
        evaluate_overlap_metrics(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            homography_results=homography_results,
            output_csv=f"overlap_metrics_brute_force_ovr{int(CONFIG['overlap_percentage'] * 100)}.csv",
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
    _tag = ""
    if CONFIG.get('flatfield_correction', False):
        _tag += "_ff"
    if CONFIG.get('gain_compensation', False):
        _tag += "_gain"
    if CONFIG.get('overlap_shading', False):
        _tag += "_shade"
    save_path = str(Path(folder_path) / f"result_brute_force.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")
    if CONFIG.get('save_preview', True):
        scale   = CONFIG.get('preview_scale', 0.125)
        preview = cv2.resize(final_result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        preview_path = str(Path(save_path).with_name(Path(save_path).stem + "_preview" + Path(save_path).suffix))
        cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print(f"Saved preview ({scale:.3f}x, {preview.shape[1]}x{preview.shape[0]}): {preview_path}")
    if tracker:
        tracker.record_step("6. Feather Blending, ROI Crop & Saving")

    # Output Benchmark Summary Report
    if tracker:
        tracker.print_summary()

    if CONFIG.get('debug', False):
        plt.figure(figsize=(20, 10))
        plt.imshow(final_result)
        plt.axis('off')
        plt.title('Brute-Force Subtraction Tile Stitching Result', fontsize=16)
        plt.show()