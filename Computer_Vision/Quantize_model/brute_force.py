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
    'resize_factor':        1.0,
    'overlap_percentage':   0.50,      # expected overlap from stage step size
    'canvas_padding':       50,
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
    'preprocess':           'highpass_zscore',
                                  # applied ONCE to each ROI before the slide-
                                  # and-subtract, so "aligned => difference ~ 0"
                                  # actually holds on real microscope tiles.
                                  # The value is a set of steps whose names
                                  # appear in the string; they run in this
                                  # fixed order: clahe -> highpass -> zscore.
                                  #   'none'    -- raw grayscale (only OK with
                                  #                flat illumination)
                                  #   'clahe'   -- contrast-limited adaptive
                                  #                histogram equalization; lifts
                                  #                local contrast in low-texture
                                  #                agar/colony regions so the
                                  #                cost minimum is sharper.
                                  #                NOTE: nonlinear & tile-local,
                                  #                so it can also inject a small
                                  #                systematic mismatch between
                                  #                the two crops -- measure,
                                  #                don't assume it helps.
                                  #   'zscore'  -- (x - mean) / std; kills
                                  #                brightness/contrast offset
                                  #   'highpass'-- x - GaussianBlur(x); kills
                                  #                vignetting / low-frequency shading
                                  # Combine with underscores, e.g.
                                  # 'highpass_zscore' (recommended default),
                                  # 'clahe_highpass_zscore', 'clahe_zscore'.
    'clahe_clip':           2.0,  # CLAHE clipLimit (higher = stronger contrast
                                  # boost, more noise amplification).
    'clahe_grid':           8,    # CLAHE tileGridSize (NxN). Smaller = more
                                  # local / more aggressive equalization.
    'highpass_sigma':       12.0, # Gaussian sigma (px) for the 'highpass' step.
                                  # ~ the scale of the shading to remove; larger
                                  # keeps more structure, smaller flattens harder.
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

    # --- Device ---
    'debug': True,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': False,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
    'enable_benchmark': True,       # Toggle Performance & Memory Tracker benchmarking
    'verbose_pair_metrics': False,  # Print PSNR/SSIM/NCC per pair (off = final summary only)
}


# ============================================================
# STEP 1 : Image Loading & Coordinate Extraction
# ============================================================

def load_image(folder_path, resize_factor=1.0):
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

    Returns
    -------
    image_data : dict  {(x, y): {'image', 'image_gray', 'filename'}}
    grid_info  : dict  {'dimensions', 'x_range', 'y_range', 'unique_x', 'unique_y'}
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' tidak ditemukan")

    focused_pattern  = re.compile(r'Focused_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.jpg')
    tile_pattern     = re.compile(r'tile_r(\d+)_c(\d+)\.jpg')
    embedded_pattern = re.compile(r'_r(\d+)_c(\d+)(?:_|\.)', re.IGNORECASE)

    image_data = {}
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
            image_data[coords] = {
                'image':      img_rgb,
                'image_gray': cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
                'filename':   name,
            }
        except Exception as e:
            print(f"[ERROR] Gagal memuat {name}: {e}")
            traceback.print_exc()

    if not image_data:
        raise ValueError("Tidak ada gambar yang ditemukan di folder.")

    x_coords = [c[0] for c in image_data]
    y_coords = [c[1] for c in image_data]
    unique_x = sorted(set(x_coords), reverse=True)
    unique_y = sorted(set(y_coords))

    grid_info = {
        'dimensions': (len(unique_x), len(unique_y)),
        'x_range':    (min(x_coords), max(x_coords)),
        'y_range':    (min(y_coords), max(y_coords)),
        'unique_x':   unique_x,
        'unique_y':   unique_y,
    }
    print(f"{len(image_data)} gambar dimuat dari {folder_path}")
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


def _preprocess_roi(img, method, sigma, clahe_clip=2.0, clahe_grid=8):
    """
    Condition an ROI crop ONCE so that the slide-and-score actually reaches
    its minimum where the tiles align. *method* is a set of step names found
    as substrings of the string; they are applied in this fixed order:

      'clahe'    -- contrast-limited adaptive histogram equalization. Boosts
                    local contrast so low-texture regions (agar, flat colony
                    interiors) still pin the cost minimum. It is nonlinear and
                    computed per CLAHE tile independently on each crop, so it
                    can also add a small systematic crop-vs-crop mismatch --
                    a net win only when the contrast gain outweighs that.
      'highpass' -- subtract a Gaussian-blurred copy; removes microscope
                    vignetting / low-frequency shading.
      'zscore'   -- (x - mean) / std; removes any remaining global
                    brightness/contrast offset between the two crops.

    Without at least one of these, two perfectly registered tiles can still
    differ by tens of grey levels across the overlap, leaving the cost
    surface flat and the search rudderless.
    """
    x = to_float_gray(img)
    if 'clahe' in method:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                                tileGridSize=(int(clahe_grid), int(clahe_grid)))
        x = clahe.apply(np.clip(x, 0, 255).astype(np.uint8)).astype(np.float32)
    if 'highpass' in method:
        x = x - cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    if 'zscore' in method:
        std = float(x.std())
        x = (x - float(x.mean())) / (std if std > 1e-6 else 1.0)
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
                       preprocess='highpass_zscore', highpass_sigma=12.0,
                       clahe_clip=2.0, clahe_grid=8):
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

    Returns (dx, dy, error, response)
      error    : the winning offset's match cost -- mean |a - b| ('sad'),
                 mean (a - b)^2 ('ssd'), or 1 - ZNCC ('ncc'). 0 == identical;
                 a small positive number for a good real-world match.
      response : (median_error - min_error) / median_error over the FINE
                 surface, clipped to [0, 1]. High => the winning offset stands
                 out sharply from neighbouring offsets (confident). Low =>
                 flat cost surface, the shift is unreliable.
    """
    a = _preprocess_roi(roi1, preprocess, highpass_sigma, clahe_clip, clahe_grid)
    b = _preprocess_roi(roi2, preprocess, highpass_sigma, clahe_clip, clahe_grid)
    step = max(1, int(search_step))

    # --- 1. coarse pass on downsampled ROIs -------------------------------
    ds = max(1, int(coarse_downsample))
    a_c = a[::ds, ::ds] if ds > 1 else a
    b_c = b[::ds, ::ds] if ds > 1 else b
    radius_c = max(1, int(round(search_radius / ds)))
    _, csx, csy, _ = _sweep(a_c, b_c, 0, 0, radius_c, step,
                            metric, min_overlap_fraction)
    coarse_sx, coarse_sy = csx * ds, csy * ds

    # --- 2. fine pass at full resolution --------------------------------
    fr = max(int(fine_radius), ds)
    err_surface, best_sx, best_sy, best_err = _sweep(
        a, b, coarse_sx, coarse_sy, fr, 1, metric, min_overlap_fraction)

    if not err_surface:
        return 0.0, 0.0, float('inf'), 0.0

    errs = np.fromiter(err_surface.values(), dtype=np.float64)
    med  = float(np.median(errs))
    response = float(np.clip((med - best_err) / (med + 1e-9), 0.0, 1.0))

    dx, dy = float(best_sx), float(best_sy)
    if subpixel:
        dx += _parabolic_offset(err_surface, best_sx, best_sy, 1, axis='x')
        dy += _parabolic_offset(err_surface, best_sx, best_sy, 1, axis='y')

    return dx, dy, best_err, response


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
            'dx', 'dy', 'error', 'response', 'direction', 'region1', 'region2'
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

            dx, dy, error, response = _brute_force_shift(
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
                clahe_clip=CONFIG['clahe_clip'],
                clahe_grid=CONFIG['clahe_grid'],
            )
            if response < threshold:
                low_confidence.append((coord1, coord2, response))

            match_result[(coord1, coord2)] = {
                'dx': dx, 'dy': dy, 'error': error, 'response': response,
                'direction': direction,
                'region1': r1, 'region2': r2,
            }
            print(f"Pair {i}/{len(overlap_pairs)}: {coord1} <-> {coord2} "
                  f"| shift=({dx:.2f},{dy:.2f}) error={error:.3f} response={response:.3f}")
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


# ============================================================
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
    homography_results = {}
    success = failed = 0

    for (coord1, coord2), info in match_result.items():
        try:
            if info['response'] < response_threshold:
                print(f"[SKIP] {coord1}<->{coord2}: response {info['response']:.3f} < threshold")
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


def calculate_optimal_canvas(image_data, transforms):
    all_corners = []
    for coord, T in transforms.items():
        h, w = image_data[coord]['image'].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        all_corners.extend(cv2.perspectiveTransform(corners, T).reshape(-1, 2))

    arr    = np.array(all_corners)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    pad    = CONFIG['canvas_padding']

    canvas_w = int(max_xy[0] - min_xy[0]) + 2 * pad
    canvas_h = int(max_xy[1] - min_xy[1]) + 2 * pad
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


def detect_pairwise_overlap(image_data, transforms, coord1, coord2,
                             canvas_w, canvas_h, offset_x, offset_y):
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    img1 = image_data[coord1]['image'].astype(np.float32)
    img2 = image_data[coord2]['image'].astype(np.float32)
    T1   = offset_matrix @ transforms[coord1]
    T2   = offset_matrix @ transforms[coord2]

    w1 = cv2.warpPerspective(img1, T1, (canvas_w, canvas_h))
    w2 = cv2.warpPerspective(img2, T2, (canvas_w, canvas_h))

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

    Memory-efficient two-pass strategy
    -----------------------------------
    Pass 1 -- warp each tile once to build overlap_count, then discard the
              warp immediately.
    Pass 2 -- re-warp each tile and blend it into the canvas immediately,
              then discard the warp.
    Peak RAM is O(canvas_size), regardless of tile count N.
    """
    offset_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    canvas        = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map    = np.zeros((canvas_h, canvas_w),    dtype=np.float32)
    overlap_count = np.zeros((canvas_h, canvas_w),    dtype=np.int32)

    print(f"Pass 1 -- computing overlap map ({len(reachable_images)} tiles) ...")
    for coord in reachable_images:
        img   = image_data[coord]['image'].astype(np.float32)
        T_adj = offset_matrix @ transforms[coord]
        w     = cv2.warpPerspective(img, T_adj, (canvas_w, canvas_h))
        overlap_count += (w.sum(axis=2) > 0).astype(np.int32)
        del w

    kernel = np.ones((5, 5), np.uint8)
    print(f"Pass 2 -- blending ({len(reachable_images)} tiles) ...")
    for coord in reachable_images:
        img    = image_data[coord]['image'].astype(np.float32)
        T_adj  = offset_matrix @ transforms[coord]
        warped = cv2.warpPerspective(img, T_adj, (canvas_w, canvas_h))
        mask   = (warped.sum(axis=2) > 0).astype(np.uint8)

        inner        = cv2.erode(mask, kernel, iterations=2)
        dist         = cv2.distanceTransform(inner, cv2.DIST_L2, 5)
        max_dist     = dist.max()
        feather_zone = min(CONFIG['feather_distance'], max_dist * 0.3) if max_dist > 0 else 1
        feather      = np.minimum(dist / feather_zone, 1.0) * mask.astype(np.float32)

        overlap_here = (overlap_count * mask) > 1
        new_here     = (weight_map == 0) & (feather > 0)

        for c in range(3):
            canvas[:, :, c][new_here] = warped[:, :, c][new_here]
        weight_map[new_here] = feather[new_here]

        if overlap_here.any():
            cur_w  = feather[overlap_here]
            ext_w  = weight_map[overlap_here]
            total  = cur_w + ext_w
            alpha  = np.divide(cur_w, total, out=np.zeros_like(cur_w), where=total != 0)
            for c in range(3):
                canvas[:, :, c][overlap_here] = (
                    alpha * warped[:, :, c][overlap_here] +
                    (1 - alpha) * canvas[:, :, c][overlap_here]
                )
            weight_map[overlap_here] = np.maximum(ext_w, cur_w)

        del warped

    valid = weight_map > 0
    final = np.zeros_like(canvas, dtype=np.uint8)
    final[valid] = np.clip(canvas[valid], 0, 255).astype(np.uint8)

    unique_ov, counts_ov = np.unique(overlap_count, return_counts=True)
    debug_info = {
        'overlap_count': overlap_count,
        'weight_map':    weight_map,
        'valid_pixels':  valid,
        'overlap_stats': dict(zip(unique_ov.tolist(), counts_ov.tolist())),
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

    # Step 3-4 -- Brute-force subtraction registration (combined extract+match stage)
    match_result = register_overlap_pairs(image_data, overlap_pairs)
    if tracker:
        tracker.record_step("3-4. Brute-Force Subtraction Registration")

    # Step 5 -- Translation-only homography from measured shifts (no RANSAC)
    homography_results = calculate_homographies_batch(image_data, match_result)
    if tracker:
        tracker.record_step("5. Homography from Brute-Force Shift")

    # Step 5.1 -- N/A for brute-force subtraction; see note above STEP 5 (cont.)

    # Step 5.5 -- Overlap quality metrics (PSNR, SSIM, RMSE, NCC)
    transforms, reference, reachable_images = calculate_all_transforms(image_data, homography_results)
    canvas_w, canvas_h, offset_x, offset_y  = calculate_optimal_canvas(image_data, transforms)
    if CONFIG.get('evaluate_metrics', False):
        evaluate_overlap_metrics(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            homography_results=homography_results,
            output_csv="overlap_metrics_brute_force.csv",
            visualize_sample=False,
        )
    if tracker and CONFIG.get('evaluate_metrics', False):
        tracker.record_step("5.5 Overlap Quality Metrics Evaluation")

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
    save_path = str(Path(folder_path) / "result_brute_force.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")
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
