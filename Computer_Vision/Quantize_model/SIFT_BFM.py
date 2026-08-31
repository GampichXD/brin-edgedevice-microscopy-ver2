"""
Tile Stitching Pipeline -- SIFT + BFMatcher Variant
=====================================================
Feature extraction             : SIFT  (OpenCV)
Feature matching               : BFMatcher  (brute-force, Lowe's ratio test)
Homography + RANSAC            : cv2.estimateAffinePartial2D (unchanged)
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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
    'resize_factor':        1.0,
    'overlap_percentage':   0.3,
    'canvas_padding':       50,
    'display_feather_distance':  30,  # soft feather for human viewing
    'analysis_feather_distance': 0,   # hard seam for analysis/training copy
    'feather_distance':     30,       # legacy alias -- used by blend_panorama
    'save_analysis_copy':   False,     # also save a hard-seam copy alongside the soft display one
    # 'cuda' moves blend_panorama's per-tile blend arithmetic (masking, feather-weight
    # computation, canvas accumulation) onto the GPU via PyTorch tensors -- this is the
    # O(canvas_size x 3 channels x num_tiles) part that gets slow on grids beyond ~5x5.
    # The geometric warp + mask erode/distanceTransform stay on CPU via OpenCV either way
    # (this build's OpenCV has no CUDA support to swap those to). Set to 'cpu' to disable.
    'feather_blend_device': 'cpu' if HAS_TORCH and torch.cuda.is_available() else 'cpu',
    'normalization_method': 'skimage_histogram_match',
    'min_overlap_area':     500,

    # --- SIFT (feature detector) ---
    'feature_method':      'sift',    # Focused only on SIFT
    'sift_max_keypoints':  1024,      # Maximum keypoints per ROI crop (reduced for edge devices)
    'sift_peak_threshold': 0.01,      # DoG peak threshold; lower = more keypoints
    'sift_edge_threshold': 10,        # Harris edge threshold

    # --- BFMatcher ---
    'lowe_ratio': 0.75,

    # --- RANSAC / homography ---
    'reproj_thresh':    4.0,   # px -- unified to 4.0 across all pipeline variants
    'min_inlier_ratio': 0.3,   # informational -- not used to reject pairs
    'affine_scale_bounds': (0.5, 2.0),  # reject a pair's affine transform if its
        # uniform scale falls outside this. Adjacent tiles come from the same
        # fixed lens, so scale should sit near 1.0 -- RANSAC can still return a
        # locally-consistent-looking but globally wrong transform on repetitive
        # colony textures, and an unchecked scale gets multiplied across the BFS
        # transform chain in calculate_all_transforms(), which is what blew
        # calculate_optimal_canvas() up to a multi-terabyte allocation.

    # --- Device ---
    'use_gpu': True,                # Enable CUDA BFMatcher acceleration if available
    'debug': False,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': False,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
    'enable_benchmark': True,       # Toggle Performance & Memory Tracker benchmarking
    'verbose_pair_metrics': False,  # Print PSNR/SSIM/NCC/RepError per pair (off = final summary only)
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
      longer filename carrying extra metadata (e.g. capture timestamp, a
      session hash, a save timestamp -- as in
      ``IMG_20260831030147_6d37fc_r0_c0_20260831_030313.jpg`` from
      Euglena_tiles/E_Coli_5x5). Matched with `.search()` rather than
      `.match()` so surrounding text is ignored; only the r_/c_ indices
      matter (col->x, row->y, same as the plain tile_r_c pattern).

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
    embedded_pattern = re.compile(r'_r(\d+)_c(\d+)_', re.IGNORECASE)

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
            # Apply CLAHE to the ROI crop before detection to improve keypoint
            # density on low-contrast agar images.  Applied only to the crop,
            # NOT to the stored image_gray, so photometric metrics are unaffected.
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

def calculate_homography(kp1, kp2, matches, reproj_thresh, scale_bounds=None):
    """Estimate transformation from matched keypoints.

    Uses estimateAffinePartial2D (4 DOF: translation + rotation + uniform
    scale) instead of findHomography (8 DOF) because the CNC stage only
    translates in X and Y.  A full perspective model has 6 free parameters
    to absorb noise from repeated colony patterns, making RANSAC susceptible
    to confident-looking but geometrically wrong solutions.  The affine
    result is promoted to a 3×3 matrix for compatibility with warpPerspective.
    """
    if len(matches) <= 4:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    # estimateAffinePartial2D: 4 DOF (tx, ty, rotation, uniform scale)
    # Much more robust on repetitive colony domains than full 8DOF homography.
    H_affine, status = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh
    )
    if H_affine is None:
        return None
    det = np.linalg.det(H_affine[:, :2])
    # Reject a degenerate similarity transform (near-zero rotation/scale
    # determinant) -- RANSAC can return one when inliers are few or
    # near-collinear. A singular H is unusable for warpPerspective and
    # would blow up np.linalg.inv() later in build_transformation_graph.
    if abs(det) < 1e-6:
        return None
    # Reject an implausible uniform scale too (see CONFIG['affine_scale_bounds']):
    # a stray scale of e.g. 50x compounds multiplicatively across the BFS
    # transform chain and produces a canvas many orders of magnitude too big.
    scale_lo, scale_hi = scale_bounds or (0.5, 2.0)
    scale = np.sqrt(abs(det))
    if not (scale_lo <= scale <= scale_hi):
        return None
    # Promote 2×3 affine matrix to 3×3 for warpPerspective compatibility
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = H_affine
    return (matches, H, status)


def calculate_homographies_batch(image_data, match_result, reproj_thresh=4):
    scale_bounds = CONFIG.get('affine_scale_bounds', (0.5, 2.0))
    homography_results = {}
    success = failed = 0

    for (coord1, coord2), info in match_result.items():
        try:
            result = calculate_homography(
                info['keypoints1'], info['keypoints2'],
                info['matches'], reproj_thresh, scale_bounds
            )
            if result is None:
                print(f"[SKIP] {coord1}↔{coord2}: <4 matches or degenerate/implausible transform")
                failed += 1
                continue

            matches, H, status = result
            if H is None:
                print(f"[SKIP] {coord1}↔{coord2}: findHomography returned None")
                failed += 1
                continue

            inliers = int(np.sum(status)) if status is not None else 0
            homography_results[(coord1, coord2)] = {
                'homography_matrix': H,
                'inliers':           inliers,
                'total_matches':     len(matches),
                'inlier_ratio':      inliers / len(matches) if matches else 0,
                'direction':         info['direction'],
                'status':            status,
            }
            print(f"   OK {coord1}↔{coord2}: {inliers}/{len(matches)} inliers")
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
    return homography_results


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

def build_transformation_graph(homography_results):
    graph = defaultdict(dict)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        # calculate_homography() already rejects near-singular H via a
        # determinant check, but guard the inverse anyway -- belt-and-braces
        # against any degenerate matrix reaching here, since an uninvertible
        # edge is unusable for the graph in either direction.
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print(f"[SKIP] {c1}<->{c2}: singular homography matrix, dropping edge")
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


def calculate_optimal_canvas(image_data, transforms):
    all_corners = []
    for coord, T in transforms.items():
        h, w = image_data[coord]['image'].shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_corners.extend(cv2.perspectiveTransform(corners, T).reshape(-1, 2))

    arr    = np.array(all_corners)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    pad    = CONFIG['canvas_padding']

    canvas_w = int(max_xy[0] - min_xy[0]) + 2 * pad
    canvas_h = int(max_xy[1] - min_xy[1]) + 2 * pad
    offset_x = int(-min_xy[0]) + pad
    offset_y = int(-min_xy[1]) + pad

    # Last-resort guard: a physically plausible canvas is bounded by grid
    # size * tile size (every tile placed end-to-end, no overlap). Blowing
    # past this means a bad homography (implausible scale/translation) got
    # composed across the BFS transform chain -- fail loudly here with a
    # diagnosable message instead of trying (and failing) to allocate a
    # multi-terabyte canvas array.
    max_tile_dim = max(max(d['image'].shape[:2]) for d in image_data.values())
    sane_limit   = max_tile_dim * (len(image_data) + 2)
    if canvas_w > sane_limit or canvas_h > sane_limit:
        raise RuntimeError(
            f"Canvas size {canvas_w}x{canvas_h} is implausible for {len(image_data)} "
            f"tile(s) of max dimension {max_tile_dim}px (sane limit ~{sane_limit}px/side). "
            f"This means at least one accepted homography is geometrically wrong -- check "
            f"the per-pair inlier ratios/scale printed above, or tighten "
            f"CONFIG['affine_scale_bounds']."
        )

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


def detect_pairwise_overlap(image_data, transforms, coord1, coord2,
                             canvas_w, canvas_h, offset_x, offset_y):
    offset_matrix = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float32)

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
                   canvas_w, canvas_h, offset_x, offset_y, feather_distance=None):
    """
    Warp and feather-blend all reachable tiles onto a single canvas.

    Memory-efficient two-pass strategy
    -----------------------------------
    Pass 1 -- warp each tile once to build overlap_count, then discard the
              warp immediately.  Only one full-canvas float32 image lives in
              RAM at a time.
    Pass 2 -- re-warp each tile and blend it into the canvas immediately,
              then discard the warp.

    For an N-tile mosaic the old approach kept N warped canvases in RAM
    simultaneously (e.g. ~7.5 GB for 25 × 300 MB tiles).  The new approach
    peak RAM is O(canvas_size), regardless of N.

    feather_distance : width in px of the soft blend transition at tile
        seams. Defaults to CONFIG['feather_distance']. <= 0 switches to a
        hard, winner-take-all seam instead: each overlap pixel goes
        entirely to whichever tile is more "interior" there (larger
        distance-to-its-own-edge), no color mixing -- for an
        analysis/training copy where blended/ghosted seam pixels are
        undesirable (matches CONFIG['analysis_feather_distance']). Naively
        passing feather_distance=0 into the old soft-blend formula divided
        by zero (silently producing inf/nan at seams) -- this is the fix
        for that, not just a threshold tweak.
    """
    feather_distance = CONFIG['feather_distance'] if feather_distance is None else feather_distance
    hard_seam = feather_distance <= 0

    offset_matrix = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float32)

    canvas        = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_map    = np.zeros((canvas_h, canvas_w),    dtype=np.float32)
    overlap_count = np.zeros((canvas_h, canvas_w),    dtype=np.int32)

    # ------------------------------------------------------------------
    # Pass 1 : accumulate overlap_count -- one warp in RAM at a time
    # ------------------------------------------------------------------
    print(f"Pass 1 -- computing overlap map ({len(reachable_images)} tiles) ...")
    for coord in reachable_images:
        img   = image_data[coord]['image'].astype(np.float32)
        T_adj = offset_matrix @ transforms[coord]
        w     = cv2.warpPerspective(img, T_adj, (canvas_w, canvas_h))
        overlap_count += (w.sum(axis=2) > 0).astype(np.int32)
        del w   # free immediately -- only one warped image in RAM at a time

    # ------------------------------------------------------------------
    # Pass 2 : blend -- one warp in RAM at a time
    # ------------------------------------------------------------------
    # The geometric warp + mask erode/distanceTransform stay on CPU via
    # OpenCV regardless of feather_blend_device -- this build's OpenCV has
    # no CUDA support (cv2.cuda.getCudaEnabledDeviceCount() == 0) and
    # reimplementing warpPerspective's exact interpolation in PyTorch risks
    # subtly changing stitching quality for a part that isn't the actual
    # bottleneck. What moves to GPU when requested is the per-tile blend
    # arithmetic below -- full 3-channel canvas-sized elementwise ops,
    # repeated once per tile, which is what actually scales up on grids
    # beyond ~5x5.
    blend_device  = CONFIG.get('feather_blend_device', 'cpu')
    requested_gpu = blend_device == 'cuda'
    use_gpu       = requested_gpu and HAS_TORCH and torch.cuda.is_available()
    if requested_gpu and not use_gpu:
        print("[WARN] feather_blend_device='cuda' requested but CUDA is not available -- using CPU.")

    kernel = np.ones((5, 5), np.uint8)
    print(f"Pass 2 -- blending ({len(reachable_images)} tiles) ... [{'GPU' if use_gpu else 'CPU'}]")

    if use_gpu:
        canvas_t        = torch.zeros((canvas_h, canvas_w, 3), dtype=torch.float32, device='cuda')
        weight_map_t    = torch.zeros((canvas_h, canvas_w),    dtype=torch.float32, device='cuda')
        overlap_count_t = torch.from_numpy(overlap_count).to('cuda')

    for coord in reachable_images:
        img    = image_data[coord]['image'].astype(np.float32)
        T_adj  = offset_matrix @ transforms[coord]
        warped = cv2.warpPerspective(img, T_adj, (canvas_w, canvas_h))
        mask   = (warped.sum(axis=2) > 0).astype(np.uint8)

        inner = cv2.erode(mask, kernel, iterations=2)
        dist  = cv2.distanceTransform(inner, cv2.DIST_L2, 5)

        if use_gpu:
            warped_t = torch.from_numpy(warped).to('cuda')
            mask_t   = torch.from_numpy(mask).to('cuda').bool()
            dist_t   = torch.from_numpy(dist).to('cuda')
            mask_f_t = mask_t.to(torch.float32)

            if hard_seam:
                feather_t = dist_t * mask_f_t
            else:
                max_dist     = dist_t.max().item()
                feather_zone = min(feather_distance, max_dist * 0.3) if max_dist > 0 else 1
                feather_t    = torch.clamp(dist_t / feather_zone, max=1.0) * mask_f_t

            overlap_here_t = (overlap_count_t * mask_t) > 1
            new_here_t     = (weight_map_t == 0) & (feather_t > 0)

            canvas_t[new_here_t]     = warped_t[new_here_t]
            weight_map_t[new_here_t] = feather_t[new_here_t]

            if overlap_here_t.any():
                cur_w = feather_t[overlap_here_t]
                ext_w = weight_map_t[overlap_here_t]
                if hard_seam:
                    win_mask_t = torch.zeros_like(mask_t)
                    win_mask_t[overlap_here_t] = cur_w > ext_w
                    canvas_t[win_mask_t] = warped_t[win_mask_t]
                else:
                    total = cur_w + ext_w
                    alpha = torch.where(total != 0, cur_w / total, torch.zeros_like(cur_w))
                    canvas_t[overlap_here_t] = (
                        alpha.unsqueeze(-1) * warped_t[overlap_here_t] +
                        (1 - alpha).unsqueeze(-1) * canvas_t[overlap_here_t]
                    )
                weight_map_t[overlap_here_t] = torch.maximum(ext_w, cur_w)

            del warped, warped_t, mask_t, dist_t, feather_t  # free immediately

        else:
            if hard_seam:
                # Priority score for winner-take-all ownership -- deliberately
                # NOT capped to [0,1] the way the soft-blend alpha below is,
                # since it's never used as a blend weight in this mode, only
                # compared against other tiles' scores at the same pixel.
                feather = dist * mask.astype(np.float32)
            else:
                max_dist     = dist.max()
                feather_zone = min(feather_distance, max_dist * 0.3) if max_dist > 0 else 1
                feather      = np.minimum(dist / feather_zone, 1.0) * mask.astype(np.float32)

            overlap_here = (overlap_count * mask) > 1
            new_here     = (weight_map == 0) & (feather > 0)

            for c in range(3):
                canvas[:,:,c][new_here] = warped[:,:,c][new_here]
            weight_map[new_here] = feather[new_here]

            if overlap_here.any():
                cur_w  = feather[overlap_here]
                ext_w  = weight_map[overlap_here]
                if hard_seam:
                    # No color mixing: each overlap pixel goes entirely to
                    # whichever tile currently has the higher priority score.
                    win_mask = np.zeros_like(mask, dtype=bool)
                    win_mask[overlap_here] = cur_w > ext_w
                    for c in range(3):
                        canvas[:,:,c][win_mask] = warped[:,:,c][win_mask]
                else:
                    total  = cur_w + ext_w
                    alpha  = np.divide(cur_w, total, out=np.zeros_like(cur_w), where=total != 0)
                    for c in range(3):
                        canvas[:,:,c][overlap_here] = (
                            alpha * warped[:,:,c][overlap_here] +
                            (1 - alpha) * canvas[:,:,c][overlap_here]
                        )
                weight_map[overlap_here] = np.maximum(ext_w, cur_w)

            del warped  # free immediately

    if use_gpu:
        canvas     = canvas_t.cpu().numpy()
        weight_map = weight_map_t.cpu().numpy()
        del canvas_t, weight_map_t, overlap_count_t
        torch.cuda.empty_cache()

    valid     = weight_map > 0
    final     = np.zeros_like(canvas, dtype=np.uint8)
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
    _parser.add_argument('--path', default="/home/brin-microscope/Documents/Tugas-Akhir/Hardware/local_datasets/bfe9a132",
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

    # Step 5 -- Homography via RANSAC
    homography_results = calculate_homographies_batch(image_data, match_result, CONFIG['reproj_thresh'])
    if tracker:
        tracker.record_step("5. Homography RANSAC")

    # Step 5.1 -- Reprojection error report
    if CONFIG.get('evaluate_metrics', False):
        evaluate_homography_reprojection(match_result, homography_results,
                                         output_csv="homography_reprojection_sift_bfm.csv")

    # Step 5.5 -- Overlap quality metrics (PSNR, SSIM, RMSE, NCC)
    transforms, reference, reachable_images = calculate_all_transforms(image_data, homography_results)
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

    # Step 6 -- Feather blending
    # Hard-seam analysis copy first (if enabled), display copy second -- in
    # that order so this copy's own save message (deliberately NOT phrased
    # "Saved: ...", see below) never becomes the last "Saved: <path>.jpg"
    # match in the process output, which is what automate.py's tile-runner
    # parses to find the stitched panorama. Keeps automate.py picking up
    # the same (display) file it always has, unaffected by this addition.
    if CONFIG.get('save_analysis_copy', False):
        analysis_blended, _ = blend_panorama(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            feather_distance=CONFIG.get('analysis_feather_distance', 0),
        )
        analysis_roi_info = find_roi(analysis_blended, min_threshold=1, debug=False)
        analysis_result, _ = extract_roi(analysis_blended, analysis_roi_info, padding=10)
        analysis_save_path = str(Path(folder_path) / "result_sift_bfm_analysis.jpg")
        cv2.imwrite(analysis_save_path, cv2.cvtColor(analysis_result, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Hard-seam analysis copy written: {analysis_save_path}")

    blended, debug_info = blend_panorama(
        image_data, transforms, reachable_images,
        canvas_w, canvas_h, offset_x, offset_y,
        feather_distance=CONFIG.get('display_feather_distance', CONFIG['feather_distance']),
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
