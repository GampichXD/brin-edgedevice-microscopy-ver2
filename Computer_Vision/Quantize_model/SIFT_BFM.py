import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv
from pathlib import Path
import warnings
import traceback
from collections import defaultdict, deque
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

# Installation:  pip install -r requirements_BFMatcher.txt
# Full guide:    see INSTALL.md in this directory

warnings.filterwarnings('ignore')
cv2.ocl.setUseOpenCL(False)

import time
import resource

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
        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
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

# ============================================================
# CONFIGURATION — stop scattering magic numbers everywhere
# ============================================================
CONFIG = {
    'resize_factor': 1.0,
    'overlap_percentage': 0.4,
    'feature_method': 'sift',       # Focused only on SIFT
    'lowe_ratio': 0.75,
    'reproj_thresh': 4.0,           # px -- unified to 4.0 across all pipeline variants
    'min_inlier_ratio': 0.3,        # informational -- not used to reject pairs
    'min_overlap_area': 500,
    'canvas_padding': 50,
    'normalization_method': 'skimage_histogram_match',
    'display_feather_distance':  30,  # soft feather for human viewing
    'analysis_feather_distance': 0,   # hard seam for analysis/training copy
    'feather_distance': 30,           # legacy alias -- used by blend_panorama
    'peak_threshold': 0.01,
    'edge_threshold': 10,
    'use_gpu': True,                # Enable GPU acceleration (CUDA / OpenCL)
    'sift_max_keypoints': 2048,      # Maximum keypoints per ROI crop (reduced for edge devices)
    'debug': True,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': True,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
}
# ============================================================
# STEP 1: Image Loading & Coordinate Extraction
# ============================================================

def load_image(folder_path, resize_factor=1.0):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' tidak ditemukan")

    focused_pattern = re.compile(r'Focused_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.jpg')
    tile_pattern    = re.compile(r'tile_r(\d+)_c(\d+)\.jpg')

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
                coords = (float(m.group(2)), float(m.group(1)))   # col → x, row → y

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
    print(f"Dimensi Grid  : {grid_info['dimensions'][0]} × {grid_info['dimensions'][1]}")
    print(f"Rentang X     : {grid_info['x_range']}")
    print(f"Rentang Y     : {grid_info['y_range']}")
    return image_data, grid_info


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

    plt.suptitle(f'Image Grid: {grid_w}×{grid_h} | {len(image_data)} images loaded', fontsize=20)
    plt.tight_layout()
    return fig


def _normalise_axes(axes, rows, cols):
    """Guarantee axes is always a list-of-lists, regardless of grid shape."""
    if rows == 1 and cols == 1:
        return [[axes]]
    if rows == 1:
        return [list(axes)]
    if cols == 1:
        return [[ax] for ax in axes]
    return [list(row) for row in axes]
# ============================================================
# STEP 2: Feature Extraction
# ============================================================

# Track acceleration methods used at runtime
DETECTOR_ACCEL = "CPU"
MATCHER_ACCEL = "CPU"


def select_descriptor(image, method='sift'):
    """
    Extract SIFT features. Always returns (keypoints, descriptors) as CPU types
    for keypoints and numpy arrays for descriptors.
    Uses GPU/OpenCL acceleration if CONFIG['use_gpu'] is enabled and supported.
    """
    global DETECTOR_ACCEL
    sift_max_kps = CONFIG.get('sift_max_keypoints', 2048)
    nfeatures = sift_max_kps if sift_max_kps != -1 else 0
    sift_peak_threshold = CONFIG['peak_threshold']
    sift_edge_threshold = CONFIG['edge_threshold']
    detector = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=sift_peak_threshold,
                               edgeThreshold=sift_edge_threshold, sigma=1.6)

    use_gpu = CONFIG.get('use_gpu', True)
    if use_gpu and cv2.ocl.haveOpenCL():
        try:
            cv2.ocl.setUseOpenCL(True)
            umat_image = cv2.UMat(image)
            kps, descs = detector.detectAndCompute(umat_image, None)
            if descs is not None:
                descs = descs.get()  # Convert cv2.UMat back to numpy array
            DETECTOR_ACCEL = "GPU (OpenCL SIFT)"
            return kps, descs
        except Exception as e:
            print(f"[WARN] OpenCL SIFT extraction failed, falling back to CPU: {e}")

    DETECTOR_ACCEL = "CPU"
    return detector.detectAndCompute(image, None)


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
# STEP 3: Feature Matching
# ============================================================

def keypoints_matching(feat1, feat2, method='sift', ratio=0.75):
    """
    Perform keypoint matching using SIFT descriptors.
    Uses CUDA-accelerated DescriptorMatcher if available,
    otherwise falls back to OpenCL-accelerated or standard CPU BFMatcher.
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

    # 2. Try OpenCL BFMatcher (Transparent API)
    if use_gpu and cv2.ocl.haveOpenCL():
        try:
            cv2.ocl.setUseOpenCL(True)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            umat_feat1 = cv2.UMat(feat1)
            umat_feat2 = cv2.UMat(feat2)
            raw = bf.knnMatch(umat_feat1, umat_feat2, k=2)
            MATCHER_ACCEL = "GPU (OpenCL BFMatcher)"
            return [m for m, n in raw if m.distance < n.distance * ratio]
        except Exception as e:
            print(f"[WARN] OpenCL BFMatcher failed, falling back: {e}")

    # 3. CPU Fallback
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
        print(f"   Avg quality score      : {np.mean(quality):.2f}")# ============================================================
# STEP 4: Homography Calculation
# ============================================================

def calculate_homography(kp1, kp2, matches, reproj_thresh):
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
    # Promote 2×3 affine matrix to 3×3 for warpPerspective compatibility
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = H_affine
    return (matches, H, status)


def calculate_homographies_batch(image_data, match_result, reproj_thresh=4):
    homography_results = {}
    success = failed = 0

    for (coord1, coord2), info in match_result.items():
        try:
            result = calculate_homography(
                info['keypoints1'], info['keypoints2'],
                info['matches'], reproj_thresh
            )
            if result is None:
                print(f"[SKIP] {coord1}↔{coord2}: <4 matches")
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


def evaluate_homography_reprojection(match_result, homography_results,
                                     output_csv="homography_reprojection.csv"):
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
        print(f"✅ {coord1}↔{coord2}: reprojection mean={reproj['mean']:.3f} "
              f"median={reproj['median']:.3f} min={reproj['min']:.3f} max={reproj['max']:.3f}")

    if results:
        _save_csv(results, output_csv,
                  ['tile1_coord','tile2_coord','match_count','inliers','mean_error','median_error','min_error','max_error'])
    return results


def build_transformation_graph(homography_results):
    graph = defaultdict(dict)
    for (c1, c2), info in homography_results.items():
        H = info['homography_matrix']
        graph[c1][c2] = H
        graph[c2][c1] = np.linalg.inv(H)
    return dict(graph)


def find_path_bfs(source, target, graph):
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


def calculate_transform_to_reference(source, reference, graph):
    path = find_path_bfs(source, reference, graph)
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
            T = calculate_transform_to_reference(coord, reference, graph)
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

    print(f"   Canvas  : {canvas_w} × {canvas_h}")
    print(f"   Offset  : ({offset_x}, {offset_y})")
    return canvas_w, canvas_h, offset_x, offset_y
# ============================================================
# STEP 5: Blending
# ============================================================

def blend_panorama(image_data, transforms, reachable_images,
                   canvas_w, canvas_h, offset_x, offset_y):
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
    """
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
            canvas[:,:,c][new_here] = warped[:,:,c][new_here]
        weight_map[new_here] = feather[new_here]

        if overlap_here.any():
            cur_w  = feather[overlap_here]
            ext_w  = weight_map[overlap_here]
            total  = cur_w + ext_w
            alpha  = np.divide(cur_w, total, out=np.zeros_like(cur_w), where=total != 0)
            for c in range(3):
                canvas[:,:,c][overlap_here] = (
                    alpha * warped[:,:,c][overlap_here] +
                    (1 - alpha) * canvas[:,:,c][overlap_here]
                )
            weight_map[overlap_here] = np.maximum(ext_w, cur_w)

        del warped  # free immediately

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
# ============================================================
# STEP 5 (cont.): ROI Cropping
# ============================================================

def _build_height_matrix_vectorized(valid_mask):
    """
    Build height matrix using vectorized row operations -- O(rows*cols) in NumPy.

    For each row, the height at each column is the running count of consecutive
    valid cells ending at that row (reset to 0 on invalid cells).  This uses
    a full row-wise numpy operation instead of a nested Python loop, matching
    the LG variants for consistency and speed on large mosaics.
    """
    h_matrix = np.zeros_like(valid_mask, dtype=np.int32)
    h_matrix[0] = valid_mask[0].astype(np.int32)
    col_data = valid_mask.astype(np.int32)
    for row in range(1, valid_mask.shape[0]):
        h_matrix[row] = (h_matrix[row - 1] + 1) * col_data[row]  # full row at once
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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    valid_mask = (gray >= min_threshold).astype(np.int32)
    rows, cols = valid_mask.shape

    if debug:
        pct = valid_mask.sum() / (rows * cols) * 100
        print(f"🔍 Image: {cols}×{rows} | Valid pixels: {valid_mask.sum()} ({pct:.1f}%)")

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
        print(f"✅ ROI: pos=({best_roi['x']},{best_roi['y']}) "
              f"size={best_roi['width']}×{best_roi['height']} "
              f"eff={eff:.1f}% ar={ar:.2f}")
    return best_roi


def extract_roi(image, roi_info, padding=5):
    if roi_info is None:
        return image, {}
    pad = padding
    x  = max(0, roi_info['x'] - pad)
    y  = max(0, roi_info['y'] - pad)
    x2 = min(image.shape[1], roi_info['x'] + roi_info['width']  + pad)
    y2 = min(image.shape[0], roi_info['y'] + roi_info['height'] + pad)
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
# Metrics  (defined ONCE — not three times like before)
# ============================================================

def normalize_intensity(img1, img2, method='clahe'):
    """Normalize intensity of two images. Returns uint8 pair.

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
    elif method == 'histogram_matching':
        if img1.ndim == 3:
            for c in range(img1.shape[2]):
                i1[:,:,c] = cv2.equalizeHist(i1[:,:,c].astype(np.uint8))
                i2[:,:,c] = cv2.equalizeHist(i2[:,:,c].astype(np.uint8))
        else:
            i1 = cv2.equalizeHist(i1.astype(np.uint8)).astype(np.float32)
            i2 = cv2.equalizeHist(i2.astype(np.uint8)).astype(np.float32)
    elif method == 'mean_std':
        m1, s1 = np.mean(i1), np.std(i1)
        m2, s2 = np.mean(i2), np.std(i2)
        if s2 > 0:
            i2 = (i2 - m2) * (s1 / s2) + m1
        i2 = np.clip(i2, 0, 255)
    elif method == 'minmax':
        for img in (i1, i2):
            lo, hi = img.min(), img.max()
            if hi - lo > 0:
                img[:] = (img - lo) * 255.0 / (hi - lo)
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
    elif method == 'binary_threshold':
        thresh = 50
        i1 = (i1 > thresh).astype(np.float32) * 255
        i2 = (i2 > thresh).astype(np.float32) * 255
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return i1.astype(np.uint8), i2.astype(np.uint8)


def _to_gray(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img


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
# ============================================================
# STEP 4.1: Reprojection Error Calculation
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
# ============================================================
# STEP 4.2: Overlap NCC Calculation
# ============================================================

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

        gray1 = cv2.cvtColor(w1.astype(np.uint8), cv2.COLOR_RGB2GRAY) if w1.ndim == 3 else w1
        gray2 = cv2.cvtColor(w2.astype(np.uint8), cv2.COLOR_RGB2GRAY) if w2.ndim == 3 else w2
        ov1 = gray1[overlap]
        ov2 = gray2[overlap]

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
# STEP 4.5: Overlap Evaluation
# ============================================================

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
                              output_csv="overlap_evaluation.csv",
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
                g1 = cv2.cvtColor(ov1_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) \
                     if ov1_u8.ndim == 3 else ov1_u8.astype(np.float32)
                g2 = cv2.cvtColor(ov2_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) \
                     if ov2_u8.ndim == 3 else ov2_u8.astype(np.float32)
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
            print(f"✅ {coord1} ↔ {coord2}: PSNR={psnr:.2f} SSIM={s:.3f} RMSE={rmse:.2f} NCC={ncc if ncc is not None else 'N/A'} | {n_pixels}px")

            if visualize_sample:
                _show_overlap_comparison(ov1_u8, ov2_u8, ov_mask, coord1, coord2)

        except Exception as e:
            print(f"[ERROR] {coord1} ↔ {coord2}: {e}")

    _save_csv(results, output_csv,
              ['tile1_coord','tile2_coord','overlap_pixels','overlap_width','overlap_height','psnr','ssim','rmse','ncc'])
    _print_metric_summary(results)
    return results


def _show_overlap_comparison(ov1, ov2, mask, coord1, coord2):
    g1   = _to_gray(ov1)
    g2   = _to_gray(ov2)
    diff = cv2.absdiff(g1, g2)
    _, thr = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow(f"Overlap 1 - {coord1}", g1)
    cv2.imshow(f"Overlap 2 - {coord2}", g2)
    cv2.imshow("Difference", diff)
    cv2.imshow("Threshold", thr)
    cv2.imshow("Mask", (mask * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
# VISUALISATION HELPERS
# ============================================================

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
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    folder_path = "/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/5x5_ecoli"   # ← change this

    # Initialize Performance & Memory Tracker
    tracker = PerformanceTracker(f"SIFT + BFMatcher Pipeline ({CONFIG['feature_method'].upper()})")

    # Step 1
    image_data, grid_info = load_image(folder_path, CONFIG['resize_factor'])
    if CONFIG.get('debug', False):
        fig = visualize_grid_preview(image_data, grid_info)
        plt.show()
    tracker.record_step("1. Image Loading & Grid Preview")

    # Step 2
    overlap_pairs    = calculate_overlap(image_data, grid_info, CONFIG['overlap_percentage'])
    if CONFIG.get('debug', False):
        visualize_overlap_regions(image_data, overlap_pairs, grid_info)
        plt.show()
    overlap_features = extract_overlap_features(image_data, overlap_pairs, CONFIG['feature_method'])
    tracker.record_step("2. Overlap Region & Feature Extraction")

    # Step 3
    match_result = match_overlap_features(overlap_features, CONFIG['feature_method'], CONFIG['lowe_ratio'])
    tracker.record_step("3. BFMatcher Feature Matching")

    # Step 4
    homography_results = calculate_homographies_batch(image_data, match_result, CONFIG['reproj_thresh'])
    if CONFIG.get('evaluate_metrics', False):
        evaluate_homography_reprojection(match_result, homography_results,
                                         output_csv="homography_reprojection.csv")
    tracker.record_step("4. Homography RANSAC & Reprojection Report")

    # Step 4.5 — evaluate BEFORE stitching, using homography results
    transforms, reference, reachable_images = calculate_all_transforms(image_data, homography_results)
    canvas_w, canvas_h, offset_x, offset_y  = calculate_optimal_canvas(image_data, transforms)
    if CONFIG.get('evaluate_metrics', False):
        evaluate_overlap_metrics(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            homography_results=homography_results,   # adjacent-only -- O(N) not O(N²)
            output_csv="overlap_metrics.csv",
            visualize_sample=False,
        )
    tracker.record_step("4.5 Overlap Quality Metrics Evaluation")

    # Step 5 — blend
    blended, debug_info = blend_panorama(
        image_data, transforms, reachable_images,
        canvas_w, canvas_h, offset_x, offset_y
    )
    if CONFIG.get('debug', False):
        visualize_blending_debug(debug_info, blended)
        plt.show()

    # ROI crop
    roi_info = find_roi(blended, min_threshold=1, debug=True)
    final_result, roi_stats = extract_roi(blended, roi_info, padding=10)

    # Save
    save_path = str(Path(folder_path) / "result_final_SIFTBf.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {save_path}")
    tracker.record_step("5. Feather Blending, ROI Crop & Saving")

    # Output Benchmark Summary Report
    tracker.print_summary()

    if CONFIG.get('debug', False):
        plt.figure(figsize=(20, 10))
        plt.imshow(final_result)
        plt.axis('off')
        plt.show()
