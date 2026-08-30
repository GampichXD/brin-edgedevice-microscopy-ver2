"""
Tile Stitching Pipeline -- SuperPoint + LightGlue Variant
==========================================================
Feature extraction             : SuperPoint  (ETH CVG LightGlue repo)
Feature matching               : LightGlue  (features='superpoint')
Homography + RANSAC            : cv2.findHomography  (unchanged)
Reprojection error             : calculate_reprojection_error (unchanged)
NCC evaluation                 : compute_ncc / calculate_overlap_ncc (unchanged)
Overlap quality metrics        : PSNR, SSIM, RMSE, NCC (unchanged)
Feather blending               : blend_panorama (unchanged)
ROI crop                       : find_roi / extract_roi (unchanged)

Why SuperPoint + LightGlue over SIFT + LightGlue?
  - SuperPoint is a fully learned detector+descriptor — keypoints and
    256-dim descriptors are jointly optimised end-to-end on homographic
    adaptation, giving superior repeatability on indoor/microscope textures.
  - 256-dim descriptors encode richer context than SIFT's 128-dim.
  - LightGlue was originally benchmarked with SuperPoint and achieves its
    best published accuracy with this combination.

Key difference from SIFT variant:
  - Extractor : SuperPoint  (detection_threshold, no edge_threshold)
  - Matcher   : LightGlue(features='superpoint')
  - Descriptors are (N, 256), not (N, 128)

Installation:
    See requirements_LightGlue.txt and INSTALL.md in this directory.

    Quick start (GPU, CUDA 12.1):
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install -r requirements_LightGlue.txt

    CPU-only:
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements_LightGlue.txt
"""

import re
import csv
import traceback
import warnings
import os
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
import tensorrt as trt
from kornia.geometry.transform import resize as kornia_resize
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd   # remove_batch_dim helper

warnings.filterwarnings('ignore')
cv2.ocl.setUseOpenCL(False)

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
    def __init__(self, name="Pipeline Benchmark", backend=None, onnx_providers=None):
        self.name = name
        self.backend = backend                # 'pytorch' | 'onnx' | None
        self.onnx_providers = onnx_providers   # actual providers in use, when backend == 'onnx'
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.step_times = {}
        self._ext_gpu_peak_mb = 0.0            # nvidia-smi-sampled peak (covers onnx/tensorrt
                                                # backends, whose CUDA allocations torch.cuda can't see)

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
        mb = self._query_nvidia_smi_mem()
        if mb is not None:
            self._ext_gpu_peak_mb = max(self._ext_gpu_peak_mb, mb)

    def _query_nvidia_smi_mem(self):
        """Per-process GPU memory (MB) via `nvidia-smi`, sampled at each
        record_step() to build a running peak (mirrors what torch.cuda's own
        peak counter does for the pytorch backend). Returns None when
        nvidia-smi isn't present -- e.g. on Jetson, which has no discrete
        VRAM to query: GPU allocations there share system RAM and are
        already reflected in the Peak RAM figure printed below."""
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if out.returncode != 0:
            return None
        pid = os.getpid()
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2:
                try:
                    if int(parts[0]) == pid:
                        return float(parts[1])
                except ValueError:
                    continue
        return None

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
        if self.backend == 'onnx':
            providers = self.onnx_providers or ['unknown']
            active = providers[0]
            label = {
                'CUDAExecutionProvider':      'GPU (ONNX Runtime CUDAExecutionProvider)',
                'TensorrtExecutionProvider':   'GPU (ONNX Runtime TensorrtExecutionProvider)',
                'CPUExecutionProvider':        'CPU (ONNX Runtime CPUExecutionProvider)',
            }.get(active, f'{active} (ONNX Runtime)')
            print(f"    - SuperPoint Device    : {label}")
            print(f"    - Matcher Device       : {label}")
        elif self.backend == 'tensorrt':
            sp_prec = CONFIG.get('trt_runtime_precision_sp', '?')
            lg_prec = CONFIG.get('trt_runtime_precision_lg', '?')
            print(f"    - SuperPoint Device    : GPU (TensorRT {sp_prec} engine)")
            print(f"    - Matcher Device       : GPU (TensorRT {lg_prec} engine)")
        else:
            device_label = 'GPU (PyTorch CUDA)' if torch.cuda.is_available() else 'CPU (PyTorch)'
            print(f"    - SuperPoint Device    : {device_label}")
            print(f"    - Matcher Device       : {device_label}")
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
            print(f"    - Current GPU VRAM    : {curr_gpu:.2f} MB   (PyTorch CUDA allocator)")
            print(f"    - Peak GPU VRAM       : {peak_gpu:.2f} MB ({peak_gpu / 1024:.2f} GB)   (PyTorch CUDA allocator)")
        elif self._ext_gpu_peak_mb > 0:
            print(f"    - Peak GPU VRAM       : {self._ext_gpu_peak_mb:.2f} MB "
                  f"({self._ext_gpu_peak_mb / 1024:.2f} GB)   (nvidia-smi, sampled per step)")
        elif self.backend in ('onnx', 'tensorrt'):
            print(f"    - Peak GPU VRAM       : {peak_ram:.2f} MB ({peak_ram / 1024:.2f} GB)   "
                  f"(== Peak RAM above -- Jetson has no discrete VRAM to query separately; GPU "
                  f"allocations share unified system RAM, already counted there)")
        else:
            print("    - GPU VRAM            : N/A (Running on CPU)")
        print("=" * 65 + "\n")


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # --- General ---
    'resize_factor':        1.0,
    'overlap_percentage':   0.4,
    'canvas_padding':       50,
    'display_feather_distance':  30,  # soft feather for human viewing
    'analysis_feather_distance': 0,   # hard seam for analysis/training copy
    'feather_distance':     30,       # legacy alias — used by blend_panorama
    'save_analysis_copy':   False,     # also save a hard-seam copy alongside the soft display one
    # 'cuda' moves blend_panorama's per-tile blend arithmetic (masking, feather-weight
    # computation, canvas accumulation) onto the GPU via PyTorch tensors -- this is the
    # O(canvas_size x 3 channels x num_tiles) part that gets slow on grids beyond ~5x5.
    # The geometric warp + mask erode/distanceTransform stay on CPU via OpenCV either way
    # (this build's OpenCV has no CUDA support to swap those to). Set to 'cpu' to disable.
    'feather_blend_device': 'cpu' if torch.cuda.is_available() else 'cpu',
    'normalization_method': 'skimage_histogram_match',

    # --- Weights (local path — no internet required after setup) ---
    'weights_dir': os.path.join(os.path.dirname(__file__), 'Weights'),

    # --- Backend ---
    # 'pytorch'   -- official `lightglue` package (adaptive early-exit/pruning enabled below).
    # 'onnx'      -- ONNX Runtime, models from onnx_export/export_to_onnx.py (early-exit/pruning
    #                is hardcoded off in that export -- always runs the full 9-layer network, so
    #                match counts will differ from the pytorch backend; see onnx_export/lightglue_onnx).
    # 'tensorrt'  -- raw TensorRT Python API against the .engine files built by
    #                onnx_export/build_tensorrt_engines.py (see 'trt_runtime_precision_sp'/'_lg'
    #                below). Same architecture/caveats as onnx (full 9-layer LightGlue, no
    #                early-exit), plus whatever that precision tier actually does on this graph --
    #                see that script's docstring.
    'backend':  'tensorrt',  # 'pytorch' | 'onnx' | 'tensorrt'
    'onnx_dir': os.path.join(os.path.dirname(__file__), 'ONNX'),
    # Long-side resize applied before the ONNX SuperPoint model, replicating what the official
    # `lightglue.SuperPoint.preprocess_conf = {"resize": 1024}` does inside `.extract()` for the
    # pytorch backend. Must match for the two backends to be comparable.
    'onnx_sp_resize': 1024,

    # --- TensorRT engine build (see onnx_export/build_tensorrt_engines.py) ---
    # Not a runtime backend toggle -- .engine files must be rebuilt (re-run that script) whenever
    # sp_max_keypoints/onnx_sp_resize/these shape bounds change, same caveat as the onnx backend's
    # keypoint cap. TensorRT needs an explicit min/opt/max shape profile per dynamic input; SuperPoint's
    # image is always resized so its long side == onnx_sp_resize, so H and W are each bounded by that.
    'trt_dir':          os.path.join(os.path.dirname(__file__), 'TensorRT'),
    # int4 was tried and dropped: trtexec's bare --int4 is a silent no-op on a
    # plain (non-Q/DQ) ONNX graph like ours -- see build_tensorrt_engines.py's
    # docstring. int8 (implicit, no calibration cache) gives real ~4x weight
    # compression on this graph instead.
    'trt_precisions':   ['fp32', 'fp16', 'int8'],   # engines to build; trtexec flag per precision
    'trt_sp_shape_min': (256, 256),                 # (H, W) floor for the SuperPoint image profile
    'trt_sp_shape_opt': (1024, 600),                # (H, W) representative shape to optimize for
    'trt_sp_shape_max': (1024, 1024),               # (H, W) ceiling == (onnx_sp_resize, onnx_sp_resize)
    # Which built engine the 'tensorrt' backend loads at runtime, per model (each must be one of
    # trt_precisions above, and must actually have been built -- run build_tensorrt_engines.py
    # first). Both default to fp16 -- the only tier confirmed correct on real data for both
    # models. DO NOT set trt_runtime_precision_sp = 'int8': it builds, passes trtexec's own
    # (random-data) inference self-check, and is genuinely ~2.8x smaller, but on real images its
    # score head outputs exactly -1.0 for all sp_max_keypoints entries -- i.e. zero real
    # keypoints, silently. That's uncalibrated INT8 corrupting the detector head, not a build
    # config issue; the fix would be a proper --calib pass with representative data (not done
    # here). See build_tensorrt_engines.py's docstring.
    'trt_runtime_precision_sp': 'fp16',
    'trt_runtime_precision_lg': 'fp16',  # int8 is a measured no-op (== fp32) for this graph anyway
    'trt_lg_kpts_min':  1,                          # LightGlue keypoint-count profile floor

    # --- SuperPoint extractor ---
    'sp_max_keypoints':       1024,   # keypoints per ROI; -1 = unlimited (reduced for edge devices)
    'sp_detection_threshold': 0.005,  # lower → more keypoints detected

    # --- LightGlue matcher ---
    # Re-enabled (0.95 / 0.99) adaptive depth pruning and early-exit for edge deployment on Orin Nano.
    'lg_depth_confidence': 0.95,     # early-exit confidence (0-1; -1 = off)
    'lg_width_confidence': 0.99,     # pruning confidence   (0-1; -1 = off)
    'lg_filter_threshold': 0.1,    # match score threshold

    # --- RANSAC / homography ---
    'reproj_thresh':    4.0,   # px -- reprojection threshold for RANSAC
    'min_inlier_ratio': 0.15,  # informational -- not used to reject pairs

    # --- Device ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'debug': False,                 # Gate visualization plotting to avoid headless display hangs
    'evaluate_metrics': False,       # Gate skimage PSNR/SSIM/NCC CPU metrics calculation
    'enable_benchmark': True,       # Toggle Performance & Memory Tracker benchmarking
    'use_amp': False,                # Toggle PyTorch Automatic Mixed Precision (AMP)
    'verbose_pair_metrics': False,   # Print PSNR/SSIM/NCC/RepError per pair (off = final summary only)
}


# ============================================================
# MODEL — instantiated once, reused for all pairs
# ============================================================

def build_extractor_matcher(cfg=None):
    """
    Instantiate SuperPoint extractor and LightGlue matcher
    using LOCAL pretrained weights (no internet required).

    Weight files expected in cfg['weights_dir']:
        superpoint_v1.pth
        superpoint_lightglue_v0-1.pth
    """
    cfg    = cfg or CONFIG
    device = torch.device(cfg['device'])

    weights_dir = cfg['weights_dir']

    # Verify weight files exist before attempting to load
    required = {
        'superpoint_v1.pth':                  'SuperPoint detector',
        'superpoint_lightglue_v0-1.pth': 'LightGlue (superpoint) matcher',
    }
    for fname, label in required.items():
        fpath = os.path.join(weights_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[ERROR] {label} weights not found: {fpath}\n"
                f"Run download_weights.py first to download all weights."
            )

    # Redirect torch.hub cache to our local weights folder.
    # torch.hub.load_state_dict_from_url checks
    # <hub_dir>/checkpoints/<filename> before downloading.
    # Setting hub dir to weights_dir makes it look there directly.
    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(weights_dir)

    # Create the checkpoints subfolder symlink that torch.hub expects
    checkpoints_dir = os.path.join(weights_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Copy weight files into the checkpoints subfolder if not already there
    for fname in required:
        src  = os.path.join(weights_dir, fname)
        link = os.path.join(checkpoints_dir, fname)
        if not os.path.exists(link):
            shutil.copy2(src, link)

    try:
        extractor = (
            SuperPoint(max_num_keypoints=cfg['sp_max_keypoints'],
                       detection_threshold=cfg['sp_detection_threshold'])
            .eval()
            .to(device)
        )
        matcher = (
            LightGlue(features='superpoint',
                      depth_confidence=cfg['lg_depth_confidence'],
                      width_confidence=cfg['lg_width_confidence'],
                      filter_threshold=cfg['lg_filter_threshold'])
            .eval()
            .to(device)
        )
    finally:
        # Always restore original hub dir, even if loading fails
        torch.hub.set_dir(original_hub_dir)

    print(f"[INFO] SuperPoint loaded from: {os.path.join(weights_dir, 'superpoint_v1.pth')}")
    print(f"[INFO] LightGlue (SP) loaded from: "
          f"{os.path.join(weights_dir, 'superpoint_lightglue_v0-1.pth')}")
    print(f"[INFO] Device: {device}")
    return extractor, matcher, device


def build_onnx_sessions(cfg=None):
    """
    Instantiate ONNX Runtime sessions for SuperPoint + LightGlue, exported by
    onnx_export/export_to_onnx.py. Prefers CUDAExecutionProvider when the
    installed onnxruntime build offers it, otherwise falls back to CPU.
    """
    cfg      = cfg or CONFIG
    onnx_dir = cfg['onnx_dir']

    required = {
        'superpoint.onnx':            'SuperPoint (ONNX)',
        'superpoint_lightglue.onnx':  'LightGlue (ONNX)',
    }
    for fname, label in required.items():
        fpath = os.path.join(onnx_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[ERROR] {label} not found: {fpath}\n"
                f"Run onnx_export/export_to_onnx.py first."
            )

    available = ort.get_available_providers()
    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available] \
                or ['CPUExecutionProvider']

    sp_sess = ort.InferenceSession(os.path.join(onnx_dir, 'superpoint.onnx'), providers=providers)
    lg_sess = ort.InferenceSession(os.path.join(onnx_dir, 'superpoint_lightglue.onnx'), providers=providers)

    print(f"[INFO] SuperPoint ONNX loaded from: {os.path.join(onnx_dir, 'superpoint.onnx')}")
    print(f"[INFO] LightGlue ONNX loaded from: {os.path.join(onnx_dir, 'superpoint_lightglue.onnx')}")
    print(f"[INFO] ONNX Runtime providers: {providers}")
    if providers == ['CPUExecutionProvider']:
        print("[WARN] Running ONNX backend on CPU -- timing comparisons against the "
              "CUDA pytorch backend will not be apples-to-apples (install onnxruntime-gpu "
              "or use the TensorRT engine for a fair speed comparison).")
    return sp_sess, lg_sess


class _TRTOutputAllocator(trt.IOutputAllocator):
    """
    TensorRT 10.x callback used during execute_async_v3: called once per
    output tensor with the number of bytes actually needed (which for a
    data-dependent output like LightGlue's matches0/mscores0 isn't known
    until the network runs), and again afterward with the resulting shape.
    Backs the buffer with a plain torch CUDA tensor so no pycuda/cuda-python
    dependency is needed -- SP_LG.py already requires torch+CUDA.
    """
    def __init__(self, torch_dtype):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.buf = None
        self.shape = None

    def reallocate_output(self, tensor_name, memory, size, alignment):
        n_elem = max(-(-size // self.torch_dtype.itemsize), 1)  # ceil div
        self.buf = torch.empty((n_elem,), device='cuda', dtype=self.torch_dtype)
        return self.buf.data_ptr()

    def notify_shape(self, tensor_name, shape):
        self.shape = tuple(shape)


class TRTEngine:
    """
    Loads one serialized .engine and runs it via TensorRT 10's
    execute_async_v3 API. Every input/output is backed by a torch CUDA
    tensor (via .data_ptr()) -- static and data-dependent (NonZero-derived)
    output shapes are both handled through _TRTOutputAllocator, so callers
    don't need to know which kind a given output is.
    """
    _DTYPE_TRT_TO_TORCH = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            # Observed transient on this Jetson: create_execution_context()
            # returns None under a momentary GPU memory allocator hiccup
            # (NvMapMemAllocInternalTagged .../ NvMapMemHandleAlloc error in
            # the TRT log) rather than raising -- fails loud here instead of
            # a confusing 'NoneType has no attribute set_input_shape' later.
            raise RuntimeError(
                f"TensorRT create_execution_context() returned None for {engine_path} "
                f"(often a transient GPU memory allocator hiccup on this device -- retry)."
            )
        self.stream = torch.cuda.Stream()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names  = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def infer(self, inputs):
        """inputs: {name: contiguous torch.cuda.Tensor}. Returns {name: torch.cuda.Tensor}
        sliced/reshaped to the actual output shape for this call."""
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        allocators = {}
        for name in self.output_names:
            torch_dtype = self._DTYPE_TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            alloc = _TRTOutputAllocator(torch_dtype)
            self.context.set_output_allocator(name, alloc)
            allocators[name] = alloc

        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        outputs = {}
        for name, alloc in allocators.items():
            n_elem = 1
            for d in alloc.shape:
                n_elem *= d
            outputs[name] = alloc.buf[:n_elem].view(alloc.shape)
        return outputs


def build_tensorrt_sessions(cfg=None):
    """
    Load the SuperPoint + LightGlue TensorRT engines built by
    onnx_export/build_tensorrt_engines.py, independently for
    cfg['trt_runtime_precision_sp'] and cfg['trt_runtime_precision_lg'].
    """
    cfg          = cfg or CONFIG
    trt_dir      = cfg['trt_dir']
    sp_precision = cfg['trt_runtime_precision_sp']
    lg_precision = cfg['trt_runtime_precision_lg']

    required = {
        f'superpoint_{sp_precision}.engine':           'SuperPoint (TensorRT)',
        f'superpoint_lightglue_{lg_precision}.engine': 'LightGlue (TensorRT)',
    }
    for fname, label in required.items():
        fpath = os.path.join(trt_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[ERROR] {label} not found: {fpath}\n"
                f"Run onnx_export/build_tensorrt_engines.py first."
            )

    sp_engine = TRTEngine(os.path.join(trt_dir, f'superpoint_{sp_precision}.engine'))
    lg_engine = TRTEngine(os.path.join(trt_dir, f'superpoint_lightglue_{lg_precision}.engine'))

    print(f"[INFO] SuperPoint TensorRT engine loaded: superpoint_{sp_precision}.engine")
    print(f"[INFO] LightGlue TensorRT engine loaded: superpoint_lightglue_{lg_precision}.engine")
    print(f"[INFO] TensorRT precision: SuperPoint={sp_precision}  LightGlue={lg_precision}")
    return sp_engine, lg_engine


def build_pipeline(cfg=None):
    """
    Dispatch on cfg['backend'] and return a uniform pipeline dict consumed by
    extract_overlap_features() / match_overlap_features(), so the rest of the
    pipeline (homography, blending, metrics) doesn't need to know which
    backend produced the keypoints/matches.
    """
    cfg = cfg or CONFIG
    if cfg['backend'] == 'onnx':
        sp_sess, lg_sess = build_onnx_sessions(cfg)
        return {'type': 'onnx', 'sp_sess': sp_sess, 'lg_sess': lg_sess}
    elif cfg['backend'] == 'tensorrt':
        sp_engine, lg_engine = build_tensorrt_sessions(cfg)
        return {'type': 'tensorrt', 'sp_engine': sp_engine, 'lg_engine': lg_engine}
    elif cfg['backend'] == 'pytorch':
        extractor, matcher, device = build_extractor_matcher(cfg)
        return {'type': 'pytorch', 'extractor': extractor, 'matcher': matcher, 'device': device}
    else:
        raise ValueError(f"Unknown backend: {cfg['backend']!r} (expected 'pytorch', 'onnx', or 'tensorrt')")


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

    Returns
    -------
    image_data : dict  {(x, y): {'image', 'image_gray', 'filename'}}
    grid_info  : dict  {'dimensions', 'x_range', 'y_range', 'unique_x', 'unique_y'}
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' tidak ditemukan")

    focused_pat = re.compile(r'Focused_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.jpg')
    tile_pat    = re.compile(r'tile_r(\d+)_c(\d+)\.jpg')

    image_data = {}
    for fp in sorted(folder_path.glob('*.jpg')):
        coords = None
        m = focused_pat.match(fp.name)
        if m:
            coords = (float(m.group(1)), float(m.group(2)))
        else:
            m = tile_pat.match(fp.name)
            if m:
                coords = (float(m.group(2)), float(m.group(1)))  # col->x, row->y
        if coords is None:
            continue

        try:
            raw = cv2.imread(str(fp))
            if raw is None:
                print(f"[WARN] Gagal memuat (None): {fp.name}")
                continue
            img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            if resize_factor != 1.0:
                h, w    = img_rgb.shape[:2]
                img_rgb = cv2.resize(img_rgb,
                                     (int(w * resize_factor), int(h * resize_factor)))
            image_data[coords] = {
                'image':      img_rgb,
                'image_gray': cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
                'filename':   fp.name,
            }
        except Exception as e:
            print(f"[ERROR] Gagal memuat {fp.name}: {e}")
            traceback.print_exc()

    if not image_data:
        raise ValueError("Tidak ada gambar yang ditemukan di folder.")

    xs = [c[0] for c in image_data]
    ys = [c[1] for c in image_data]
    unique_x = sorted(set(xs), reverse=True)
    unique_y = sorted(set(ys))

    grid_info = {
        'dimensions': (len(unique_x), len(unique_y)),
        'x_range':    (min(xs), max(xs)),
        'y_range':    (min(ys), max(ys)),
        'unique_x':   unique_x,
        'unique_y':   unique_y,
    }
    print(f"{len(image_data)} gambar dimuat dari {folder_path}")
    print(f"Dimensi Grid  : {grid_info['dimensions'][0]} x {grid_info['dimensions'][1]}")
    print(f"Rentang X     : {grid_info['x_range']}")
    print(f"Rentang Y     : {grid_info['y_range']}")
    return image_data, grid_info


def _normalise_axes(axes, rows, cols):
    """Guarantee axes is always list-of-lists regardless of grid shape."""
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
    grid_w, grid_h   = grid_info['dimensions']
    unique_x, unique_y = grid_info['unique_x'], grid_info['unique_y']
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3 + 1))
    axes = _normalise_axes(axes, grid_h, grid_w)
    for row in axes:
        for ax in row:
            ax.axis('off')
    for coords, data in image_data.items():
        xi      = unique_x.index(coords[0])
        yi      = unique_y.index(coords[1])
        row_idx = grid_h - 1 - yi
        axes[row_idx][xi].imshow(data['image'])
        axes[row_idx][xi].set_title(f'({coords[0]},{coords[1]})', fontsize=16)
    plt.suptitle(f'Image Grid: {grid_w}x{grid_h} | {len(image_data)} images loaded',
                 fontsize=20)
    plt.tight_layout()
    return fig


# ============================================================
# STEP 2 : Overlap Region Definition
# ============================================================

def calculate_overlap(image_data, grid_info, overlap_percentage=0.4):
    """
    Enumerate all adjacent tile pairs and compute the expected overlap ROI
    for each pair (horizontal = right neighbour; vertical = upper neighbour).

    Returns
    -------
    overlap_pairs : list of dicts
        Each dict has keys: coord1, coord2, direction, region1, region2.
    """
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
                    'region1':   {'x': w - ow, 'y': 0,
                                  'width': ow, 'height': h},
                    'region2':   {'x': 0, 'y': 0,
                                  'width': int(nb_w * overlap_percentage),
                                  'height': nb_h},
                })

        # Upper neighbour
        if yi < len(unique_y) - 1:
            upper_coord = (x, unique_y[yi + 1])
            if upper_coord in image_data:
                nb_h, nb_w = image_data[upper_coord]['image'].shape[:2]
                oh    = int(h    * overlap_percentage)
                nb_oh = int(nb_h * overlap_percentage)
                overlap_pairs.append({
                    'coord1':    coord,
                    'coord2':    upper_coord,
                    'direction': 'vertical',
                    'region1':   {'x': 0, 'y': 0,
                                  'width': w, 'height': oh},
                    'region2':   {'x': 0, 'y': nb_h - nb_oh,
                                  'width': nb_w, 'height': nb_oh},
                })

    print(f"{len(overlap_pairs)} pasangan overlap ditemukan")
    return overlap_pairs


def visualize_overlap_regions(image_data, overlap_pairs, grid_info):
    grid_w, grid_h   = grid_info['dimensions']
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
# STEP 3 : Feature Extraction -- SuperPoint
# ============================================================

def _image_to_tensor(gray_uint8, device):
    """
    SuperPoint extractor expects float32 tensor in [0, 1] with shape (1, 1, H, W).
    """
    t = torch.from_numpy(gray_uint8).float() / 255.0   # (H, W)
    return t.unsqueeze(0).unsqueeze(0).to(device)       # (1, 1, H, W)


def _extract_region_pytorch(roi, extractor, device):
    """SuperPoint via the official `lightglue` package. `.extract()` applies
    its own preprocess_conf resize=1024 internally and rescales keypoints
    back to `roi`'s pixel space before returning."""
    tensor = _image_to_tensor(roi, device)
    use_amp = CONFIG.get('use_amp', True)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    amp_dtype = torch.bfloat16 if device_type == 'cpu' else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            feats = extractor.extract(tensor)
    feats = rbd(feats)                                   # remove batch dim

    kps   = feats['keypoints'].cpu().numpy()             # (N, 2)
    descs = feats['descriptors'].cpu().numpy()           # (N, 256)
    scs   = feats['keypoint_scores'].cpu().numpy()       # (N,)
    if len(kps) == 0:
        return None
    return kps, descs, scs


def _extract_region_onnx(roi, sp_sess, resize_size=1024):
    """SuperPoint via ONNX Runtime. Manually replicates the resize=1024 +
    rescale-back that the pytorch backend's `.extract()` does internally,
    so both backends see the network at the same effective resolution."""
    h0, w0 = roi.shape
    t = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,H,W)
    t = kornia_resize(t, resize_size, side='long', antialias=True, align_corners=None)
    scale = np.array([t.shape[-1] / w0, t.shape[-2] / h0], dtype=np.float32)  # (sx, sy)

    kps, scs, descs = sp_sess.run(None, {'image': t.numpy().astype(np.float32)})
    kps, scs, descs = kps[0], scs[0], descs[0]

    # The exported graph's top_k_keypoints is unconditional (see onnx_export/
    # lightglue_onnx/superpoint.py) so TensorRT can build a static-shape
    # engine -- it always returns exactly sp_max_keypoints entries, padding
    # with score=-1 dummy keypoints when fewer real ones were found. Drop
    # those here so onnx/tensorrt behave like the pytorch backend (variable
    # keypoint count), not like they always hit the cap.
    real = scs > 0
    kps, scs, descs = kps[real], scs[real], descs[real]
    if len(kps) == 0:
        return None

    kps = (kps.astype(np.float32) + 0.5) / scale - 0.5   # ROI-local pixel space
    return kps, descs.astype(np.float32), scs.astype(np.float32)


def _extract_region_tensorrt(roi, sp_engine, resize_size=1024):
    """SuperPoint via the raw TensorRT engine. Same resize/rescale-back and
    padding-filter as _extract_region_onnx -- the two exported graphs are
    architecturally identical, only the runtime differs."""
    h0, w0 = roi.shape
    t = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0).cuda() / 255.0  # (1,1,H,W)
    t = kornia_resize(t, resize_size, side='long', antialias=True, align_corners=None).contiguous()
    scale = np.array([t.shape[-1] / w0, t.shape[-2] / h0], dtype=np.float32)  # (sx, sy)

    out = sp_engine.infer({'image': t})
    kps   = out['keypoints'][0].float().cpu().numpy()
    scs   = out['scores'][0].cpu().numpy()
    descs = out['descriptors'][0].cpu().numpy()

    # Same top_k_keypoints padding as the onnx backend -- drop score<=0 dummies.
    real = scs > 0
    kps, scs, descs = kps[real], scs[real], descs[real]
    if len(kps) == 0:
        return None

    kps = (kps.astype(np.float32) + 0.5) / scale - 0.5   # ROI-local pixel space
    return kps, descs.astype(np.float32), scs.astype(np.float32)


def _extract_region(gray, region, pipeline):
    """
    Run SuperPoint (pytorch or onnx backend, per `pipeline['type']`) on one
    ROI crop.

    Returns
    -------
    (keypoints, descriptors, scores) in full-image pixel coordinates,
    or None if no keypoints found.

    keypoints   : (N, 2)   float32  -- (x, y), shifted to full-image space
    descriptors : (N, 256) float32  -- 256-dim SuperPoint descriptors
    scores      : (N,)     float32  -- detector confidence scores
    """
    x, y, w, h = region['x'], region['y'], region['width'], region['height']
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # Apply CLAHE to the ROI crop before detection to improve keypoint density
    # on low-contrast agar images.  Applied only to the crop, NOT to the stored
    # image_gray, so photometric metrics remain unaffected.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    if pipeline['type'] == 'onnx':
        result = _extract_region_onnx(roi, pipeline['sp_sess'],
                                       CONFIG.get('onnx_sp_resize', 1024))
    elif pipeline['type'] == 'tensorrt':
        result = _extract_region_tensorrt(roi, pipeline['sp_engine'],
                                          CONFIG.get('onnx_sp_resize', 1024))
    else:
        result = _extract_region_pytorch(roi, pipeline['extractor'], pipeline['device'])
    if result is None:
        return None

    kps, descs, scs = result
    # Shift from ROI-local to full-image coordinates
    kps = kps.copy()
    kps[:, 0] += x
    kps[:, 1] += y
    return kps, descs, scs


def extract_overlap_features(image_data, overlap_pairs, pipeline):
    """
    Run SuperPoint on every overlap ROI (pytorch or onnx, per `pipeline['type']`).

    Returns
    -------
    overlap_features : dict
        {(coord1, coord2): {
            'coord1', 'coord2', 'direction',
            'keypoints1'  (N1, 2),  'keypoints2'  (N2, 2),
            'descriptors1'(N1,256), 'descriptors2'(N2,256),
            'scores1'     (N1,),    'scores2'     (N2,),
            'region1', 'region2'
        }}
    """
    overlap_features = {}
    total_kp = 0

    for i, pair in enumerate(overlap_pairs, 1):
        coord1, coord2 = pair['coord1'], pair['coord2']
        r1, r2         = pair['region1'], pair['region2']
        direction      = pair['direction']

        try:
            feats1 = _extract_region(image_data[coord1]['image_gray'],
                                     r1, pipeline)
            feats2 = _extract_region(image_data[coord2]['image_gray'],
                                     r2, pipeline)

            if feats1 is None or feats2 is None:
                print(f"[SKIP] {coord1}<->{coord2}: SuperPoint returned nothing")
                continue

            kp1, desc1, sc1 = feats1
            kp2, desc2, sc2 = feats2

            total_kp += len(kp1) + len(kp2)
            overlap_features[(coord1, coord2)] = {
                'coord1': coord1, 'coord2': coord2, 'direction': direction,
                'keypoints1':   kp1,   'keypoints2':   kp2,
                'descriptors1': desc1, 'descriptors2': desc2,
                'scores1':      sc1,   'scores2':      sc2,
                'region1': r1, 'region2': r2,
            }
            print(f"Pair {i}/{len(overlap_pairs)}: {coord1}<->{coord2} "
                  f"| kp: {len(kp1)}+{len(kp2)}")

        except Exception as e:
            print(f"[ERROR] Feature extraction {coord1}<->{coord2}: {e}")
            traceback.print_exc()

    print(f"\n SUPERPOINT EXTRACTION SUMMARY:")
    print(f"   Pairs with features : {len(overlap_features)}")
    print(f"   Total keypoints     : {total_kp}")
    # Diagnostic: log per-ROI keypoint count to validate sp_detection_threshold.
    # If per-ROI mean falls below ~50, lower sp_detection_threshold to 0.001-0.003.
    if overlap_features:
        kp_counts = [len(fd['keypoints1']) + len(fd['keypoints2'])
                     for fd in overlap_features.values()]
        print(f"   Keypoints per ROI (min/mean/max): "
              f"{min(kp_counts)//2:.0f} / {np.mean(kp_counts)/2:.0f} / {max(kp_counts)//2:.0f}")
    return overlap_features


def visualize_overlap_features(image_data, overlap_features, grid_info):
    """Visualize extracted SuperPoint keypoints overlaid on each tile."""
    grid_w, grid_h   = grid_info['dimensions']
    unique_x, unique_y = grid_info['unique_x'], grid_info['unique_y']
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3 + 1))
    axes = _normalise_axes(axes, grid_h, grid_w)
    for row in axes:
        for ax in row:
            ax.axis('off')

    coord_kps = defaultdict(list)
    for (c1, c2), fd in overlap_features.items():
        for kp in fd['keypoints1']:
            coord_kps[c1].append((kp, fd['direction']))
        for kp in fd['keypoints2']:
            coord_kps[c2].append((kp, fd['direction']))

    for coords, data in image_data.items():
        xi      = unique_x.index(coords[0])
        yi      = unique_y.index(coords[1])
        row_idx = grid_h - 1 - yi
        disp    = cv2.cvtColor(data['image_gray'], cv2.COLOR_GRAY2RGB)

        dir_counts = {'horizontal': 0, 'vertical': 0}
        for kp, direction in coord_kps.get(coords, []):
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(disp, (cx, cy), 2, (0, 255, 0), -1)
            cv2.circle(disp, (cx, cy), 3, (0,   0, 0),  1)
            dir_counts[direction] += 1

        axes[row_idx][xi].imshow(disp)
        axes[row_idx][xi].set_title(
            f'{coords}\nH:{dir_counts["horizontal"]} V:{dir_counts["vertical"]}',
            fontsize=10)

    plt.suptitle('SuperPoint Keypoints -- LightGlue extractor (overlap regions only)',
                 fontsize=16)
    plt.tight_layout()
    return fig


# ============================================================
# STEP 4 : Feature Matching -- LightGlue
# ============================================================

def _pack_for_lightglue(kps, descs, scores, image_hw, device):
    """
    Pack SuperPoint features into the dict format LightGlue expects.

    Parameters
    ----------
    kps       : (N, 2)   float32  -- pixel coords
    descs     : (N, 256) float32  -- 256-dim SuperPoint descriptors
    scores    : (N,)     float32  -- detector confidence scores
    image_hw  : (H, W)   tuple   -- full image size (for coord normalisation)
    device    : torch.device

    Returns
    -------
    dict with batched tensors:
        'keypoints'       (1, N, 2)
        'descriptors'     (1, N, 256)
        'keypoint_scores' (1, N)
        'image_size'      (1, 2)  [W, H]
    """
    H, W = image_hw
    return {
        'keypoints':       torch.from_numpy(kps).float().unsqueeze(0).to(device),
        'descriptors':     torch.from_numpy(descs).float().unsqueeze(0).to(device),
        'keypoint_scores': torch.from_numpy(scores).float().unsqueeze(0).to(device),
        'image_size':      torch.tensor([[W, H]], dtype=torch.float32).to(device),
    }


def _normalize_keypoints_onnx(kpts, h, w):
    """Same formula LightGlue applies internally to pixel keypoints before
    the transformer (see lightglue.lightglue.normalize_keypoints), done here
    explicitly since the ONNX graph expects pre-normalised input."""
    size  = np.array([w, h], dtype=np.float32)
    shift = size / 2.0
    scale = size.max() / 2.0
    return (kpts.astype(np.float32) - shift) / scale


def _match_overlap_features_pytorch(overlap_features, image_data, matcher, device):
    match_result  = {}
    for pair_key, fd in overlap_features.items():
        coord1, coord2 = pair_key
        try:
            h1, w1 = image_data[coord1]['image'].shape[:2]
            h2, w2 = image_data[coord2]['image'].shape[:2]

            f0 = _pack_for_lightglue(fd['keypoints1'], fd['descriptors1'],
                                     fd['scores1'], (h1, w1), device)
            f1 = _pack_for_lightglue(fd['keypoints2'], fd['descriptors2'],
                                     fd['scores2'], (h2, w2), device)

            use_amp = CONFIG.get('use_amp', True)
            device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
            amp_dtype = torch.bfloat16 if device_type == 'cpu' else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                    result = matcher({'image0': f0, 'image1': f1})

            result = rbd(result)   # remove batch dim

            matches = result['matches'].cpu().numpy()           # (M, 2) or (N,)
            scores  = result['matching_scores0'].cpu().numpy()  # (N,)

            # LightGlue ≥0.1 returns matches as (M, 2) index pairs
            if matches.ndim == 2:
                idx0 = matches[:, 0]
                idx1 = matches[:, 1]
            else:
                # Older API: matches0[i] = index into kp1; -1 = unmatched
                valid = matches >= 0
                idx0  = np.where(valid)[0]
                idx1  = matches[valid]
                scores = scores[valid]

            if len(idx0) == 0:
                print(f"[SKIP] {coord1}<->{coord2}: no matches returned")
                continue

            mkp1 = fd['keypoints1'][idx0].astype(np.float32)  # (M, 2)
            mkp2 = fd['keypoints2'][idx1].astype(np.float32)  # (M, 2)

            min_kp  = min(len(fd['keypoints1']), len(fd['keypoints2']))
            quality = len(idx0) / min_kp if min_kp > 0 else 0.0

            match_result[pair_key] = {
                'pts1':         mkp1,
                'pts2':         mkp2,
                'match_scores': scores,
                'match_count':  len(idx0),
                'direction':    fd['direction'],
                'quality':      quality,
            }
            print(f"   {coord1}<->{coord2}: {len(idx0)} matches "
                  f"| quality={quality:.3f}")

        except Exception as e:
            print(f"[ERROR] Matching {coord1}<->{coord2}: {e}")
            traceback.print_exc()
    return match_result


def _match_overlap_features_onnx(overlap_features, image_data, lg_sess):
    match_result = {}
    for pair_key, fd in overlap_features.items():
        coord1, coord2 = pair_key
        try:
            h1, w1 = image_data[coord1]['image'].shape[:2]
            h2, w2 = image_data[coord2]['image'].shape[:2]

            n_kp1 = _normalize_keypoints_onnx(fd['keypoints1'], h1, w1)
            n_kp2 = _normalize_keypoints_onnx(fd['keypoints2'], h2, w2)

            matches0, mscores0 = lg_sess.run(
                None,
                {
                    'kpts0': n_kp1[None].astype(np.float32),
                    'kpts1': n_kp2[None].astype(np.float32),
                    'desc0': fd['descriptors1'][None].astype(np.float32),
                    'desc1': fd['descriptors2'][None].astype(np.float32),
                },
            )

            if len(matches0) == 0:
                print(f"[SKIP] {coord1}<->{coord2}: no matches returned")
                continue

            idx0, idx1 = matches0[:, 0], matches0[:, 1]
            mkp1 = fd['keypoints1'][idx0].astype(np.float32)
            mkp2 = fd['keypoints2'][idx1].astype(np.float32)

            min_kp  = min(len(fd['keypoints1']), len(fd['keypoints2']))
            quality = len(idx0) / min_kp if min_kp > 0 else 0.0

            match_result[pair_key] = {
                'pts1':         mkp1,
                'pts2':         mkp2,
                'match_scores': mscores0.astype(np.float32),
                'match_count':  len(idx0),
                'direction':    fd['direction'],
                'quality':      quality,
            }
            print(f"   {coord1}<->{coord2}: {len(idx0)} matches "
                  f"| quality={quality:.3f}")

        except Exception as e:
            print(f"[ERROR] Matching {coord1}<->{coord2}: {e}")
            traceback.print_exc()
    return match_result


def _match_overlap_features_tensorrt(overlap_features, image_data, lg_engine):
    match_result = {}
    for pair_key, fd in overlap_features.items():
        coord1, coord2 = pair_key
        try:
            h1, w1 = image_data[coord1]['image'].shape[:2]
            h2, w2 = image_data[coord2]['image'].shape[:2]

            n_kp1 = _normalize_keypoints_onnx(fd['keypoints1'], h1, w1)
            n_kp2 = _normalize_keypoints_onnx(fd['keypoints2'], h2, w2)

            inputs = {
                'kpts0': torch.from_numpy(n_kp1[None]).float().cuda().contiguous(),
                'kpts1': torch.from_numpy(n_kp2[None]).float().cuda().contiguous(),
                'desc0': torch.from_numpy(fd['descriptors1'][None]).float().cuda().contiguous(),
                'desc1': torch.from_numpy(fd['descriptors2'][None]).float().cuda().contiguous(),
            }
            out = lg_engine.infer(inputs)
            matches0 = out['matches0'].cpu().numpy()
            mscores0 = out['mscores0'].cpu().numpy()

            if len(matches0) == 0:
                print(f"[SKIP] {coord1}<->{coord2}: no matches returned")
                continue

            idx0, idx1 = matches0[:, 0], matches0[:, 1]
            mkp1 = fd['keypoints1'][idx0].astype(np.float32)
            mkp2 = fd['keypoints2'][idx1].astype(np.float32)

            min_kp  = min(len(fd['keypoints1']), len(fd['keypoints2']))
            quality = len(idx0) / min_kp if min_kp > 0 else 0.0

            match_result[pair_key] = {
                'pts1':         mkp1,
                'pts2':         mkp2,
                'match_scores': mscores0.astype(np.float32),
                'match_count':  len(idx0),
                'direction':    fd['direction'],
                'quality':      quality,
            }
            print(f"   {coord1}<->{coord2}: {len(idx0)} matches "
                  f"| quality={quality:.3f}")

        except Exception as e:
            print(f"[ERROR] Matching {coord1}<->{coord2}: {e}")
            traceback.print_exc()
    return match_result


def match_overlap_features(overlap_features, image_data, pipeline):
    """
    Run LightGlue (superpoint mode) on every extracted pair, via the
    pytorch or onnx backend per `pipeline['type']`.

    Returns
    -------
    match_result : dict
        {(coord1, coord2): {
            'pts1':        np.float32 (M, 2)  -- matched kp in image1 (full-image)
            'pts2':        np.float32 (M, 2)  -- matched kp in image2 (full-image)
            'match_scores': np.float32 (M,)
            'match_count': int
            'direction':   str
            'quality':     float   -- matched / min(N1, N2)
        }}

    Note: key names 'pts1'/'pts2' are used (same as LoFTR/EfficientLoFTR/SIFT-LG
    variants) so all downstream functions are interchangeable. The onnx and
    tensorrt backends both run LightGlue with adaptive early-exit/pruning
    hardcoded off (see onnx_export/lightglue_onnx), so match counts will
    differ from the pytorch backend even on identical keypoints -- both are
    "correct", they just run a different amount of compute.
    """
    if pipeline['type'] == 'onnx':
        match_result = _match_overlap_features_onnx(overlap_features, image_data, pipeline['lg_sess'])
    elif pipeline['type'] == 'tensorrt':
        match_result = _match_overlap_features_tensorrt(overlap_features, image_data, pipeline['lg_engine'])
    else:
        match_result = _match_overlap_features_pytorch(overlap_features, image_data,
                                                        pipeline['matcher'], pipeline['device'])

    total_matches = sum(v['match_count'] for v in match_result.values())
    print(f"\n LIGHTGLUE MATCHING SUMMARY:")
    print(f"   Pairs attempted    : {len(overlap_features)}")
    print(f"   Successful pairs   : {len(match_result)}")
    print(f"   Total matches      : {total_matches}")
    if match_result:
        counts = [v['match_count'] for v in match_result.values()]
        quals  = [v['quality']     for v in match_result.values()]
        print(f"   Avg matches/pair   : {np.mean(counts):.1f}")
        print(f"   Avg quality        : {np.mean(quals):.3f}")
    return match_result


def visualize_lightglue_matches(image_data, match_result, top_n=3):
    """
    Side-by-side visualization of the top-N and bottom-N matched pairs.
    Draws subsampled match lines between the two image halves.
    """
    if not match_result:
        print("No match results to visualize")
        return None

    sorted_pairs = sorted(match_result.items(),
                          key=lambda x: x[1]['match_count'], reverse=True)
    top_pairs   = sorted_pairs[:min(top_n, len(sorted_pairs))]
    least_pairs = sorted_pairs[-min(top_n, len(sorted_pairs)):][::-1]

    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 5, 12))
    if top_n == 1:
        axes = axes.reshape(2, 1)

    def _draw_row(pairs, row_idx, label):
        for idx, ((coord1, coord2), result) in enumerate(pairs):
            img1 = image_data[coord1]['image']
            img2 = image_data[coord2]['image']
            pts1 = result['pts1']
            pts2 = result['pts2']
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            combined_h = max(h1, h2)
            combined   = np.zeros((combined_h, w1 + w2, 3), dtype=np.uint8)
            combined[:h1, :w1]      = img1
            combined[:h2, w1:w1+w2] = img2
            ax = axes[row_idx, idx]
            ax.imshow(combined)
            step = max(1, len(pts1) // 50)
            for p1, p2 in zip(pts1[::step], pts2[::step]):
                ax.plot([p1[0], p2[0] + w1], [p1[1], p2[1]],
                        'c-', lw=0.6, alpha=0.6)
                ax.plot(p1[0], p1[1], 'r.', ms=3)
                ax.plot(p2[0] + w1, p2[1], 'g.', ms=3)
            ax.set_title(
                f'{label} #{idx+1}: {coord1}<>{coord2}\n'
                f'{result["match_count"]} matches | q={result["quality"]:.3f}',
                fontsize=9)
            ax.axis('off')

    _draw_row(top_pairs,   0, 'TOP')
    _draw_row(least_pairs, 1, 'LEAST')
    for row in range(2):
        n = len(top_pairs) if row == 0 else len(least_pairs)
        for j in range(n, top_n):
            axes[row, j].axis('off')

    plt.suptitle('SuperPoint + LightGlue Match Analysis -- Best vs Worst Pairs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# STEP 5 : Homography via RANSAC  (unchanged logic)
# ============================================================

def _homography_from_pts(pts1, pts2, reproj_thresh):
    """Estimate transformation between point sets.

    Uses estimateAffinePartial2D (4 DOF: translation + rotation + uniform
    scale) instead of findHomography (8 DOF) because the CNC stage only
    translates in X and Y.  A full perspective model has 6 free parameters
    to absorb noise from repeated colony patterns, making RANSAC susceptible
    to confident-looking but geometrically wrong solutions.  The affine
    result is promoted to a 3×3 matrix for compatibility with warpPerspective.
    """
    if len(pts1) < 4:
        return None
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
    return (H, status)


def calculate_homographies_batch(image_data, match_result, reproj_thresh=None):
    """
    Run RANSAC homography estimation for every matched pair.

    Returns
    -------
    homography_results : dict
        {(coord1, coord2): {
            'homography_matrix': np.ndarray (3,3)
            'inliers':           int
            'total_matches':     int
            'inlier_ratio':      float
            'direction':         str
            'status':            np.ndarray
        }}
    """
    reproj_thresh      = reproj_thresh or CONFIG['reproj_thresh']
    homography_results = {}
    success = failed   = 0

    for (coord1, coord2), info in match_result.items():
        try:
            result = _homography_from_pts(info['pts1'], info['pts2'],
                                          reproj_thresh)
            if result is None:
                print(f"[SKIP] {coord1}<>{coord2}: <4 matches")
                failed += 1
                continue

            H, status = result
            if H is None:
                print(f"[SKIP] {coord1}<>{coord2}: findHomography returned None")
                failed += 1
                continue

            inliers = int(np.sum(status)) if status is not None else 0
            n       = info['match_count']
            homography_results[(coord1, coord2)] = {
                'homography_matrix': H,
                'inliers':           inliers,
                'total_matches':     n,
                'inlier_ratio':      inliers / n if n else 0,
                'direction':         info['direction'],
                'status':            status,
            }
            print(f"   OK {coord1}<>{coord2}: "
                  f"{inliers}/{n} inliers ({inliers / n * 100:.1f}%)")
            success += 1

        except Exception as e:
            print(f"[ERROR] Homography failed {coord1}<>{coord2}: {e}")
            failed += 1

    total = success + failed
    print(f"\n HOMOGRAPHY SUMMARY:")
    print(f"   Success : {success} / {total}")
    if total:
        print(f"   Rate    : {success / total * 100:.1f}%")
    if homography_results:
        ratios = [r['inlier_ratio'] for r in homography_results.values()]
        print(f"   Avg inlier ratio: {np.mean(ratios):.3f} "
              f"(min {min(ratios):.3f}, max {max(ratios):.3f})")
    return homography_results


# ============================================================
# STEP 5.1 : Reprojection Error  (unchanged logic)
# ============================================================

def calculate_reprojection_error(pts1, pts2, H):
    """Project pts1 through H and measure Euclidean distance to pts2."""
    empty = {
        'errors': np.array([], dtype=np.float32),
        'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None,
    }
    if H is None or pts1 is None or pts2 is None or len(pts1) == 0:
        return empty
    src       = np.float32(pts1).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    actual    = np.float32(pts2).reshape(-1, 2)
    errors    = np.linalg.norm(projected - actual, axis=1)
    return {
        'errors': errors,
        'count':  int(errors.size),
        'mean':   float(np.mean(errors)),
        'median': float(np.median(errors)),
        'min':    float(np.min(errors)),
        'max':    float(np.max(errors)),
    }


def evaluate_homography_reprojection(match_result, homography_results,
                                     output_csv="homography_reprojection_sp_lg.csv"):
    """Compute and report reprojection error for every successful pair."""
    results = []
    for (coord1, coord2), info in homography_results.items():
        match_info = match_result.get((coord1, coord2), {})
        pts1 = match_info.get('pts1')
        pts2 = match_info.get('pts2')
        if pts1 is None or pts2 is None:
            continue

        reproj = calculate_reprojection_error(pts1, pts2,
                                              info['homography_matrix'])
        results.append({
            'tile1_coord':   f"({coord1[0]},{coord1[1]})",
            'tile2_coord':   f"({coord2[0]},{coord2[1]})",
            'match_count':   info['total_matches'],
            'inliers':       reproj['count'],
            'mean_error':    round(reproj['mean'],   3),
            'median_error':  round(reproj['median'], 3),
            'min_error':     round(reproj['min'],    3),
            'max_error':     round(reproj['max'],    3),
        })
        if CONFIG.get('verbose_pair_metrics', False):
            print(f"OK {coord1}<>{coord2}: reprojection "
                  f"mean={reproj['mean']:.3f} "
                  f"median={reproj['median']:.3f} "
                  f"min={reproj['min']:.3f} "
                  f"max={reproj['max']:.3f}")

    if results:
        _save_csv(results, output_csv,
                  ['tile1_coord', 'tile2_coord', 'match_count', 'inliers',
                   'mean_error', 'median_error', 'min_error', 'max_error'])
    _print_reprojection_summary(results)
    return results


def _print_reprojection_summary(results):
    if not results:
        return
    print(f"\n REPROJECTION ERROR SUMMARY:")
    for key in ('mean_error', 'median_error', 'min_error', 'max_error'):
        vals = [r[key] for r in results]
        print(f"{key.upper():12s} -- mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} max={np.max(vals):.3f}")


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
    print(f"Reference image: {reference} "
          f"({counts[reference]} connections, closest to grid centre)")
    return reference


def calculate_all_transforms(image_data, homography_results):
    """Build global transforms from every tile to the reference tile."""
    graph     = build_transformation_graph(homography_results)
    reference = select_reference_image(graph)

    transforms, reachable_images = {}, []
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
    """Compute the canvas size that fits all warped tiles without clipping."""
    all_corners = []
    for coord, T in transforms.items():
        h, w    = image_data[coord]['image'].shape[:2]
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

    print(f"   Canvas  : {canvas_w} x {canvas_h}  |  Offset: ({offset_x}, {offset_y})")
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
    """Compute NCC inside the warped overlap region between two tiles."""
    try:
        off  = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]],
                        dtype=np.float32)
        img1 = image_data[coord1]['image'].astype(np.float32)
        img2 = image_data[coord2]['image'].astype(np.float32)
        w1   = cv2.warpPerspective(img1, off @ transforms[coord1],
                                   (canvas_w, canvas_h))
        w2   = cv2.warpPerspective(img2, off @ transforms[coord2],
                                   (canvas_w, canvas_h))

        mask1    = (w1.sum(axis=2) > 0)
        mask2    = (w2.sum(axis=2) > 0)
        overlap  = mask1 & mask2
        n_overlap = int(overlap.sum())
        if n_overlap < 100:
            return None

        g1  = cv2.cvtColor(w1.astype(np.uint8), cv2.COLOR_RGB2GRAY) \
              if w1.ndim == 3 else w1
        g2  = cv2.cvtColor(w2.astype(np.uint8), cv2.COLOR_RGB2GRAY) \
              if w2.ndim == 3 else w2
        ov1 = g1[overlap].astype(np.float32)
        ov2 = g2[overlap].astype(np.float32)

        mean1, mean2 = ov1.mean(), ov2.mean()
        std1,  std2  = ov1.std(),  ov2.std()
        if std1 < 1e-6 or std2 < 1e-6:
            ncc = 0.0
        else:
            ncc = float(np.mean((ov1 - mean1) * (ov2 - mean2) / (std1 * std2)))
            ncc = float(np.clip(ncc, -1.0, 1.0))

        return {'ncc': ncc, 'overlap_pixels': n_overlap}
    except Exception as e:
        print(f"[WARNING] NCC error for {coord1}<>{coord2}: {e}")
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
    """Normalize intensity of two images before metric computation."""
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
    """Warp both tiles to canvas space and extract the overlapping region."""
    off  = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]],
                    dtype=np.float32)
    img1 = image_data[coord1]['image'].astype(np.float32)
    img2 = image_data[coord2]['image'].astype(np.float32)
    w1   = cv2.warpPerspective(img1, off @ transforms[coord1],
                               (canvas_w, canvas_h))
    w2   = cv2.warpPerspective(img2, off @ transforms[coord2],
                               (canvas_w, canvas_h))
    m1   = (w1.sum(axis=2) > 0).astype(np.uint8)
    m2   = (w2.sum(axis=2) > 0).astype(np.uint8)
    ov   = (m1 > 0) & (m2 > 0)

    if ov.sum() < 100:
        return None, None, None, 0

    ov_r1, ov_r2 = w1.copy(), w2.copy()
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
                              output_csv="overlap_metrics_sp_lg.csv",
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
                canvas_w, canvas_h, offset_x, offset_y)
            if ov1 is None:
                continue

            a = np.clip(ov1, 0, 255).astype(np.uint8)
            b = np.clip(ov2, 0, 255).astype(np.uint8)

            psnr = compute_psnr(a, b, norm_method)
            s    = compute_ssim(a, b, norm_method)
            rmse = compute_rmse(a, b, norm_method)

            # NCC computed inline from already-warped crops -- no re-warp
            ncc = None
            if n_pixels >= 100:
                g1       = _to_gray_for_ncc(ov1)
                g2       = _to_gray_for_ncc(ov2)
                ov_flat1 = g1[ov_mask].astype(np.float32)
                ov_flat2 = g2[ov_mask].astype(np.float32)
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
                print(f"OK {coord1} <> {coord2}: "
                      f"PSNR={psnr:.2f} SSIM={s:.3f} "
                      f"RMSE={rmse:.2f} NCC={ncc if ncc is not None else 'N/A'}"
                      f" | {n_pixels}px")

        except Exception as e:
            print(f"[ERROR] {coord1} <> {coord2}: {e}")

    _save_csv(results, output_csv,
              ['tile1_coord', 'tile2_coord', 'overlap_pixels',
               'overlap_width', 'overlap_height', 'psnr', 'ssim', 'rmse', 'ncc'])
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
    skip_keys = {'tile1_coord', 'tile2_coord',
                 'overlap_pixels', 'overlap_width', 'overlap_height'}
    for key in results[0]:
        if key in skip_keys:
            continue
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            continue
        print(f"{key.upper():4s} -- mean={np.mean(vals):.3f} "
              f"std={np.std(vals):.3f} "
              f"min={np.min(vals):.3f} "
              f"max={np.max(vals):.3f}")


# ============================================================
# STEP 6 : Feather Blending  (unchanged)
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

    Parameters
    ----------
    feather_distance : float, optional
        Width in px of the soft blend transition at tile seams. Defaults to
        CONFIG['feather_distance']. <= 0 switches to a hard, winner-take-all
        seam instead: each overlap pixel goes entirely to whichever tile is
        more "interior" there (larger distance-to-its-own-edge) with no
        color mixing, for an analysis/training copy where blended/ghosted
        seam pixels are undesirable (matches CONFIG['analysis_feather_distance']).
        Naively passing feather_distance=0 into the old soft-blend formula
        divided by zero (silently producing inf/nan at seams) -- this is
        the fix for that, not just a threshold tweak.

    Returns
    -------
    final      : np.uint8  (H, W, 3)  -- blended panorama
    debug_info : dict  -- overlap_count, weight_map, valid_pixels, overlap_stats
    """
    feather_distance = CONFIG['feather_distance'] if feather_distance is None else feather_distance
    hard_seam = feather_distance <= 0

    offset_matrix = np.array([[1, 0, offset_x],
                               [0, 1, offset_y],
                               [0, 0, 1]], dtype=np.float32)
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
    use_gpu       = requested_gpu and torch.cuda.is_available()
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
                feather_zone = (min(feather_distance, max_dist * 0.3) if max_dist > 0 else 1)
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
                feather_zone = (min(feather_distance, max_dist * 0.3)
                                if max_dist > 0 else 1)
                feather      = np.minimum(dist / feather_zone, 1.0) * mask.astype(np.float32)

            overlap_here = (overlap_count * mask) > 1
            new_here     = (weight_map == 0) & (feather > 0)

            for c in range(3):
                canvas[:, :, c][new_here] = warped[:, :, c][new_here]
            weight_map[new_here] = feather[new_here]

            if overlap_here.any():
                cur_w = feather[overlap_here]
                ext_w = weight_map[overlap_here]
                if hard_seam:
                    # No color mixing: each overlap pixel goes entirely to
                    # whichever tile currently has the higher priority score.
                    win_mask = np.zeros_like(mask, dtype=bool)
                    win_mask[overlap_here] = cur_w > ext_w
                    for c in range(3):
                        canvas[:, :, c][win_mask] = warped[:, :, c][win_mask]
                else:
                    total = cur_w + ext_w
                    alpha = np.divide(cur_w, total,
                                      out=np.zeros_like(cur_w), where=total != 0)
                    for c in range(3):
                        canvas[:, :, c][overlap_here] = (
                            alpha * warped[:, :, c][overlap_here] +
                            (1 - alpha) * canvas[:, :, c][overlap_here]
                        )
                weight_map[overlap_here] = np.maximum(ext_w, cur_w)

            del warped  # free immediately

    if use_gpu:
        canvas     = canvas_t.cpu().numpy()
        weight_map = weight_map_t.cpu().numpy()
        del canvas_t, weight_map_t, overlap_count_t
        torch.cuda.empty_cache()

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
    gray       = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    valid_mask = (gray >= min_threshold).astype(np.int32)
    rows, cols = valid_mask.shape
    if debug:
        pct = valid_mask.sum() / (rows * cols) * 100
        print(f"Image: {cols}x{rows} | Valid pixels: {valid_mask.sum()} ({pct:.1f}%)")

    h_matrix         = _build_height_matrix_vectorized(valid_mask)
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
        'area_efficiency':    (x2 - x) * (y2 - y) /
                              (image.shape[0] * image.shape[1]) * 100,
        'content_efficiency': roi_info['area'] / ((x2 - x) * (y2 - y)) * 100,
    }
    return roi_img, stats


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_pipeline(cfg=None, folder_path=None):
    """
    Run the full tile-stitching pipeline once, under whichever backend
    cfg['backend'] selects ('pytorch' or 'onnx'). Output files (CSVs, result
    image) are tagged with the backend name so a pytorch run and an onnx run
    on the same folder don't clobber each other -- see compare_backends.py.

    Returns a dict of intermediate results/artifacts so callers (e.g. a
    comparison script) can inspect them without re-parsing CSVs or images.
    """
    cfg = cfg or CONFIG
    folder_path =  folder_path or "/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/10x10_euglena_red"
    backend_tag = cfg['backend']

    tracker = None
    if cfg.get('enable_benchmark', True):
        tracker = PerformanceTracker(f"SuperPoint + LightGlue Pipeline (SP_LG, backend={backend_tag})",
                                      backend=backend_tag)

    # Build the extractor/matcher pipeline (pytorch or onnx) once
    pipeline = build_pipeline(cfg)
    if tracker and pipeline['type'] == 'onnx':
        tracker.onnx_providers = pipeline['sp_sess'].get_providers()
    if tracker:
        tracker.record_step("0. Model Loading & Initialization")

    # Step 1 -- Load images & build grid
    image_data, grid_info = load_image(folder_path, cfg['resize_factor'])
    if cfg.get('debug', False):
        visualize_grid_preview(image_data, grid_info)
        plt.show()
    if tracker:
        tracker.record_step("1. Image Loading & Grid Preview")

    # Step 2 -- Compute overlap zones
    overlap_pairs = calculate_overlap(image_data, grid_info,
                                      cfg['overlap_percentage'])
    if cfg.get('debug', False):
        visualize_overlap_regions(image_data, overlap_pairs, grid_info)
        plt.show()
    if tracker:
        tracker.record_step("2. Overlap Region Setup")

    # Step 3 -- SuperPoint extraction on each ROI
    overlap_features = extract_overlap_features(
        image_data, overlap_pairs, pipeline)
    if cfg.get('debug', False):
        visualize_overlap_features(image_data, overlap_features, grid_info)
        plt.show()
    if tracker:
        tracker.record_step("3. SuperPoint Feature Extraction")

    # Step 4 -- LightGlue matching (superpoint mode)
    match_result = match_overlap_features(
        overlap_features, image_data, pipeline)
    if tracker:
        tracker.record_step("4. LightGlue Feature Matching")

    # Optional: visualize top/worst match pairs
    if cfg.get('debug', False):
        viz = visualize_lightglue_matches(image_data, match_result, top_n=3)
        if viz:
            plt.show()

    # Step 5 -- Homography via RANSAC
    homography_results = calculate_homographies_batch(
        image_data, match_result, cfg['reproj_thresh'])
    if tracker:
        tracker.record_step("5. Homography RANSAC")

    # Step 5.1 -- Reprojection error report
    reprojection_results = None
    if cfg.get('evaluate_metrics', False):
        reprojection_results = evaluate_homography_reprojection(
            match_result, homography_results,
            output_csv=f"homography_reprojection_sp_lg_{backend_tag}.csv")

    # Step 5.5 -- Overlap quality metrics (PSNR, SSIM, RMSE, NCC)
    transforms, reference, reachable_images = calculate_all_transforms(
        image_data, homography_results)
    canvas_w, canvas_h, offset_x, offset_y = calculate_optimal_canvas(
        image_data, transforms)
    overlap_metrics = None
    if cfg.get('evaluate_metrics', False):
        overlap_metrics = evaluate_overlap_metrics(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            homography_results=homography_results,   # adjacent-only -- O(N) not O(N²)
            output_csv=f"overlap_metrics_sp_lg_{backend_tag}.csv",
            visualize_sample=False,
        )
    if tracker and cfg.get('evaluate_metrics', False):
        tracker.record_step("5.5 Overlap Quality Metrics Evaluation")

    # Step 6 -- Feather blending
    # Hard-seam analysis copy first (if enabled), display copy second -- in
    # that order so this copy's own save message (deliberately NOT phrased
    # "Saved: ...", see below) never becomes the last "Saved: <path>.jpg"
    # match in the process output, which is what automate.py's tile-runner
    # parses to find the stitched panorama. Keeps automate.py picking up
    # the same (display) file it always has, unaffected by this addition.
    analysis_save_path = None
    if cfg.get('save_analysis_copy', False):
        analysis_blended, _ = blend_panorama(
            image_data, transforms, reachable_images,
            canvas_w, canvas_h, offset_x, offset_y,
            feather_distance=cfg.get('analysis_feather_distance', 0),
        )
        analysis_roi_info = find_roi(analysis_blended, min_threshold=1, debug=False)
        analysis_result, _ = extract_roi(analysis_blended, analysis_roi_info, padding=10)
        analysis_save_path = str(Path(folder_path) / f"result_sp_lg_{backend_tag}_analysis.jpg")
        cv2.imwrite(analysis_save_path, cv2.cvtColor(analysis_result, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Hard-seam analysis copy written: {analysis_save_path}")

    blended, debug_info = blend_panorama(
        image_data, transforms, reachable_images,
        canvas_w, canvas_h, offset_x, offset_y,
        feather_distance=cfg.get('display_feather_distance', cfg['feather_distance']),
    )
    if cfg.get('debug', False):
        visualize_blending_debug(debug_info, blended)
        plt.show()

    # ROI crop -- remove black borders
    roi_info     = find_roi(blended, min_threshold=1, debug=True)
    final_result, roi_stats = extract_roi(blended, roi_info, padding=10)

    # Save result
    save_path = str(Path(folder_path) / f"result_sp_lg_{backend_tag}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")
    if tracker:
        tracker.record_step("6. Feather Blending, ROI Crop & Saving")

    if tracker:
        tracker.print_summary()

    if cfg.get('debug', False):
        plt.figure(figsize=(20, 10))
        plt.imshow(final_result)
        plt.axis('off')
        plt.title(f'SuperPoint + LightGlue Tile Stitching Result ({backend_tag})', fontsize=16)
        plt.show()

    return {
        'backend':              backend_tag,
        'pipeline':             pipeline,
        'image_data':           image_data,
        'grid_info':            grid_info,
        'overlap_pairs':        overlap_pairs,
        'overlap_features':     overlap_features,
        'match_result':         match_result,
        'homography_results':   homography_results,
        'reprojection_results': reprojection_results,
        'transforms':           transforms,
        'reference':            reference,
        'reachable_images':     reachable_images,
        'overlap_metrics':      overlap_metrics,
        'final_result':         final_result,
        'roi_stats':            roi_stats,
        'save_path':            save_path,
        'analysis_save_path':   analysis_save_path,
        'tracker':              tracker,
    }


if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser(description="SuperPoint + LightGlue tile stitching")
    _parser.add_argument('--path', default="/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/10x10_euglena_red",
                         help="Folder of tile images to stitch")
    folder_path = _parser.parse_args().path
    run_pipeline(CONFIG, folder_path)
