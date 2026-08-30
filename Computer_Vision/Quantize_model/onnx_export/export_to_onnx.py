"""
Export SuperPoint + LightGlue (superpoint variant) to ONNX
============================================================
Uses the vendored, ONNX-traceable reimplementation in ./lightglue_onnx
(fabio-sim/LightGlue-ONNX v1.0.0, Apache-2.0) instead of the official
`lightglue` package, because the official LightGlue's adaptive
depth/width pruning (early-exit) is data-dependent control flow that
does not trace to a static ONNX graph. The vendored version hardcodes
that pruning off and always runs the full 9-layer transformer, but
loads the exact same pretrained weights (cvg/LightGlue release v0.1_arxiv)
as SP_LG.py, so keypoints/descriptors/matches are architecturally
equivalent modulo the disabled early-exit shortcut.

Outputs two dynamic-shape ONNX graphs (mirrors SP_LG.py's per-ROI
extract-then-match structure, not a fused end-to-end graph):
    ONNX/superpoint.onnx            image (1,1,H,W) -> keypoints, scores, descriptors
    ONNX/superpoint_lightglue.onnx  kpts0,kpts1,desc0,desc1 -> matches0, mscores0

H, W and the number of keypoints are all dynamic axes, since SP_LG.py's
overlap ROIs vary in size per tile pair.

Usage:
    python export_to_onnx.py
"""

import sys
from pathlib import Path

import cv2
import kornia
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from lightglue_onnx import LightGlue, SuperPoint          # noqa: E402
from lightglue_onnx.end2end import normalize_keypoints    # noqa: E402
import SP_LG                                              # noqa: E402

WEIGHTS_DIR = THIS_DIR.parent / "Weights"
OUTPUT_DIR  = THIS_DIR.parent / "ONNX"
TILE_DIR    = Path("/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/5x5_ecoli")

# Pulled from SP_LG.py's CONFIG (not duplicated constants) so the exported
# graph can never silently drift out of sync with it. IMPORTANT: unlike the
# pytorch backend, max_num_keypoints/detection_threshold/filter_threshold
# are baked into the ONNX graph as constants at export time (torch.onnx.export
# freezes the Python ints used by top_k_keypoints etc.) -- they are NOT
# runtime-configurable via CONFIG for the onnx backend. Changing CONFIG and
# re-running SP_LG.py does nothing to an already-exported .onnx file; you
# must re-run this script (`python export_to_onnx.py`) after changing any
# of these values in SP_LG.CONFIG.
SP_MAX_KEYPOINTS       = SP_LG.CONFIG['sp_max_keypoints']
SP_DETECTION_THRESHOLD = SP_LG.CONFIG['sp_detection_threshold']
LG_FILTER_THRESHOLD    = SP_LG.CONFIG['lg_filter_threshold']
OPSET                  = 17

# Official `lightglue.SuperPoint.preprocess_conf = {"resize": 1024}` -- the
# production path (SP_LG.py's `extractor.extract(tensor)`) upsamples every
# ROI crop so its long side is 1024px *before* running the network. That
# resize happens outside the traced graph, so it is NOT baked into
# superpoint.onnx -- whatever calls the exported model (ONNX Runtime /
# TensorRT deployment code) MUST reproduce it, or keypoint yield will be
# far lower than production (a ~90px raw ROI crop yields ~10 keypoints
# unresized vs. the full max_num_keypoints cap once resized to 1024).
SP_PREPROCESS_RESIZE = SP_LG.CONFIG.get('onnx_sp_resize', 1024)


def _load_roi_tensor(path, clahe):
    """Mirrors SP_LG.py's _extract_region preprocessing: grayscale -> CLAHE -> [0,1] tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = clahe.apply(img)
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def resize_long_side(image, size=SP_PREPROCESS_RESIZE):
    """Same resize `lightglue.utils.ImagePreprocessor` applies inside `.extract()`."""
    return kornia.geometry.transform.resize(
        image, size, side="long", antialias=True, align_corners=None
    )


def build_models():
    """Instantiate SuperPoint + LightGlue from local weights (no internet)."""
    required = {
        'superpoint_v1.pth':                    'SuperPoint detector',
        'superpoint_lightglue_v0-1_arxiv.pth':  'LightGlue (superpoint) matcher',
    }
    for fname, label in required.items():
        fpath = WEIGHTS_DIR / 'checkpoints' / fname
        if not fpath.exists():
            raise FileNotFoundError(f"[ERROR] {label} weights not found: {fpath}")

    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(WEIGHTS_DIR))
    try:
        extractor = SuperPoint(
            max_num_keypoints=SP_MAX_KEYPOINTS,
            detection_threshold=SP_DETECTION_THRESHOLD,
        ).eval()
        lightglue = LightGlue(
            "superpoint", filter_threshold=LG_FILTER_THRESHOLD
        ).eval()
    finally:
        torch.hub.set_dir(original_hub_dir)
    return extractor, lightglue


def main():
    print(f"[INFO] Exporting with sp_max_keypoints={SP_MAX_KEYPOINTS} "
          f"sp_detection_threshold={SP_DETECTION_THRESHOLD} "
          f"lg_filter_threshold={LG_FILTER_THRESHOLD} "
          f"onnx_sp_resize={SP_PREPROCESS_RESIZE} (from SP_LG.CONFIG)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor, lightglue = build_models()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Trace at production scale (resized so the long side is 1024px, same
    # as `extractor.extract()` does at inference time -- see
    # SP_PREPROCESS_RESIZE above) using two different tiles/aspect ratios,
    # so the trace doesn't bake in a single fixed shape and exercises the
    # dynamic axes declared below.
    image0 = resize_long_side(_load_roi_tensor(TILE_DIR / "tile_r0_c0.jpg", clahe))
    image1 = resize_long_side(_load_roi_tensor(TILE_DIR / "tile_r0_c1.jpg", clahe))[:, :, :900, :300]

    # ---- SuperPoint extractor -------------------------------------
    extractor_path = OUTPUT_DIR / "superpoint.onnx"
    torch.onnx.export(
        extractor,
        image0,
        str(extractor_path),
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        opset_version=OPSET,
        dynamic_axes={
            "image":       {2: "height", 3: "width"},
            "keypoints":   {1: "num_keypoints"},
            "scores":      {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
        },
    )
    print(f"[OK] SuperPoint    -> {extractor_path}")

    # ---- LightGlue matcher ------------------------------------------
    with torch.no_grad():
        kpts0, scores0, desc0 = extractor(image0)
        kpts1, scores1, desc1 = extractor(image1)
    n_kpts0 = normalize_keypoints(kpts0, image0.shape[2], image0.shape[3])
    n_kpts1 = normalize_keypoints(kpts1, image1.shape[2], image1.shape[3])

    lightglue_path = OUTPUT_DIR / "superpoint_lightglue.onnx"
    torch.onnx.export(
        lightglue,
        (n_kpts0, n_kpts1, desc0, desc1),
        str(lightglue_path),
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "mscores0"],
        opset_version=OPSET,
        dynamic_axes={
            "kpts0":     {1: "num_keypoints0"},
            "kpts1":     {1: "num_keypoints1"},
            "desc0":     {1: "num_keypoints0"},
            "desc1":     {1: "num_keypoints1"},
            "matches0":  {0: "num_matches0"},
            "mscores0":  {0: "num_matches0"},
        },
    )
    print(f"[OK] LightGlue     -> {lightglue_path}")


if __name__ == "__main__":
    main()
