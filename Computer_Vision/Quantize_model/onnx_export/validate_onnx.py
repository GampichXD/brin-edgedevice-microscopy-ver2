"""
Validate the exported ONNX graphs against their PyTorch source.
==================================================================
Two checks, on a real overlap-ROI-sized crop the exported models have
never seen during tracing (different tile pair, different crop shape
than export_to_onnx.py used):

  1. Numerical equivalence: ONNX Runtime (CPU EP) output vs the
     vendored PyTorch lightglue_onnx model that produced the export.
     This is the check that actually proves the export is correct.

  2. Sanity cross-check against SP_LG.py's production path (the
     official `lightglue` package). Keypoint/match COUNTS will differ
     somewhat because production runs with depth_confidence=0.95 /
     width_confidence=0.99 (adaptive early-exit + pruning enabled),
     while the ONNX graph always runs the full 9-layer network
     (do_early_stop / do_point_pruning hardcoded off -- see
     onnx_export/lightglue_onnx/__init__.py). That divergence is
     expected, not a bug; this check is informational, not pass/fail.

Usage:
    python validate_onnx.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from export_to_onnx import (  # noqa: E402
    LG_FILTER_THRESHOLD,
    SP_DETECTION_THRESHOLD,
    SP_MAX_KEYPOINTS,
    TILE_DIR,
    WEIGHTS_DIR,
    _load_roi_tensor,
    build_models,
    resize_long_side,
)
from lightglue_onnx.end2end import normalize_keypoints  # noqa: E402

ONNX_DIR = THIS_DIR.parent / "ONNX"
ATOL = 1e-4


def run_extractor_checks(extractor_pt, clahe):
    sess = ort.InferenceSession(str(ONNX_DIR / "superpoint.onnx"),
                                 providers=["CPUExecutionProvider"])

    min_overlap = 1.0
    max_desc_diff = 0.0
    outputs = {}
    for name, image in (
        # Tiny raw ROI crop (production-realistic pre-resize shape, and a
        # non-multiple-of-8 edge case) plus a production-scale (long
        # side = 1024px) shape -- both must match ORT bit-for-bit-ish.
        ("tile_r1_c1.jpg [raw crop]", _load_roi_tensor(TILE_DIR / "tile_r1_c1.jpg", clahe)[:, :, 5:75, 3:60]),
        ("tile_r1_c2.jpg [raw crop]", _load_roi_tensor(TILE_DIR / "tile_r1_c2.jpg", clahe)[:, :, 5:75, 3:60]),
        ("tile_r1_c1.jpg [resized]", resize_long_side(_load_roi_tensor(TILE_DIR / "tile_r1_c1.jpg", clahe))),
        ("tile_r1_c2.jpg [resized]", resize_long_side(_load_roi_tensor(TILE_DIR / "tile_r1_c2.jpg", clahe))),
    ):
        with torch.no_grad():
            kpts_pt, scores_pt, desc_pt = extractor_pt(image)

        kpts_ort, scores_ort, desc_ort = sess.run(
            None, {"image": image.numpy().astype(np.float32)}
        )

        assert kpts_pt.shape == kpts_ort.shape, \
            f"{name}: keypoint shape mismatch {kpts_pt.shape} vs {kpts_ort.shape}"

        # torch.topk(sorted=True) can pick a different (but equally valid)
        # keypoint at the max_num_keypoints cutoff than the ONNX Runtime
        # TopK kernel when candidate scores cluster tightly near the
        # boundary (typical once the image is upsampled to 1024px and the
        # cap is hit) -- ~1e-6 float noise between backends is enough to
        # flip which borderline candidate makes the cut. Compare as SETS
        # (by coordinate) and require >=99% overlap rather than an exact
        # index-aligned match, then diff descriptors only over the
        # intersection.
        kp_pt  = kpts_pt[0].numpy()
        kp_ort = kpts_ort[0]
        set_pt  = {tuple(p) for p in kp_pt.astype(int).tolist()}
        set_ort = {tuple(p) for p in kp_ort.astype(int).tolist()}
        shared  = set_pt & set_ort
        overlap = len(shared) / len(set_pt) if set_pt else 1.0

        idx_pt  = {tuple(p): i for i, p in enumerate(kp_pt.astype(int).tolist())}
        idx_ort = {tuple(p): i for i, p in enumerate(kp_ort.astype(int).tolist())}
        shared_l = list(shared)
        i_pt  = [idx_pt[p]  for p in shared_l]
        i_ort = [idx_ort[p] for p in shared_l]
        desc_diff = (np.abs(desc_pt[0].numpy()[i_pt] - desc_ort[0][i_ort]).max()
                     if shared_l else 0.0)
        min_overlap   = min(min_overlap, overlap)
        max_desc_diff = max(max_desc_diff, desc_diff)

        print(f"[extractor] {name}: shape={tuple(image.shape[-2:])} "
              f"n_kpts={kpts_pt.shape[1]} keypoint_set_overlap={overlap:.4%} "
              f"max|desc diff| (over shared kpts)={desc_diff:.2e}")

        # top_k_keypoints is now unconditional (always emits exactly
        # sp_max_keypoints, padding with score=-1 dummies) so TensorRT can
        # build a static-shape engine -- see onnx_export/lightglue_onnx/
        # superpoint.py. Drop padding before downstream matching, same as
        # SP_LG.py's _extract_region_onnx does for the real pipeline.
        real_ort = scores_ort[0] > 0             # numpy bool mask, indexes the ORT numpy outputs
        real_pt  = scores_pt[0] > 0               # torch bool mask, indexes the pytorch tensors
        outputs[name] = (image,
                          kpts_ort[:, real_ort], scores_ort[:, real_ort], desc_ort[:, real_ort],
                          kpts_pt[:, real_pt],   scores_pt[:, real_pt],   desc_pt[:, real_pt])

    ok = min_overlap >= 0.99 and max_desc_diff < ATOL
    print(f"[extractor] {'PASS' if ok else 'FAIL'} "
          f"(min keypoint_set_overlap={min_overlap:.4%}, max desc diff over shared kpts={max_desc_diff:.2e}, atol={ATOL})")
    return ok, outputs


def run_lightglue_checks(lightglue_pt, outputs):
    sess = ort.InferenceSession(str(ONNX_DIR / "superpoint_lightglue.onnx"),
                                 providers=["CPUExecutionProvider"])

    (name0, name1) = list(outputs)[:2]
    img0, k0_ort, s0_ort, d0_ort, k0_pt, s0_pt, d0_pt = outputs[name0]
    img1, k1_ort, s1_ort, d1_ort, k1_pt, s1_pt, d1_pt = outputs[name1]

    n0_pt = normalize_keypoints(k0_pt, img0.shape[2], img0.shape[3])
    n1_pt = normalize_keypoints(k1_pt, img1.shape[2], img1.shape[3])
    with torch.no_grad():
        matches_pt, mscores_pt = lightglue_pt(n0_pt, n1_pt, d0_pt, d1_pt)

    n0_ort = normalize_keypoints(torch.from_numpy(k0_ort), img0.shape[2], img0.shape[3]).numpy()
    n1_ort = normalize_keypoints(torch.from_numpy(k1_ort), img1.shape[2], img1.shape[3]).numpy()
    matches_ort, mscores_ort = sess.run(
        None,
        {
            "kpts0": n0_ort.astype(np.float32),
            "kpts1": n1_ort.astype(np.float32),
            "desc0": d0_ort.astype(np.float32),
            "desc1": d1_ort.astype(np.float32),
        },
    )

    same_matches = np.array_equal(matches_pt.numpy(), matches_ort)
    score_diff = (np.abs(mscores_pt.numpy() - mscores_ort).max()
                  if mscores_pt.numel() else 0.0)

    print(f"[lightglue] {name0}<->{name1}: "
          f"pt_matches={matches_pt.shape[0]} ort_matches={matches_ort.shape[0]} "
          f"identical_indices={same_matches} max|score diff|={score_diff:.2e}")

    ok = same_matches and score_diff < ATOL
    print(f"[lightglue] {'PASS' if ok else 'FAIL'} (atol={ATOL})")
    return ok


def run_production_sanity_check(extractor_pt, lightglue_pt):
    """Compare against SP_LG.py's actual PyTorch path (official `lightglue`
    package, early-exit ENABLED) on the SAME production-scale (resize=1024)
    input the official `.extract()` would produce internally, so this is an
    apples-to-apples comparison rather than an artifact of skipping the
    resize (see SP_PREPROCESS_RESIZE note in export_to_onnx.py)."""
    sys.path.insert(0, str(THIS_DIR.parent))
    sys.path.insert(0, str(THIS_DIR.parent.parent / "LightGlue"))
    from lightglue import LightGlue as OfficialLightGlue
    from lightglue import SuperPoint as OfficialSuperPoint
    from lightglue.utils import rbd

    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(WEIGHTS_DIR))
    try:
        official_extractor = OfficialSuperPoint(
            max_num_keypoints=SP_MAX_KEYPOINTS,
            detection_threshold=SP_DETECTION_THRESHOLD,
        ).eval()
        official_matcher = OfficialLightGlue(
            features="superpoint",
            depth_confidence=0.95,
            width_confidence=0.99,
            filter_threshold=LG_FILTER_THRESHOLD,
        ).eval()
    finally:
        torch.hub.set_dir(original_hub_dir)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img0_raw = _load_roi_tensor(TILE_DIR / "tile_r1_c1.jpg", clahe)[:, :, 5:75, 3:60]
    img1_raw = _load_roi_tensor(TILE_DIR / "tile_r1_c2.jpg", clahe)[:, :, 5:75, 3:60]

    with torch.no_grad():
        f0 = official_extractor.extract(img0_raw)   # resizes to 1024 internally
        f1 = official_extractor.extract(img1_raw)
        result = rbd(official_matcher({"image0": f0, "image1": f1}))
    n_official = int(result["matches"].shape[0])
    print(f"[production sanity] official lightglue (early-exit ON, resize=1024): "
          f"kpts0={f0['keypoints'].shape[1]} kpts1={f1['keypoints'].shape[1]} "
          f"matches={n_official}")

    img0 = resize_long_side(img0_raw)
    img1 = resize_long_side(img1_raw)
    with torch.no_grad():
        kpts0, scores0, desc0 = extractor_pt(img0)
        kpts1, scores1, desc1 = extractor_pt(img1)
        # Drop top_k_keypoints' padding (score=-1) -- see superpoint.py note.
        r0, r1 = scores0[0] > 0, scores1[0] > 0
        kpts0, desc0 = kpts0[:, r0], desc0[:, r0]
        kpts1, desc1 = kpts1[:, r1], desc1[:, r1]
        n0 = normalize_keypoints(kpts0, img0.shape[2], img0.shape[3])
        n1 = normalize_keypoints(kpts1, img1.shape[2], img1.shape[3])
        matches, _ = lightglue_pt(n0, n1, desc0, desc1)
    n_vendored = int(matches.shape[0])
    print(f"[production sanity] ONNX/vendored graph (full compute, resize=1024): "
          f"kpts0={kpts0.shape[1]} kpts1={kpts1.shape[1]} matches={n_vendored}  "
          f"<- counts may still differ from official since early-exit/pruning is "
          f"hardcoded off in the exported graph (see onnx_export/lightglue_onnx/__init__.py); "
          f"both are 'correct', they just run different amounts of compute.")


def main():
    extractor_pt, lightglue_pt = build_models()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    ext_ok, outputs = run_extractor_checks(extractor_pt, clahe)
    lg_ok = run_lightglue_checks(lightglue_pt, outputs)

    print()
    run_production_sanity_check(extractor_pt, lightglue_pt)

    print()
    if ext_ok and lg_ok:
        print("[RESULT] ONNX export is numerically equivalent to PyTorch. OK to proceed to TensorRT.")
    else:
        print("[RESULT] Mismatch detected -- do not proceed to TensorRT until this is fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
