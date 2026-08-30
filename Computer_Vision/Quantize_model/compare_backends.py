"""
Compare SP_LG.py's pytorch backend against the onnx backend
==============================================================
Runs the full tile-stitching pipeline twice on the same tile folder --
once per backend -- and reports where they agree/diverge: keypoint counts,
match counts/quality, homography inlier ratios, overlap image-quality
metrics (PSNR/SSIM/RMSE/NCC), stitched-panorama similarity, and wall-clock
timing per stage.

Expected, NOT a bug (see SP_LG.py CONFIG comments and onnx_export/):
  - onnx match counts are typically LOWER than pytorch's, because
    SP_LG.py's pytorch backend runs LightGlue with adaptive early-exit
    (depth_confidence=0.95) and point pruning (width_confidence=0.99)
    enabled, while the onnx-exported LightGlue always runs the full
    9-layer network (that adaptive control flow doesn't trace to ONNX).
  - keypoint counts/positions can differ by a handful at the
    max_num_keypoints cutoff, from float32 backend differences (PyTorch
    eager vs ONNX Runtime kernels) flipping a near-tied topk boundary.
  - onnx here runs on ONNX Runtime's CPUExecutionProvider (no GPU-enabled
    onnxruntime installed) while pytorch runs on CUDA -- timing is not
    apples-to-apples until a TensorRT engine or onnxruntime-gpu is in use.

Usage:
    python compare_backends.py [tile_folder]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import SP_LG

DEFAULT_FOLDER = "/home/brin-microscope/Documents/Tugas-Akhir/Hardware/Computer_Vision/Euglena_Tiles/5x5_ecoli"


def _summarize_run(result):
    ov = result['overlap_features']
    mr = result['match_result']
    hr = result['homography_results']

    n_kpts = [len(fd['keypoints1']) + len(fd['keypoints2']) for fd in ov.values()]
    match_counts = [v['match_count'] for v in mr.values()]
    qualities    = [v['quality']     for v in mr.values()]
    inlier_ratios = [v['inlier_ratio'] for v in hr.values()]

    metrics = result['overlap_metrics'] or []
    psnr = [m['psnr'] for m in metrics if m.get('psnr') is not None]
    ssim_vals = [m['ssim'] for m in metrics if m.get('ssim') is not None]
    rmse = [m['rmse'] for m in metrics if m.get('rmse') is not None]
    ncc  = [m['ncc']  for m in metrics if m.get('ncc')  is not None]

    step_times = result['tracker'].step_times if result['tracker'] else {}

    return {
        'backend':            result['backend'],
        'n_pairs_extracted':  len(ov),
        'total_keypoints':    int(np.sum(n_kpts)),
        'n_pairs_matched':    len(mr),
        'total_matches':      int(np.sum(match_counts)) if match_counts else 0,
        'avg_match_quality':  float(np.mean(qualities)) if qualities else 0.0,
        'n_homographies':     len(hr),
        'avg_inlier_ratio':   float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        'avg_psnr':           float(np.mean(psnr)) if psnr else None,
        'avg_ssim':           float(np.mean(ssim_vals)) if ssim_vals else None,
        'avg_rmse':           float(np.mean(rmse)) if rmse else None,
        'avg_ncc':            float(np.mean(ncc)) if ncc else None,
        'reachable_images':   len(result['reachable_images']),
        'save_path':          result['save_path'],
        'step_times':         step_times,
        'total_time':         sum(step_times.values()) if step_times else None,
        'final_result':       result['final_result'],
    }


def _print_table(rows, key_order, headers):
    widths = [max(len(str(r.get(k, ''))) for r in rows + [dict(zip(key_order, headers))]) for k in key_order]
    def _fmt(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    print(_fmt(headers))
    print(_fmt(['-' * w for w in widths]))
    for r in rows:
        print(_fmt([r.get(k, '') for k in key_order]))


def compare_panoramas(pytorch_summary, onnx_summary):
    """SSIM between the two backends' final stitched panoramas. Canvas size
    can differ slightly (different matches -> different homographies), so
    the onnx panorama is resized to the pytorch panorama's shape first --
    this is an approximation, not a pixel-registered comparison."""
    a = pytorch_summary['final_result']
    b = onnx_summary['final_result']
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return float(ssim(ga, gb, data_range=255))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER

    base_cfg = dict(SP_LG.CONFIG)
    base_cfg['debug'] = False  # never block on plt.show() for this comparison

    summaries = {}
    for backend in ('pytorch', 'onnx'):
        cfg = dict(base_cfg)
        cfg['backend'] = backend
        print(f"\n{'=' * 70}\nRunning backend: {backend}\n{'=' * 70}")
        result = SP_LG.run_pipeline(cfg, folder)
        summaries[backend] = _summarize_run(result)

    pt, ox = summaries['pytorch'], summaries['onnx']

    print(f"\n{'=' * 70}\nCOMPARISON: pytorch vs onnx\n{'=' * 70}")
    rows = [
        {'metric': 'pairs extracted',    'pytorch': pt['n_pairs_extracted'], 'onnx': ox['n_pairs_extracted']},
        {'metric': 'total keypoints',    'pytorch': pt['total_keypoints'],   'onnx': ox['total_keypoints']},
        {'metric': 'pairs matched',      'pytorch': pt['n_pairs_matched'],   'onnx': ox['n_pairs_matched']},
        {'metric': 'total matches',      'pytorch': pt['total_matches'],     'onnx': ox['total_matches']},
        {'metric': 'avg match quality',  'pytorch': f"{pt['avg_match_quality']:.3f}", 'onnx': f"{ox['avg_match_quality']:.3f}"},
        {'metric': 'homographies OK',    'pytorch': pt['n_homographies'],    'onnx': ox['n_homographies']},
        {'metric': 'avg inlier ratio',   'pytorch': f"{pt['avg_inlier_ratio']:.3f}", 'onnx': f"{ox['avg_inlier_ratio']:.3f}"},
        {'metric': 'reachable tiles',    'pytorch': pt['reachable_images'],  'onnx': ox['reachable_images']},
        {'metric': 'avg PSNR',           'pytorch': f"{pt['avg_psnr']:.2f}" if pt['avg_psnr'] else 'N/A',
                                          'onnx':    f"{ox['avg_psnr']:.2f}" if ox['avg_psnr'] else 'N/A'},
        {'metric': 'avg SSIM (overlap)', 'pytorch': f"{pt['avg_ssim']:.3f}" if pt['avg_ssim'] else 'N/A',
                                          'onnx':    f"{ox['avg_ssim']:.3f}" if ox['avg_ssim'] else 'N/A'},
        {'metric': 'avg RMSE',           'pytorch': f"{pt['avg_rmse']:.2f}" if pt['avg_rmse'] else 'N/A',
                                          'onnx':    f"{ox['avg_rmse']:.2f}" if ox['avg_rmse'] else 'N/A'},
        {'metric': 'avg NCC',            'pytorch': f"{pt['avg_ncc']:.3f}" if pt['avg_ncc'] else 'N/A',
                                          'onnx':    f"{ox['avg_ncc']:.3f}" if ox['avg_ncc'] else 'N/A'},
        {'metric': 'total pipeline time (s)', 'pytorch': f"{pt['total_time']:.1f}" if pt['total_time'] else 'N/A',
                                               'onnx':    f"{ox['total_time']:.1f}" if ox['total_time'] else 'N/A'},
    ]
    _print_table(rows, ['metric', 'pytorch', 'onnx'], ['metric', 'pytorch', 'onnx'])

    panorama_ssim = compare_panoramas(pt, ox)
    print(f"\nFinal stitched panorama SSIM (pytorch vs onnx, onnx resized to match): "
          f"{panorama_ssim:.4f}" if panorama_ssim is not None else "N/A")

    print(f"\nSaved panoramas:")
    print(f"  pytorch -> {pt['save_path']}")
    print(f"  onnx    -> {ox['save_path']}")
    print(f"\nPer-backend CSVs (Quantize_model/): "
          f"homography_reprojection_sp_lg_{{backend}}.csv, overlap_metrics_sp_lg_{{backend}}.csv")


if __name__ == '__main__':
    main()
