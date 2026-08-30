"""
End-to-end automation: tile stitching -> colony counting
=============================================================
Runs one of the tile-stitching pipelines (SIFT_BFM / SIFT_LG / SP_LG) as a
fresh subprocess against the given tile folder, locates the stitched
panorama it saves (by parsing the "Saved: <path>" line each script already
prints -- see their respective __main__ blocks), then feeds that image into
colony_counting.py's YOLO-based counter.

A fresh subprocess per stitching run (rather than importing and calling the
pipeline in-process) mirrors benchmark_trials.py's approach: it gives each
run a clean CUDA/TensorRT context and doesn't require SIFT_BFM.py/SIFT_LG.py
to be refactored into an importable run_pipeline()-style function the way
SP_LG.py already is.

--counting accepts any model format Ultralytics' YOLO() can load: .pt, .onnx,
or .engine (TensorRT -- export one via
`YOLO("best_yolo26obj.pt").export(format="engine", half=True)`; dispatch is
automatic based on the file extension, same colony_counting.py code path
either way). .onnx/.engine don't embed task metadata the way .pt does, so
pass --task explicitly for those if the model isn't a plain detector.

Usage:
    python automate.py --tile sp_lg --counting best_yolo26obj.pt --path /path/to/tiles
    python automate.py --tile sift_lg --counting best_yolo26obj.pt --path /path/to/tiles --conf 0.3
    python automate.py --tile sp_lg --counting best_yolo26obj.engine --path /path/to/tiles --no-masks
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from colony_counting import count_colonies

THIS_DIR = Path(__file__).resolve().parent

TILE_SCRIPTS = {
    'sift_bfm': THIS_DIR / 'SIFT_BFM.py',
    'sift_lg':  THIS_DIR / 'SIFT_LG.py',
    'sp_lg':    THIS_DIR / 'SP_LG.py',
}

# Matches the final panorama's "Saved: <path>.jpg" line. All three scripts
# also print "[INFO] Saved: <path>.csv" for their metrics CSVs earlier in the
# run -- the .jpg-only pattern skips those without needing to special-case them.
_RE_SAVED_IMAGE = re.compile(r'Saved:\s*(\S+\.jpg)')


def run_tile_stitching(tile_choice, input_path, timeout=600):
    script_path = TILE_SCRIPTS[tile_choice]
    print(f"[STEP 1/2] Tile stitching: {script_path.name}  --path {input_path}")
    print('-' * 70)
    result = subprocess.run(
        [sys.executable, str(script_path), '--path', str(input_path)],
        capture_output=True, text=True, timeout=timeout, cwd=str(THIS_DIR),
    )
    output = (result.stdout or '') + '\n' + (result.stderr or '')
    print(output)

    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} failed (exit code {result.returncode}); see output above.")

    matches = _RE_SAVED_IMAGE.findall(output)
    if not matches:
        raise RuntimeError(f"{script_path.name} finished but no 'Saved: <path>.jpg' line was found in its "
                           f"output -- could not locate the stitched panorama.")
    stitched_path = matches[-1]
    if not Path(stitched_path).exists():
        raise RuntimeError(f"{script_path.name} reported saving to {stitched_path}, but that file doesn't exist.")

    print(f"[STEP 1/2] Done -- stitched panorama: {stitched_path}")
    return stitched_path


def main():
    parser = argparse.ArgumentParser(description="Tile stitching -> colony counting pipeline")
    parser.add_argument('--tile', required=True, choices=list(TILE_SCRIPTS),
                        help="Which tile-stitching pipeline to run")
    parser.add_argument('--counting', required=True,
                        help="Path to YOLO model for colony counting: .pt, .onnx, or .engine")
    parser.add_argument('--path', required=True, help="Folder of tile images to stitch")
    parser.add_argument('--conf', type=float, default=0.25, help="Colony detection confidence threshold")
    parser.add_argument('--imgsz', type=int, default=None, help="YOLO inference image size (default: model's own)")
    parser.add_argument('--task', default=None, choices=['detect', 'obb', 'segment', 'classify', 'pose'],
                        help="Model task -- only needed for .onnx/.engine models (default: Ultralytics guesses 'detect')")
    parser.add_argument('--boxes', action=argparse.BooleanOptionalAction, default=True,
                        help="Draw bounding boxes in the annotated image (--no-boxes to disable -- useful for a "
                             "clean mask-only view on segment models)")
    parser.add_argument('--masks', action=argparse.BooleanOptionalAction, default=True,
                        help="Draw mask outlines in the annotated image, for segment models (--no-masks to disable)")
    parser.add_argument('--output-dir', default=None,
                        help="Where to save colony-counting outputs (default: same folder as the stitched image)")
    parser.add_argument('--tile-timeout', type=int, default=600, help="Timeout in seconds for the tile-stitching step")
    args = parser.parse_args()

    # Resolve user-supplied paths against automate.py's own cwd (wherever it
    # was actually invoked from) *before* anything changes directory.
    # run_tile_stitching() launches the stitching script with cwd=THIS_DIR
    # (Quantize_model/) so `python SIFT_BFM.py` works regardless of where
    # automate.py itself lives -- a relative --path would otherwise get
    # silently reinterpreted against Quantize_model/ instead of the
    # directory the user actually ran this command from.
    tile_path = str(Path(args.path).resolve())
    output_dir = str(Path(args.output_dir).resolve()) if args.output_dir else None

    stitched_path = run_tile_stitching(args.tile, tile_path, timeout=args.tile_timeout)

    print(f"\n[STEP 2/2] Colony counting: {args.counting}  --image {stitched_path}")
    print('-' * 70)
    result = count_colonies(
        image_path=stitched_path,
        model_path=args.counting,
        output_dir=output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        task=args.task,
        boxes=args.boxes,
        masks=args.masks,
    )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  Tile-stitching pipeline : {args.tile}")
    print(f"  Stitched panorama       : {stitched_path}")
    print(f"  Colony count            : {result['count']}")
    print(f"  Annotated image         : {result['annotated_path']}")
    print(f"  Detections JSON         : {result['json_path']}")


if __name__ == '__main__':
    main()
