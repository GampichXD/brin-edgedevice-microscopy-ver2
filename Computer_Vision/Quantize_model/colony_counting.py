"""
Colony counting via YOLO (Ultralytics)
=========================================
Runs a trained YOLO model on a single image and reports a colony count plus
an annotated visualization. Handles detect, obb, and segment task
checkpoints generically -- Ultralytics' Results object exposes the detections
under .boxes (detect), .obb (oriented boxes), or .masks (segment) depending
on how the model was trained; this picks whichever is populated.

Model format: anything Ultralytics' YOLO() can load -- .pt (PyTorch, embeds
task/class-name metadata), .onnx, or .engine (TensorRT; export one from your
.pt with `YOLO("model.pt").export(format="engine", half=True)`). Dispatch is
automatic based on the file extension (Ultralytics' AutoBackend), no code
path differs here between formats. .onnx/.engine files don't carry the same
embedded task metadata .pt does, so Ultralytics falls back to assuming
task="detect" for them with a warning -- pass --task explicitly (or
task=... to count_colonies()) if your model is actually obb/segment/etc.

Usage (standalone):
    python colony_counting.py --image path/to/panorama.jpg --model best_yolo26obj.pt
    python colony_counting.py --image path/to/panorama.jpg --model best_yolo26obj.engine

Usage (as a module, e.g. from automate.py):
    from colony_counting import count_colonies
    result = count_colonies(image_path, model_path, output_dir)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _extract_detections(result):
    """Pull count/confidences/classes/boxes/mask-polygons out of a Results
    object, regardless of whether the checkpoint is a detect, obb, or
    segment model. Segment-task results still populate .boxes (one box per
    .masks entry, same order), so masks_xy is returned alongside boxes_xyxy
    whenever masks are present -- callers decide which (or both) to draw."""
    obb = getattr(result, 'obb', None)
    boxes = getattr(result, 'boxes', None)
    masks = getattr(result, 'masks', None)

    if obb is not None and len(obb) > 0:
        det = obb
        boxes_xyxy = det.xyxyxyxy.cpu().numpy().tolist() if hasattr(det, 'xyxyxyxy') else []
    elif boxes is not None and len(boxes) > 0:
        det = boxes
        boxes_xyxy = det.xyxy.cpu().numpy().tolist() if hasattr(det, 'xyxy') else []
    else:
        det = None
        boxes_xyxy = []

    if det is None:
        return {'count': 0, 'confidences': [], 'classes': [], 'boxes': [], 'masks_xy': []}

    count = len(det)
    confidences = det.conf.cpu().numpy().tolist() if hasattr(det, 'conf') else []
    classes = det.cls.cpu().numpy().astype(int).tolist() if hasattr(det, 'cls') else []
    # masks.xy: list of Nx2 polygons already in original-image pixel coords
    # (Ultralytics scales them from the low-res mask-prototype grid) -- far
    # cheaper than masks.data's dense per-pixel tensors, and what makes
    # drawing masks on a large panorama tractable at all (see the CUDA OOM
    # note on the boxes/masks argument below).
    masks_xy = [m.tolist() for m in masks.xy] if masks is not None else []

    return {'count': count, 'confidences': confidences, 'classes': classes,
            'boxes': boxes_xyxy, 'masks_xy': masks_xy}


def count_colonies(image_path, model_path, output_dir=None, conf=0.25, imgsz=None, save_annotated=True, task=None,
                   boxes=True, masks=True):
    """
    Run YOLO inference on one image and count detected objects (colonies).
    model_path may be .pt, .onnx, or .engine -- format is auto-detected by
    Ultralytics from the extension. .onnx/.engine don't embed task metadata
    the way .pt does, so pass task ('detect'/'obb'/'segment'/...) explicitly
    for those if your model isn't a plain detector (Ultralytics otherwise
    guesses 'detect' and prints a warning).

    boxes/masks control what the annotated visualization draws (matching
    Ultralytics' own Results.plot(boxes=..., masks=...) naming). Detect/obb
    models only ever have boxes -- masks=True is a no-op for them. Segment
    models have both a box and a mask per instance; boxes=False gives a
    clean mask-only view instead of both overlapping (mask polygons trace
    the actual colony shape -- see colony_counting.py's module docstring
    history / the "generic segmentation" investigation for why this
    matters at the default imgsz on a large stitched panorama).

    Returns a dict: count, confidences, classes, boxes (xyxy or obb corners),
    masks_xy, annotated_path, json_path.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    output_dir = Path(output_dir) if output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path), task=task)
    predict_kwargs = {'conf': conf}
    if imgsz is not None:
        predict_kwargs['imgsz'] = imgsz
    results = model.predict(str(image_path), **predict_kwargs)
    result = results[0]

    det = _extract_detections(result)
    count, confidences, classes = det['count'], det['confidences'], det['classes']
    boxes_xyxy, masks_xy = det['boxes'], det['masks_xy']

    annotated_path = None
    if save_annotated:
        # Ultralytics' result.plot() sizes box/label text off the image
        # resolution, not the box size -- on a large stitched panorama with
        # hundreds of small, same-class detections packed close together,
        # the per-box "Colonies 0.83" labels end up many times larger than
        # the boxes themselves and bury the image in unreadable text. Since
        # every detection here is the same one class, per-box labels add
        # nothing anyway -- draw thin boxes/mask outlines only, plus a
        # single count in the corner. Mask outlines come from masks.xy
        # (polygon points, already in original-image coords) rather than
        # result.plot()'s dense per-pixel mask compositing -- the latter
        # allocates a (num_instances, H, W) tensor at full image resolution,
        # which reproducibly OOMs on this device for a few hundred instances
        # on a multi-thousand-pixel panorama (see the CUDA OOM investigation).
        annotated = cv2.imread(str(image_path))
        thickness = max(1, round(min(annotated.shape[:2]) / 1000))

        if masks and masks_xy:
            for poly in masks_xy:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=thickness)

        if boxes and boxes_xyxy:
            for x1, y1, x2, y2 in boxes_xyxy:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), thickness)

        label = f"Colonies: {count}"
        font_scale = max(1.0, min(annotated.shape[:2]) / 700)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness * 2)
        pad = thickness * 4
        cv2.rectangle(annotated, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (0, 0, 0), -1)
        cv2.putText(annotated, label, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 255, 0), thickness * 2, cv2.LINE_AA)

        annotated_path = output_dir / f"{image_path.stem}_colonies.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        annotated_path = str(annotated_path)

    detections_out = {
        'image': str(image_path),
        'model': str(model_path),
        'count': count,
        'confidences': confidences,
        'classes': classes,
        'boxes': boxes_xyxy,
        'masks_xy': masks_xy,
        'annotated_path': annotated_path,
    }
    json_path = output_dir / f"{image_path.stem}_colonies.json"
    with open(json_path, 'w') as f:
        json.dump(detections_out, f, indent=2)
    detections_out['json_path'] = str(json_path)

    print(f"[INFO] Colony count: {count}")
    if annotated_path:
        print(f"[INFO] Annotated image saved: {annotated_path}")
    print(f"[INFO] Detections JSON saved: {json_path}")

    return detections_out


def main():
    parser = argparse.ArgumentParser(description="Count colonies in an image using a YOLO model")
    parser.add_argument('--image', required=True, help="Path to input image")
    parser.add_argument('--model', required=True, help="Path to YOLO model: .pt, .onnx, or .engine")
    parser.add_argument('--output-dir', default=None,
                        help="Where to save annotated image + JSON (default: image's own folder)")
    parser.add_argument('--conf', type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument('--imgsz', type=int, default=None, help="Inference image size (default: model's own)")
    parser.add_argument('--task', default=None, choices=['detect', 'obb', 'segment', 'classify', 'pose'],
                        help="Model task -- only needed for .onnx/.engine models, which don't embed this "
                             "the way .pt does (default: Ultralytics guesses 'detect')")
    parser.add_argument('--boxes', action=argparse.BooleanOptionalAction, default=True,
                        help="Draw bounding boxes in the annotated image (--no-boxes to disable -- useful for a "
                             "clean mask-only view on segment models, where box+mask together looks cluttered)")
    parser.add_argument('--masks', action=argparse.BooleanOptionalAction, default=True,
                        help="Draw mask outlines in the annotated image, for segment models (--no-masks to disable). "
                             "No-op for detect/obb models, which have no masks.")
    args = parser.parse_args()
    count_colonies(args.image, args.model, args.output_dir, args.conf, args.imgsz, task=args.task,
                   boxes=args.boxes, masks=args.masks)


if __name__ == '__main__':
    main()
