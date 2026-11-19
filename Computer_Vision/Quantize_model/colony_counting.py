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
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # Computer_Vision/
import image_scale  # noqa: E402

# ── Mask post-processing per batch (hemat memori GPU) ───────────────────────
# Ultralytics segment predict memanggil ops.process_mask(..., upsample=True):
# SEMUA mask N diperbesar ke resolusi input sekaligus dalam float32
# (N x 960 x 960 x 4 B, dua kali saat interpolate). Panorama dengan ratusan
# koloni -> >1 GB sekali alokasi -> OOM di Jetson (memori dibagi dgn CPU).
# Di sini diproses per MASK_CHUNK mask ke satu tensor uint8 yang dialokasikan
# sekali (tanpa salinan torch.cat); hasil identik, puncak memori ~chunk/N.
# LOW_MEM (dipakai otomatis setelah OOM berulang): mask dibiarkan di resolusi
# prototipe (1/4 input, 16x lebih hemat) -- hitungan tetap sama, kontur sedikit
# lebih kasar. Satuan luas tetap benar karena gain dihitung dari ukuran mask.
from ultralytics.utils import ops as _ops  # noqa: E402

MASK_CHUNK = 16
DEFAULT_MAX_DET = 1000        # default Ultralytics 300 memotong panorama padat
LOW_MEM = False
_orig_process_mask = _ops.process_mask


def _chunked_process_mask(protos, masks_in, bboxes, shape, upsample=False):
    if LOW_MEM:
        upsample = False
    n = masks_in.shape[0]
    if not upsample or n <= MASK_CHUNK:
        return _orig_process_mask(protos, masks_in, bboxes, shape, upsample=upsample)
    import torch
    out = torch.empty((n, *shape), dtype=torch.uint8, device=masks_in.device)
    for i in range(0, n, MASK_CHUNK):
        out[i:i + MASK_CHUNK] = _orig_process_mask(protos, masks_in[i:i + MASK_CHUNK],
                                                   bboxes[i:i + MASK_CHUNK], shape, upsample=True)
    return out


_ops.process_mask = _chunked_process_mask


def _is_oom(e):
    msg = str(e)
    return 'NVML_SUCCESS' in msg or 'out of memory' in msg.lower()


def _predict_safe(model, image_path, kwargs):
    """predict() dengan pemulihan OOM GPU (memori Jetson dibagi CPU/GPU dan
    sering penuh sesaat): coba ulang sekali setelah mengosongkan cache, lalu
    sekali lagi dalam mode LOW_MEM. Return (result, mask_mode)."""
    global LOW_MEM
    import time
    import torch
    for attempt, low in enumerate((False, False, True)):
        LOW_MEM = low
        try:
            return model.predict(image_path, **kwargs)[0], ('low-mem (prototype resolution)' if low else 'full')
        except RuntimeError as e:
            if not _is_oom(e) or attempt == 2:
                raise
            print(f"[WARN] GPU out of memory (attempt {attempt + 1}/3) -- "
                  f"{'retrying with low-memory masks' if attempt == 1 else 'retrying'}", flush=True)
            torch.cuda.empty_cache()
            time.sleep(2)


def _masks_xy_chunked(result):
    """Sama dengan result.masks.xy (poligon tiap mask, koordinat citra asli)
    tetapi menyalin mask ke CPU per MASK_CHUNK, bukan seluruh N x H x W sekaligus
    (di Jetson RAM CPU = memori GPU, jadi salinan penuh menggandakan puncak)."""
    masks = result.masks
    data = masks.data
    out = []
    for i in range(0, len(data), MASK_CHUNK):
        for seg in _ops.masks2segments(data[i:i + MASK_CHUNK]):
            out.append(_ops.scale_coords(data.shape[1:], seg, result.orig_shape, normalize=False))
    return out


def _mask_pixel_areas(result):
    """Luas tiap mask dalam piksel CITRA ASLI, dari jumlah piksel mask.

    masks.data sudah ada di memori (dibuat Ultralytics saat predict) pada
    resolusi input model (letterbox, mis. 1280x1280); satu piksel mask =
    (1/gain)² piksel asli. Menjumlah piksel lebih akurat daripada luas poligon
    masks.xy (poligon kontur melewati pusat piksel tepi -> luas kecil
    tertaksir terlalu rendah). Return None bila tidak ada mask."""
    masks = getattr(result, 'masks', None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return None
    mh, mw = masks.data.shape[1:]
    h0, w0 = result.orig_shape
    gain = min(mh / h0, mw / w0)
    # Dihitung di CPU per batch (16 mask x 1280² uint8 ~ 26 MB): tanpa alokasi
    # GPU sama sekali -- count_nonzero/sum di GPU butuh buffer sementara yang
    # memicu OOM saat memori Jetson penuh, padahal predict sudah selesai.
    data = masks.data
    counts = []
    for i in range(0, len(data), MASK_CHUNK):
        chunk = data[i:i + MASK_CHUNK].cpu().numpy()
        counts.extend(np.count_nonzero(chunk.reshape(len(chunk), -1), axis=1).tolist())
    return (np.asarray(counts, dtype=np.float64) / (gain * gain)).tolist()


def compute_colony_areas(result, masks_xy, confidences, scale):
    """Per-koloni: luas, diameter ekuivalen, keliling, sirkularitas, centroid.
    scale = dict dari image_scale.resolve_scale() atau None (-> satuan px)."""
    px_areas = _mask_pixel_areas(result)
    sx = scale['um_per_px_x'] if scale else 1.0
    sy = scale['um_per_px_y'] if scale else 1.0
    to_unit = (1e-6 if scale else 1.0)            # µm² -> mm² ; px² tetap
    len_unit = (1e-3 if scale else 1.0)           # µm  -> mm  ; px tetap

    colonies = []
    for i, poly in enumerate(masks_xy):
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        scaled = pts * [sx, sy]
        poly_area_px = float(cv2.contourArea(pts.astype(np.float32)))
        area_px = float(px_areas[i]) if px_areas is not None and i < len(px_areas) else poly_area_px
        if area_px <= 0:
            continue
        area = area_px * sx * sy * to_unit
        perim = float(cv2.arcLength(scaled.astype(np.float32), True)) * len_unit
        m = cv2.moments(pts.astype(np.float32))
        cx, cy = (m['m10'] / m['m00'], m['m01'] / m['m00']) if m['m00'] else tuple(pts.mean(axis=0))
        colonies.append({
            'id': len(colonies) + 1,
            'area': area,
            'area_px': area_px,
            'eq_diameter': 2.0 * math.sqrt(area / math.pi),
            'perimeter': perim,
            'circularity': min(1.0, 4.0 * math.pi * area / perim ** 2) if perim > 0 else 0.0,
            'centroid_x_px': float(cx),
            'centroid_y_px': float(cy),
            'confidence': float(confidences[i]) if i < len(confidences) else None,
        })
    return colonies


def summarize_areas(colonies, scale, bins='auto'):
    unit = 'mm²' if scale else 'px²'
    a = np.array([c['area'] for c in colonies], dtype=np.float64)
    stats = {'unit': unit, 'length_unit': 'mm' if scale else 'px', 'n': int(a.size),
             'scale': scale, 'calibrated': bool(scale)}
    if a.size == 0:
        stats.update(total=0.0, mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0, p10=0.0, p90=0.0,
                     mean_eq_diameter=0.0, hist_counts=[], hist_edges=[])
        return stats
    counts, edges = np.histogram(a, bins=bins if a.size > 1 else 1)
    if len(counts) > 40:                       # 'auto' bisa memberi ratusan bin pada sebaran ekstrem
        counts, edges = np.histogram(a, bins=40)
    stats.update(
        total=float(a.sum()), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
        min=float(a.min()), max=float(a.max()),
        p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
        mean_eq_diameter=float(np.mean([c['eq_diameter'] for c in colonies])),
        hist_counts=counts.astype(int).tolist(), hist_edges=edges.tolist(),
    )
    return stats


def _fmt(v):
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def save_area_histogram(stats, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    unit = stats['unit']
    if stats['n']:
        edges = np.array(stats['hist_edges'])
        ax.bar(edges[:-1], stats['hist_counts'], width=np.diff(edges), align='edge',
               color='#3b82f6', edgecolor='white', linewidth=0.6)
        ax.axvline(stats['mean'], color='#ef4444', lw=1.5, ls='--', label=f"mean {_fmt(stats['mean'])} {unit}")
        ax.axvline(stats['median'], color='#10b981', lw=1.5, ls=':', label=f"median {_fmt(stats['median'])} {unit}")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, 'No colonies detected', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel(f'Colony area ({unit})')
    ax.set_ylabel('Number of colonies')
    ax.set_title(f"{title}  —  n = {stats['n']}")
    ax.spines[['top', 'right']].set_visible(False)
    sc = stats.get('scale')
    note = (f"scale {sc['um_per_px_x']:.3f} x {sc['um_per_px_y']:.3f} µm/px ({sc['source']})" if sc
            else 'uncalibrated: no µm/px scale for this image, areas in pixels')
    fig.text(0.01, 0.01, note, fontsize=7, color='#6b7280')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def save_area_csv(colonies, stats, out_path):
    u, lu = stats['unit'], stats['length_unit']
    cols = [('id', 'id'), ('area', f'area_{u}'), ('area_px', 'area_px'), ('eq_diameter', f'eq_diameter_{lu}'),
            ('perimeter', f'perimeter_{lu}'), ('circularity', 'circularity'),
            ('centroid_x_px', 'centroid_x_px'), ('centroid_y_px', 'centroid_y_px'), ('confidence', 'confidence')]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([h for _, h in cols])
        for c in colonies:
            w.writerow([round(c[k], 6) if isinstance(c[k], float) else c[k] for k, _ in cols])


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
    masks_xy = [m.tolist() for m in _masks_xy_chunked(result)] if masks is not None and len(masks) else []

    return {'count': count, 'confidences': confidences, 'classes': classes,
            'boxes': boxes_xyxy, 'masks_xy': masks_xy}


def count_colonies(image_path, model_path, output_dir=None, conf=0.25, imgsz=None, save_annotated=True, task=None,
                   boxes=True, masks=True, um_per_px=None, save_area=True, max_det=DEFAULT_MAX_DET):
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

    Colony area (segment models): per-colony area from the mask pixel count,
    converted with the image's µm/px scale (um_per_px=(x, y) override, else the
    scale embedded in the JPEG by stitching, else the camera-tile FOV, else
    pixels -- see Computer_Vision/image_scale.py). With save_area=True this also
    exports <stem>_area_hist.png (histogram) and <stem>_areas.csv (per colony).

    Memory (Jetson, CPU and GPU share one RAM pool): masks are post-processed and
    copied to the CPU in batches, and a GPU out-of-memory error is retried, last
    in low-memory mode (prototype-resolution masks). max_det defaults to 1000
    (Ultralytics' 300 silently caps dense panoramas); 'capped' in the JSON says
    whether the limit was reached.

    Returns a dict: count, confidences, classes, boxes (xyxy or obb corners),
    masks_xy, colonies (per-colony area rows), area_stats, max_det, capped,
    mask_mode, annotated_path, histogram_path, areas_csv_path, json_path.
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
    predict_kwargs = {'conf': conf, 'max_det': max_det, 'verbose': True}
    if imgsz is not None:
        predict_kwargs['imgsz'] = imgsz
    result, mask_mode = _predict_safe(model, str(image_path), predict_kwargs)

    det = _extract_detections(result)
    count, confidences, classes = det['count'], det['confidences'], det['classes']
    boxes_xyxy, masks_xy = det['boxes'], det['masks_xy']
    capped = count >= max_det
    if capped:
        print(f"[WARN] {count} detections = max_det limit: the image may hold more colonies. "
              f"Raise --max-det to count them all.", flush=True)

    h0, w0 = result.orig_shape
    scale = image_scale.resolve_scale(str(image_path), w0, h0, override=um_per_px)
    colonies = compute_colony_areas(result, masks_xy, confidences, scale)
    area_stats = summarize_areas(colonies, scale)
    histogram_path = areas_csv_path = None
    if save_area and masks_xy:
        histogram_path = output_dir / f"{image_path.stem}_area_hist.png"
        save_area_histogram(area_stats, histogram_path, image_path.name)
        histogram_path = str(histogram_path)
        areas_csv_path = output_dir / f"{image_path.stem}_areas.csv"
        save_area_csv(colonies, area_stats, areas_csv_path)
        areas_csv_path = str(areas_csv_path)

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
        'colonies': colonies,
        'area_stats': area_stats,
        'max_det': max_det,
        'capped': capped,
        'mask_mode': mask_mode,
        'annotated_path': annotated_path,
        'histogram_path': histogram_path,
        'areas_csv_path': areas_csv_path,
    }
    json_path = output_dir / f"{image_path.stem}_colonies.json"
    with open(json_path, 'w') as f:
        json.dump(detections_out, f, indent=2)
    detections_out['json_path'] = str(json_path)

    print(f"[INFO] Colony count: {count}" + (" (capped at max_det)" if capped else "")
          + ("" if mask_mode == 'full' else f" [{mask_mode}]"))
    if area_stats['n']:
        u = area_stats['unit']
        print(f"[INFO] Colony area: total {area_stats['total']:.4g} {u}, mean {area_stats['mean']:.4g} {u}, "
              f"median {area_stats['median']:.4g} {u} "
              f"({'scale: ' + scale['source'] if scale else 'uncalibrated'})")
    if histogram_path:
        print(f"[INFO] Area histogram saved: {histogram_path}")
        print(f"[INFO] Per-colony areas saved: {areas_csv_path}")
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
    parser.add_argument('--um-per-px', type=float, nargs='+', default=None, metavar=('X', 'Y'),
                        help="Override image scale in µm/px (one value = square pixels). Default: scale "
                             "embedded by stitching, else camera tile FOV, else uncalibrated (px²)")
    parser.add_argument('--area', action=argparse.BooleanOptionalAction, default=True,
                        help="Export area histogram PNG + per-colony areas CSV (--no-area to skip)")
    parser.add_argument('--max-det', type=int, default=DEFAULT_MAX_DET,
                        help=f"Maximum colonies per image (default {DEFAULT_MAX_DET}; Ultralytics' 300 caps "
                             "dense panoramas)")
    args = parser.parse_args()
    count_colonies(args.image, args.model, args.output_dir, args.conf, args.imgsz, task=args.task,
                   boxes=args.boxes, masks=args.masks, um_per_px=args.um_per_px, save_area=args.area,
                   max_det=args.max_det)


if __name__ == '__main__':
    main()
