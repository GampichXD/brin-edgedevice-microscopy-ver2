"""
Build TensorRT engines from the exported ONNX models
========================================================
Wraps `trtexec` to build .engine files for superpoint.onnx and
superpoint_lightglue.onnx (produced by export_to_onnx.py) at each precision
in SP_LG.CONFIG['trt_precisions']. Same pattern as export_to_onnx.py: all
knobs (keypoint cap, resize, shape profile) come from SP_LG.CONFIG, not
duplicated constants, so there is one place to change them.

Shape profiles
--------------
ONNX Runtime resolves dynamic shapes at every call; TensorRT instead compiles
a fixed kernel plan ahead of time and needs an explicit min/opt/max range per
dynamic input up front:
  - superpoint 'image' (1,1,H,W): SP_LG.py's onnx backend always resizes the
    ROI crop so its long side == CONFIG['onnx_sp_resize'] before inference,
    so H and W are each bounded by that value on the long-side end; the
    short side varies with the ROI's aspect ratio. Range comes from
    CONFIG['trt_sp_shape_min'/'trt_sp_shape_opt'/'trt_sp_shape_max'].
  - superpoint_lightglue 'kptsN'/'descN' (1,N,*): N is bounded above by
    CONFIG['sp_max_keypoints'] (the same hard cap baked into superpoint.onnx
    at export time), floor from CONFIG['trt_lg_kpts_min'], opt at the
    midpoint.

If you change sp_max_keypoints, onnx_sp_resize, or any trt_sp_shape_*/
trt_lg_kpts_min value, the .onnx files (if sp_max_keypoints/onnx_sp_resize
changed -- re-run export_to_onnx.py first) and these .engine files are both
stale until rebuilt. Nothing here is runtime-configurable the way the
pytorch backend is.

Precisions
----------
  fp32 -- baseline, no precision flag (trtexec's literal default, TF32
          opportunistically enabled same as plain "fp32" everywhere else
          in this project)
  fp16 -- --fp16 (adds FP16 kernels alongside FP32; TensorRT chooses per layer)
  int8 -- --int8, no --calib cache (implicit/weight-focused quantization,
          no calibration dataset wired up here).

          MEASURED (build-time signal): real for superpoint (fp32 6230700 B
          -> int8 2210660 B, ~2.8x) -- its conv backbone is exactly what
          implicit/uncalibrated INT8 quantizes. A NO-OP for
          superpoint_lightglue (fp32 58050708 B -> int8 58054028 B, i.e. no
          smaller) -- same story as int4 below: without a calibration pass,
          TensorRT doesn't find INT8 tactics for LightGlue's
          attention/transformer layers.

          MEASURED (RUNTIME correctness -- this is the important one): the
          superpoint int8 engine builds, is genuinely smaller, and PASSES
          trtexec's own post-build inference self-check (random data) --
          but on a real ROI image it outputs a score of exactly -1.0 for
          all sp_max_keypoints entries. That -1.0 is top_k_keypoints' own
          padding sentinel (see onnx_export/lightglue_onnx/superpoint.py) --
          i.e. the engine finds ZERO real keypoints on real data, silently.
          Uncalibrated INT8's automatic per-tensor dynamic-range estimation
          is apparently too inaccurate for the detector head's very tight
          0.005 detection_threshold. A build-time size check or trtexec's
          own random-data self-check CANNOT catch this -- both look
          "successful". Do not use superpoint int8 as-is; SP_LG.CONFIG
          defaults trt_runtime_precision_sp to 'fp16' precisely because of
          this. Fixing it for real would need a proper --calib pass built
          from representative resized (1024px) ROI crops, which this script
          does not do.

          If you want real LightGlue int8 (a different, lesser problem --
          it's a no-op, not broken), that likely also needs a --calib cache
          built from representative ROI crops (not done here).

  int4 was tried first and dropped -- kept here as a note so it isn't
  re-attempted blind. MEASURED ON THIS GRAPH/HARDWARE (TensorRT 10.3.0,
  Orin): the bare --int4 flag builds successfully and trtexec reports
  "Precision: FP32+INT4", but --dumpLayerInfo on the resulting engines
  showed ZERO layers actually assigned INT4 -- every layer stayed FP32, and
  the .engine file was byte-identical in size to the plain fp32 build.
  --int4 alone is apparently a no-op on a plain (non-Q/DQ) ONNX graph like
  ours; --stronglyTyped (TensorRT's other path to INT4) is flatly
  incompatible with implicit --int4 ("setting int4 mode is not allowed if
  graph is strongly typed"). Genuine INT4 would need explicit Q/DQ
  block-quantization nodes inserted into the ONNX graph beforehand (e.g. via
  NVIDIA's TensorRT Model Optimizer / nvidia-modelopt, not installed) -- a
  separate, heavier step this script does not perform. build_engine() below
  flags this automatically (for any precision) when the resulting engine's
  size matches its fp32 sibling, as a tripwire against silent no-ops.

  IMPORTANT LIMIT OF THAT TRIPWIRE: it only catches "this precision did
  nothing" (int4, and lightglue's int8). It CANNOT catch "this precision
  did something but broke correctness" -- that's exactly the superpoint
  int8 case above, which is smaller, real, passes trtexec's own
  self-check, and still silently returns zero real keypoints on actual
  images. A size/build-time check is necessary but not sufficient;
  correctness needs checking against real data (see the CONFIG note next
  to trt_runtime_precision_sp in SP_LG.py).

Usage:
    python build_tensorrt_engines.py
    python build_tensorrt_engines.py --precisions fp16,int8
    python build_tensorrt_engines.py --models superpoint

Each build is a one-time cost (not per-inference) but can take several
minutes on Jetson -- LightGlue's transformer especially, given the wide
keypoint-count profile. Expect the full fp32+fp16+int8 x 2-model sweep to
take tens of minutes; this is expected, not a hang.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
import SP_LG  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

ONNX_DIR = Path(SP_LG.CONFIG['onnx_dir'])
TRT_DIR  = Path(SP_LG.CONFIG['trt_dir'])

PRECISION_FLAGS = {
    'fp32': [],
    'fp16': ['--fp16'],
    'int8': ['--int8'],   # no --calib cache wired up -- see the int8 note above
    'int4': ['--int4'],   # kept for completeness; confirmed a no-op here, see note above
}


def _shapes(input_name, dims):
    return f"{input_name}:" + "x".join(str(d) for d in dims)


def superpoint_profile():
    h_min, w_min = SP_LG.CONFIG['trt_sp_shape_min']
    h_opt, w_opt = SP_LG.CONFIG['trt_sp_shape_opt']
    h_max, w_max = SP_LG.CONFIG['trt_sp_shape_max']
    return {
        'min': _shapes('image', (1, 1, h_min, w_min)),
        'opt': _shapes('image', (1, 1, h_opt, w_opt)),
        'max': _shapes('image', (1, 1, h_max, w_max)),
    }


def lightglue_profile():
    k_min = SP_LG.CONFIG['trt_lg_kpts_min']
    k_max = SP_LG.CONFIG['sp_max_keypoints']
    k_opt = max(k_min, k_max // 2)

    def spec(k):
        return ",".join([
            _shapes('kpts0', (1, k, 2)),
            _shapes('kpts1', (1, k, 2)),
            _shapes('desc0', (1, k, 256)),
            _shapes('desc1', (1, k, 256)),
        ])

    return {'min': spec(k_min), 'opt': spec(k_opt), 'max': spec(k_max)}


MODELS = {
    'superpoint':            {'onnx': ONNX_DIR / 'superpoint.onnx',
                              'profile': superpoint_profile},
    'superpoint_lightglue':  {'onnx': ONNX_DIR / 'superpoint_lightglue.onnx',
                              'profile': lightglue_profile},
}


def build_engine(model_name, precision):
    spec = MODELS[model_name]
    onnx_path = spec['onnx']
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} not found -- run export_to_onnx.py first.")

    profile = spec['profile']()
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    engine_path  = TRT_DIR / f"{model_name}_{precision}.engine"
    log_path     = TRT_DIR / f"{model_name}_{precision}.log"
    timing_cache = TRT_DIR / f"{model_name}_timing.cache"

    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--minShapes={profile['min']}",
        f"--optShapes={profile['opt']}",
        f"--maxShapes={profile['max']}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache}",
        *PRECISION_FLAGS[precision],
    ]

    print(f"\n{'=' * 70}\nBuilding {model_name} [{precision}]\n{'=' * 70}")
    print(" ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""))

    ok = result.returncode == 0 and "PASSED" in result.stdout
    print(f"[{'OK' if ok else 'FAILED'}] {model_name} [{precision}] "
          f"in {elapsed:.1f}s -> {engine_path if ok else log_path}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precisions', default=None,
                        help="comma-separated (default: SP_LG.CONFIG['trt_precisions'])")
    parser.add_argument('--models', default=None,
                        help="comma-separated (default: all models)")
    args = parser.parse_args()

    precisions = args.precisions.split(',') if args.precisions else SP_LG.CONFIG['trt_precisions']
    models     = args.models.split(',') if args.models else list(MODELS)

    print(f"[INFO] sp_max_keypoints={SP_LG.CONFIG['sp_max_keypoints']} "
          f"onnx_sp_resize={SP_LG.CONFIG['onnx_sp_resize']}")
    print(f"[INFO] Models: {models}  Precisions: {precisions}")
    print(f"[INFO] Output dir: {TRT_DIR}")

    results = {}
    for model_name in models:
        for precision in precisions:
            results[(model_name, precision)] = build_engine(model_name, precision)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for (model_name, precision), ok in results.items():
        print(f"  {model_name:24s} {precision:6s} {'OK' if ok else 'FAILED'}")

    # Tripwire against a silent no-op precision (this is exactly how --int4
    # was caught above): a lower-precision engine should be smaller than its
    # fp32 sibling. If it isn't, the precision flag likely quantized nothing.
    for model_name in models:
        fp32_path = TRT_DIR / f"{model_name}_fp32.engine"
        if not fp32_path.exists():
            continue
        for precision in precisions:
            if precision == 'fp32':
                continue
            other_path = TRT_DIR / f"{model_name}_{precision}.engine"
            if other_path.exists() and other_path.stat().st_size >= fp32_path.stat().st_size:
                print(f"[WARN] {model_name}_{precision}.engine is not smaller than "
                      f"{model_name}_fp32.engine ({other_path.stat().st_size} >= "
                      f"{fp32_path.stat().st_size} bytes) -- {precision} likely applied to "
                      f"zero layers. Verify with: trtexec --loadEngine={other_path} "
                      f"--profilingVerbosity=detailed --dumpLayerInfo --skipInference | grep -i {precision}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
