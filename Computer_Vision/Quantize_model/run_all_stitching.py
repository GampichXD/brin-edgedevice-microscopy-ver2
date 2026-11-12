"""
Run all 4 tile-stitching methods back to back and collect their outputs
========================================================================
Methods: brute_force, sift_bfm, sift_lg, sp_lg

For one tile folder, each method's script is launched as a fresh subprocess
(clean CUDA/TensorRT context per method). Two combined files are produced:

  <output-dir>/all_outputs.txt   full console output (stdout+stderr) of every
                                 method, plus a short summary table at the top
  <output-dir>/<method>/*.csv   the individual CSVs (each method runs with
                                 this folder as cwd, so the old CSVs next to
                                 the scripts are not overwritten)
  <output-dir>/all_metrics.csv   every CSV each method saved (overlap metrics,
                                 homography reprojection), merged into one
                                 table with extra columns: method, csv_type,
                                 source_csv

CONFIG overrides (in memory only -- the .py files on disk are NOT modified):
  evaluate_metrics = True   (otherwise the scripts write no CSVs at all)
  debug            = False  (otherwise plt.show() blocks, e.g. brute_force.py)
Use --no-metrics / --keep-debug to leave those flags as they are in the files.

Usage:
    python run_all_stitching.py --path /path/to/tiles
    python run_all_stitching.py --path /path/to/tiles --methods sift_lg sp_lg
    python run_all_stitching.py --path /path/to/tiles --output-dir results/run1 --timeout 1800
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

METHODS = {
    'brute_force': THIS_DIR / 'brute_force.py',
    'sift_bfm':    THIS_DIR / 'SIFT_BFM.py',
    'sift_lg':     THIS_DIR / 'SIFT_LG.py',
    'sp_lg':       THIS_DIR / 'SP_LG.py',
}

_RE_SAVED_CSV = re.compile(r'Saved:\s*(\S+\.csv)')
_RE_SAVED_JPG = re.compile(r'Saved:\s*(\S+\.jpg)')
_RE_TOTAL_TIME = re.compile(r'Total Execution Time\s*:\s*([\d.]+)\s*seconds')
_RE_PEAK_RAM = re.compile(r'Peak RAM \(RSS\)\s*:\s*([\d.]+)\s*MB')
_RE_PEAK_GPU = re.compile(r'Peak GPU VRAM\s*:\s*([\d.]+)\s*MB')

# Runs inside the child process: loads the target script's source, flips the
# requested CONFIG flags (first occurrence only = the CONFIG dict entry), and
# executes it as __main__ with the given argv.
_BOOTSTRAP = r'''
import os, re, sys
script = sys.argv[1]
overrides = [kv.split("=", 1) for kv in os.environ.get("STITCH_OVERRIDES", "").split(",") if kv]
sys.argv = [script] + sys.argv[2:]
src = open(script, encoding="utf-8").read()
for key, val in overrides:
    pat = r"(['\"]" + re.escape(key) + r"['\"]\s*:\s*)(True|False)"
    src, n = re.subn(pat, lambda m: m.group(1) + val, src, count=1)
    print(f"[RUNNER] CONFIG override {key}={val}" + ("" if n else "  (key not found, ignored)"), flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
exec(compile(src, script, "exec"), g)
'''


def run_method(name, script, tile_path, overrides, timeout, python, work_dir):
    """Run one method (cwd=work_dir, so its CSVs land there), streaming output live."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLBACKEND'] = 'Agg'          # never pop up / block on plot windows
    env['STITCH_OVERRIDES'] = ','.join(f'{k}={v}' for k, v in overrides.items())

    cmd = [python, '-c', _BOOTSTRAP, str(Path(script).resolve()), '--path', str(tile_path)]
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill) if timeout and timeout > 0 else None
    if timer:
        timer.start()

    lines = []
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)
        proc.wait()
    finally:
        if timer:
            timer.cancel()

    output = ''.join(lines)
    wall = time.time() - start

    if timed_out.is_set():
        error = f'timed out after {timeout}s'
    elif proc.returncode != 0:
        error = f'exit code {proc.returncode}'
    else:
        error = ''

    def _last(rx):
        m = rx.findall(output)
        return m[-1] if m else ''

    # Resolve CSV paths (relative ones are relative to work_dir, the child's cwd)
    csvs = []
    for p in dict.fromkeys(_RE_SAVED_CSV.findall(output)):   # de-dup, keep order
        path = Path(p) if Path(p).is_absolute() else work_dir / p
        csvs.append(path)

    return {
        'method': name,
        'script': script.name,
        'ok': not error,
        'error': error,
        'wall_time_s': round(wall, 3),
        'total_time_s': _last(_RE_TOTAL_TIME),
        'peak_ram_mb': _last(_RE_PEAK_RAM),
        'peak_gpu_mb': _last(_RE_PEAK_GPU),
        'stitched_image': _last(_RE_SAVED_JPG),
        'csvs': csvs,
        'output': output,
    }


def csv_type_of(path):
    stem = path.stem
    for prefix in ('overlap_metrics', 'homography_reprojection'):
        if stem.startswith(prefix):
            return prefix
    return stem


def write_combined_csv(results, out_path):
    """Merge every per-method CSV into one long table (union of columns)."""
    lead = ['method', 'csv_type', 'source_csv']
    columns, rows = [], []
    for r in results:
        for path in r['csvs']:
            if not path.exists():
                print(f"[WARN] {r['method']}: reported CSV not found: {path}")
                continue
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for col in reader.fieldnames or []:
                    if col not in columns:
                        columns.append(col)
                for row in reader:
                    row.update(method=r['method'], csv_type=csv_type_of(path), source_csv=path.name)
                    rows.append(row)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=lead + columns, restval='')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summary_table(results):
    hdr = f"{'method':<12} {'status':<8} {'wall[s]':>9} {'total[s]':>9} {'peakRAM[MB]':>12} {'peakGPU[MB]':>12} {'#csv':>5}"
    out = [hdr, '-' * len(hdr)]
    for r in results:
        out.append(f"{r['method']:<12} {'OK' if r['ok'] else 'FAILED':<8} {r['wall_time_s']:>9.2f} "
                   f"{r['total_time_s'] or '-':>9} {r['peak_ram_mb'] or '-':>12} "
                   f"{r['peak_gpu_mb'] or '-':>12} {len(r['csvs']):>5}")
        if r['error']:
            out.append(f"    error: {r['error']}")
        if r['stitched_image']:
            out.append(f"    image: {r['stitched_image']}")
    return '\n'.join(out)


def write_combined_txt(results, out_path, tile_path, overrides):
    bar = '#' * 78
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Tile stitching comparison run\n")
        f.write(f"Date      : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Tile path : {tile_path}\n")
        f.write(f"Overrides : {overrides or 'none'}\n\n")
        f.write("SUMMARY\n" + summary_table(results) + "\n\n")
        for r in results:
            status = 'OK' if r['ok'] else f"FAILED: {r['error']}"
            f.write(f"{bar}\n# {r['method'].upper()}  ({r['script']})  [{status}, wall {r['wall_time_s']:.2f}s]\n{bar}\n")
            f.write(r['output'])
            f.write('\n\n')


def main():
    parser = argparse.ArgumentParser(description="Run all tile-stitching methods and collect outputs")
    parser.add_argument('--path', required=True, help="Folder of tile images to stitch")
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS),
                        help="Methods to run (default: all, in this order)")
    parser.add_argument('--output-dir', default=None,
                        help="Where to write all_outputs.txt / all_metrics.csv "
                             "(default: stitching_runs/<timestamp> next to this script)")
    parser.add_argument('--timeout', type=int, default=1800, help="Per-method timeout in seconds (0 = none)")
    parser.add_argument('--metrics', action=argparse.BooleanOptionalAction, default=True,
                        help="Force evaluate_metrics=True so CSVs are written (--no-metrics to keep file setting)")
    parser.add_argument('--keep-debug', action='store_true',
                        help="Don't force debug=False (plots may block the run)")
    parser.add_argument('--python', default=sys.executable, help="Python interpreter for the method scripts")
    args = parser.parse_args()

    tile_path = Path(args.path).resolve()
    if not tile_path.is_dir():
        parser.error(f"--path is not a directory: {tile_path}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else \
        THIS_DIR / 'stitching_runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.metrics:
        overrides['evaluate_metrics'] = 'True'
    if not args.keep_debug:
        overrides['debug'] = 'False'

    results = []
    for i, name in enumerate(args.methods, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(args.methods)}] {name}  ({METHODS[name].name})\n{'=' * 78}", flush=True)
        r = run_method(name, METHODS[name], tile_path, overrides, args.timeout, args.python,
                       work_dir=out_dir / name)
        results.append(r)
        print(f"[RUNNER] {name}: {'OK' if r['ok'] else 'FAILED (' + r['error'] + ')'} "
              f"in {r['wall_time_s']:.1f}s, {len(r['csvs'])} CSV(s)", flush=True)

    txt_path = out_dir / 'all_outputs.txt'
    csv_path = out_dir / 'all_metrics.csv'
    write_combined_txt(results, txt_path, tile_path, overrides)
    n_rows = write_combined_csv(results, csv_path)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}\n{summary_table(results)}")
    print(f"\nCombined output : {txt_path}")
    print(f"Combined CSV    : {csv_path}  ({n_rows} rows)")
    sys.exit(0 if all(r['ok'] for r in results) else 1)


if __name__ == '__main__':
    main()
