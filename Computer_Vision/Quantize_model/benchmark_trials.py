"""
Repeated-trial performance benchmark: SIFT_BFM / SIFT_LG / SP_LG
====================================================================
Runs each pipeline script N times as a fresh subprocess (python
<script>.py, unmodified, exactly as you'd run it by hand) and parses the
PERFORMANCE & MEMORY BENCHMARK block each one already prints via its own
PerformanceTracker, then reports mean/std/min/max/median across trials.

Why subprocess instead of importing and looping in-process
------------------------------------------------------------
SIFT_BFM.py and SIFT_LG.py only have a monolithic `if __name__ == '__main__':`
block (no run_pipeline()-style function to call repeatedly like SP_LG.py
has), and all three scripts use module-level CONFIG/globals that aren't
built for being re-entered in the same process. A fresh subprocess per
trial is also what actually gives a clean CUDA context each time -- no
leftover TensorRT/CUDA allocations from a previous trial to worry about
(see the GPU-OOM issue from earlier this session).

Trials run SEQUENTIALLY, never in parallel -- this device has 7.4GB of
memory shared between CPU and GPU, and concurrent trials would reproduce
exactly the OOM problem seen earlier when TensorRT context creation
competed with a heavy desktop/IDE for headroom.

Each script runs with whatever CONFIG is currently saved in its own file
(e.g. SP_LG.py's current CONFIG['backend']) -- this benchmarks your current
configuration, not a matrix of all backends. Edit CONFIG in the relevant
script yourself first if you want to benchmark a different backend/precision.

Usage:
    python benchmark_trials.py                       # 5 trials each, all 3 scripts
    python benchmark_trials.py --n 10
    python benchmark_trials.py --n 5 --scripts SP_LG,SIFT_LG
    python benchmark_trials.py --n 5 --timeout 300    # per-trial timeout (seconds)
"""

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    'SIFT_BFM': THIS_DIR / 'SIFT_BFM.py',
    'SIFT_LG':  THIS_DIR / 'SIFT_LG.py',
    'SP_LG':    THIS_DIR / 'SP_LG.py',
}

_RE_TOTAL_TIME = re.compile(r'Total Execution Time\s*:\s*([\d.]+)\s*seconds')
_RE_PEAK_RAM   = re.compile(r'Peak RAM \(RSS\)\s*:\s*([\d.]+)\s*MB')
_RE_PEAK_GPU   = re.compile(r'Peak GPU VRAM\s*:\s*([\d.]+)\s*MB')
_RE_STEP       = re.compile(r'^\s*-\s*(.+?)\s*:\s*([\d.]+)\s*s\s*\(', re.MULTILINE)


# Substrings seen (twice, in this exact session) when this device's shared
# CPU/GPU memory is under pressure from other processes (VS Code/Pylance,
# desktop environment, etc.) at the moment CUDA/TensorRT tries to allocate --
# a transient system condition, not a bug in the script being benchmarked.
# A trial that fails this way is retried rather than counted as a real
# failure; see the "tragic VS Code" conversation from earlier this session.
_TRANSIENT_OOM_MARKERS = (
    'out of memory',
    'outofmemory',
    'cuda initialization failure',
    'nvmapmemalloc',
    'create_execution_context() returned none',
)


def _looks_like_transient_oom(text):
    lower = text.lower()
    return any(marker in lower for marker in _TRANSIENT_OOM_MARKERS)


def _run_trial_once(script_path, timeout):
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(THIS_DIR),
        )
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'timed out after {timeout}s', 'wall_time': time.time() - t0}

    wall_time = time.time() - t0
    output = (result.stdout or '') + '\n' + (result.stderr or '')

    if result.returncode != 0:
        tail = '\n'.join(output.strip().splitlines()[-15:])
        return {'ok': False, 'error': f'exit code {result.returncode}\n{tail}',
                'wall_time': wall_time, 'transient_oom': _looks_like_transient_oom(output)}

    m_total = _RE_TOTAL_TIME.search(output)
    m_ram   = _RE_PEAK_RAM.search(output)
    m_gpu   = _RE_PEAK_GPU.search(output)
    if not m_total:
        tail = '\n'.join(output.strip().splitlines()[-15:])
        return {'ok': False, 'error': f'could not find benchmark output\n{tail}',
                'wall_time': wall_time, 'transient_oom': False}

    steps = {name.strip(): float(secs) for name, secs in _RE_STEP.findall(output)}

    return {
        'ok': True,
        'wall_time': wall_time,
        'total_time_s': float(m_total.group(1)),
        'peak_ram_mb': float(m_ram.group(1)) if m_ram else None,
        'peak_gpu_mb': float(m_gpu.group(1)) if m_gpu else None,
        'steps': steps,
    }


def run_trial(script_path, timeout, max_retries=2, retry_delay=10):
    """Runs one trial, retrying (with a short delay to let memory settle)
    when the failure looks like the transient shared-memory OOM pattern
    described above. A genuine bug in the script under test won't match
    that pattern and will surface immediately, not after wasted retries."""
    attempt = 0
    while True:
        r = _run_trial_once(script_path, timeout)
        if r['ok'] or not r.get('transient_oom') or attempt >= max_retries:
            r['retries'] = attempt
            return r
        attempt += 1
        print(f"[transient GPU memory pressure, retry {attempt}/{max_retries} in {retry_delay}s] ", end='', flush=True)
        time.sleep(retry_delay)


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        'n': len(values),
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
    }


def _fmt_stats(s, unit=''):
    if s is None:
        return 'N/A'
    return f"{s['mean']:.2f} ± {s['std']:.2f} {unit} (min {s['min']:.2f}, max {s['max']:.2f}, n={s['n']})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='trials per script (default: 5)')
    parser.add_argument('--scripts', default=None,
                        help=f"comma-separated subset of {list(SCRIPTS)} (default: all)")
    parser.add_argument('--timeout', type=int, default=300, help='per-trial timeout in seconds')
    args = parser.parse_args()

    names = args.scripts.split(',') if args.scripts else list(SCRIPTS)
    for name in names:
        if name not in SCRIPTS:
            raise SystemExit(f"Unknown script {name!r}, choose from {list(SCRIPTS)}")

    all_trials = {}
    for name in names:
        script_path = SCRIPTS[name]
        print(f"\n{'=' * 70}\n{name}  ({script_path.name}, {args.n} trials)\n{'=' * 70}")
        trials = []
        for i in range(1, args.n + 1):
            print(f"  Trial {i}/{args.n} ...", end=' ', flush=True)
            r = run_trial(script_path, args.timeout)
            trials.append(r)
            if r['ok']:
                gpu_str = f", gpu={r['peak_gpu_mb']:.0f}MB" if r['peak_gpu_mb'] is not None else ""
                retry_str = f" (after {r['retries']} retr{'y' if r['retries'] == 1 else 'ies'})" if r.get('retries') else ""
                print(f"OK  total={r['total_time_s']:.2f}s  ram={r['peak_ram_mb']:.0f}MB{gpu_str}{retry_str}")
            else:
                print(f"FAILED  ({r['error'].splitlines()[0]})")
        all_trials[name] = trials

    print(f"\n{'=' * 70}\nSTATISTICAL SUMMARY ({args.n} trials/script)\n{'=' * 70}")
    for name in names:
        trials = all_trials[name]
        ok_trials = [t for t in trials if t['ok']]
        n_failed = len(trials) - len(ok_trials)

        print(f"\n{name}:")
        if n_failed:
            print(f"  [WARN] {n_failed}/{len(trials)} trial(s) failed -- stats below exclude them")
        if not ok_trials:
            print("  No successful trials.")
            continue

        total_stats = _stats([t['total_time_s'] for t in ok_trials])
        ram_stats   = _stats([t['peak_ram_mb'] for t in ok_trials])
        gpu_stats   = _stats([t['peak_gpu_mb'] for t in ok_trials])

        print(f"  Total time    : {_fmt_stats(total_stats, 's')}")
        print(f"  Peak RAM      : {_fmt_stats(ram_stats, 'MB')}")
        print(f"  Peak GPU VRAM : {_fmt_stats(gpu_stats, 'MB')}")

        # Per-step breakdown -- union of step names seen, averaged over
        # trials that reported that step (step names/order can vary slightly
        # between scripts/backends, e.g. SP_LG's onnx vs tensorrt vs pytorch).
        step_names = []
        for t in ok_trials:
            for step in t['steps']:
                if step not in step_names:
                    step_names.append(step)
        if step_names:
            print(f"  Step breakdown (mean ± std over trials that reported it):")
            for step in step_names:
                step_stats = _stats([t['steps'].get(step) for t in ok_trials])
                print(f"    - {step:<38}: {_fmt_stats(step_stats, 's')}")

    # Cross-script comparison table (only meaningful for scripts with data)
    print(f"\n{'=' * 70}\nCROSS-SCRIPT COMPARISON\n{'=' * 70}")
    header = f"{'script':<12} {'total (s)':<22} {'peak RAM (MB)':<22} {'peak GPU (MB)':<22}"
    print(header)
    print('-' * len(header))
    for name in names:
        ok_trials = [t for t in all_trials[name] if t['ok']]
        if not ok_trials:
            print(f"{name:<12} (no successful trials)")
            continue
        t_s = _stats([t['total_time_s'] for t in ok_trials])
        r_s = _stats([t['peak_ram_mb'] for t in ok_trials])
        g_s = _stats([t['peak_gpu_mb'] for t in ok_trials])
        gpu_col = f"{g_s['mean']:.1f} ± {g_s['std']:.1f}" if g_s else 'N/A'
        print(f"{name:<12} "
              f"{t_s['mean']:>7.2f} ± {t_s['std']:<9.2f} "
              f"{r_s['mean']:>7.1f} ± {r_s['std']:<9.1f} "
              f"{gpu_col:<22}")


if __name__ == '__main__':
    main()
