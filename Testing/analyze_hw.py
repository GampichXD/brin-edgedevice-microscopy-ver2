#!/usr/bin/env python3
"""
analyze_hw.py — Analisis data uji hardware Smart Digital Microscope.

Membaca CSV di folder data (default: ../test_logs) yang dihasilkan skrip
Testing/t0x_*.py dan logger Utils/test_logger.py, lalu menghasilkan:
  - grafik PNG (300 dpi) di <out>/figures/
  - tabel metrik  <out>/summary.md  (siap tempel ke Bab IV) dan summary.json

Skema CSV (WAJIB sama dengan yang ditulis skrip pengumpul; lihat
INSTRUKSI_HARDWARE.md Bagian 6). File yang tidak ada akan dilewati.

  positioning.csv   axis,target_mm,run,direction,measured_mm,valid,score
  cmd_local.csv     cmd_id,t_sent_ns,t_ok_ns
  cmd_client.csv    cmd_id,path,t_sent_ns                (dari laptop)
  cmd.csv           cmd_id,gcode,t_rx_ns,t_ok_ns         (logger Jetson)
  video.csv         frame_id,t_cap_ns,t_recv_ns,bytes
  eta.csv           grid,run,n_tiles,predicted_s,actual_s
  ratelimit.csv     t_send_ns,channel,result             (result: ACCEPTED/REJECTED)
  telemetry.csv     t_ns,status,x,y,z,cpu,cpu_temp,gpu_temp,ram_used_gb,ram_total_gb
  watchdog_log.csv  (opsional; tidak dihasilkan versi ini -> bagian D7 dilewati)
  settle.csv        move_type,delay_s,rep,sharpness

Pemakaian:
  python3 analyze_hw.py --data ../test_logs --out ../test_results
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2, C3 = "#2563eb", "#dc2626", "#16a34a"   # warna konsisten dengan UI


def _read(data, name):
    p = os.path.join(data, name)
    if not os.path.exists(p):
        print(f"  - {name}: tidak ada, dilewati")
        return None
    df = pd.read_csv(p)
    print(f"  - {name}: {len(df)} baris")
    return df


def _stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {}
    return {"n": int(len(x)), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "min": float(np.min(x)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def _save(fig, out, name):
    path = os.path.join(out, "figures", name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ D1 posisi
def positioning(df, out, R):
    if "valid" in df.columns:
        df = df[df["valid"] == 1]
    """ISO 230-2 (disederhanakan): per target & arah hitung rata-rata deviasi,
    simpangan baku; backlash B = x̄↑ − x̄↓; repeatability R = 4s;
    akurasi A = max(x̄+2s) − min(x̄−2s) (dua arah)."""
    res = {}
    for axis, g in df.groupby("axis"):
        g = g.copy()
        g["dev_um"] = (g["measured_mm"] - g["target_mm"]) * 1000
        agg = g.groupby(["target_mm", "direction"])["dev_um"].agg(["mean", "std", "count"]).reset_index()
        agg["std"] = agg["std"].fillna(0)
        piv_m = agg.pivot(index="target_mm", columns="direction", values="mean")
        piv_s = agg.pivot(index="target_mm", columns="direction", values="std")
        backlash = (piv_m.get("fwd") - piv_m.get("bwd")).abs() if {"fwd", "bwd"} <= set(piv_m.columns) else pd.Series(dtype=float)
        rep = 4 * agg["std"].max()
        acc = float((agg["mean"] + 2 * agg["std"]).max() - (agg["mean"] - 2 * agg["std"]).min())
        rmse = float(np.sqrt(np.mean(g["dev_um"] ** 2)))
        res[axis] = {"accuracy_um": acc, "repeatability_um": float(rep),
                     "backlash_mean_um": float(backlash.mean()) if len(backlash) else None,
                     "backlash_max_um": float(backlash.max()) if len(backlash) else None,
                     "rmse_um": rmse, "n": int(len(g))}

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for d, col, mk, off in [("fwd", C1, "o", -0.01), ("bwd", C2, "s", 0.01)]:
            if d in piv_m:
                ax.errorbar(piv_m.index + off * piv_m.index.max(), piv_m[d], yerr=2 * piv_s[d],
                            fmt=mk + "-", color=col, capsize=4, label="maju (↑)" if d == "fwd" else "mundur (↓)")
        ax.axhline(0, color="#6b7280", lw=1)
        ax.set_xlabel(f"Posisi target sumbu {axis} (mm)")
        ax.set_ylabel("Deviasi posisi (µm)")
        ax.set_title(f"Akurasi & repeatability sumbu {axis} (rata-rata ± 2s)")
        ax.legend()
        _save(fig, out, f"D1_positioning_{axis}.png")

    R.append("## D1. Akurasi & Repeatability Posisi (ISO 230-2, disederhanakan)\n")
    R.append("| Sumbu | n | Akurasi A (µm) | Repeatability R=4s (µm) | Backlash rata-rata (µm) | Backlash maks (µm) | RMSE (µm) |")
    R.append("|---|---|---|---|---|---|---|")
    for a, v in res.items():
        R.append(f"| {a} | {v['n']} | {v['accuracy_um']:.1f} | {v['repeatability_um']:.1f} | "
                 f"{(v['backlash_mean_um'] or 0):.1f} | {(v['backlash_max_um'] or 0):.1f} | {v['rmse_um']:.1f} |")
    R.append("")
    return res


# ------------------------------------------------------------------ D2 latensi perintah
def command_latency(local, client, edge, out, R):
    groups, res = {}, {}
    if local is not None:
        groups["Lokal (Jetson→GRBL)"] = (local["t_ok_ns"] - local["t_sent_ns"]) / 1e6
    if client is not None and edge is not None:
        m = client.merge(edge, on="cmd_id", how="left")
        lost = int(m["t_rx_ns"].isna().sum())
        m = m.dropna(subset=["t_rx_ns"])
        for path, g in m.groupby("path"):
            groups[f"{path.capitalize()} end-to-end"] = (g["t_ok_ns"] - g["t_sent_ns"]) / 1e6
            groups[f"{path.capitalize()} satu-arah (kirim→Jetson)"] = (g["t_rx_ns"] - g["t_sent_ns"]) / 1e6
        res["lost_commands"] = lost
    if not groups:
        return res
    for k, v in groups.items():
        res[k] = _stats(v)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.boxplot(list(groups.values()), showfliers=True)
    ax.set_xticks(range(1, len(groups) + 1), list(groups.keys()), rotation=12, ha="right")
    ax.set_ylabel("Latensi (ms)")
    ax.set_title("Latensi perintah motor")
    _save(fig, out, "D2_command_latency_boxplot.png")

    R.append("## D2. Latensi Perintah Motor\n")
    R.append("| Jalur | n | Rata-rata (ms) | SD | p50 | p95 | Maks |")
    R.append("|---|---|---|---|---|---|---|")
    for k, s in res.items():
        if isinstance(s, dict) and s:
            R.append(f"| {k} | {s['n']} | {s['mean']:.1f} | {s['sd']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
    if "lost_commands" in res:
        R.append(f"\nPerintah tidak sampai ke Jetson: **{res['lost_commands']}**.")
    R.append("")
    return res


# ------------------------------------------------------------------ D3 video
def video(df, out, R):
    df = df.sort_values("t_recv_ns")
    lat = (df["t_recv_ns"] - df["t_cap_ns"]) / 1e6
    t = (df["t_recv_ns"] - df["t_recv_ns"].iloc[0]) / 1e9
    sec = np.floor(t).astype(int)
    fps = df.groupby(sec).size()
    jitter = np.diff(df["t_recv_ns"].values) / 1e6
    kbps = df.groupby(sec)["bytes"].sum() * 8 / 1000
    res = {"latency_ms": _stats(lat), "fps": _stats(fps.values), "inter_frame_ms": _stats(jitter),
           "bitrate_kbps": _stats(kbps.values)}

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].hist(lat, bins=40, color=C1)
    axs[0].axvline(np.percentile(lat, 95), color=C2, ls="--", label="p95")
    axs[0].set_xlabel("Latensi glass-to-browser (ms)"); axs[0].set_ylabel("Jumlah frame"); axs[0].legend()
    axs[0].set_title("Distribusi latensi video")
    axs[1].plot(fps.index, fps.values, color=C3)
    axs[1].set_xlabel("Waktu (s)"); axs[1].set_ylabel("FPS diterima"); axs[1].set_title("FPS terhadap waktu")
    _save(fig, out, "D3_video_latency_fps.png")

    R.append("## D3. Latensi Video & FPS\n")
    R.append("| Metrik | n | Rata-rata | SD | p50 | p95 | Maks |")
    R.append("|---|---|---|---|---|---|---|")
    for k, lab in [("latency_ms", "Latensi (ms)"), ("fps", "FPS"), ("inter_frame_ms", "Jarak antar-frame (ms)"), ("bitrate_kbps", "Bitrate (kbps)")]:
        s = res[k]
        R.append(f"| {lab} | {s['n']} | {s['mean']:.1f} | {s['sd']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
    R.append("")
    return res


# ------------------------------------------------------------------ D4 ETA
def eta(df, out, R):
    df = df.copy()
    df["ape"] = (df["actual_s"] - df["predicted_s"]).abs() / df["actual_s"] * 100
    df["ovh"] = (df["actual_s"] - df["predicted_s"]) / df["n_tiles"]
    per = df.groupby("grid").agg(n_tiles=("n_tiles", "first"), pred=("predicted_s", "mean"),
                                 actual=("actual_s", "mean"), actual_sd=("actual_s", "std"),
                                 mape=("ape", "mean"), ovh=("ovh", "mean")).reset_index().sort_values("n_tiles")
    # Model diperluas: prediksi + n_tiles * overhead capture rata-rata (diestimasi dari data)
    t_cap = float(df["ovh"].mean())
    ape2 = ((df["actual_s"] - (df["predicted_s"] + df["n_tiles"] * t_cap)).abs() / df["actual_s"] * 100)
    res = {"MAPE_total_pct": float(df["ape"].mean()), "per_grid": per.to_dict("records"),
           "t_capture_mean_s": t_cap, "MAPE_with_capture_pct": float(ape2.mean())}
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(df["predicted_s"], df["actual_s"], color=C1, zorder=3)
    lim = [0, max(df["predicted_s"].max(), df["actual_s"].max()) * 1.08]
    ax.plot(lim, lim, color="#6b7280", ls="--", label="y = x (prediksi sempurna)")
    for _, r in per.iterrows():
        ax.annotate(r["grid"], (r["pred"], r["actual"]), textcoords="offset points", xytext=(6, -10), fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Prediksi persamaan (3)–(4) (s)"); ax.set_ylabel("Waktu aktual (s)")
    ax.set_title(f"Validasi estimasi waktu scan\nMAPE = {res['MAPE_total_pct']:.1f}%"); ax.legend(loc="upper left")
    _save(fig, out, "D4_eta_scatter.png")

    R.append("## D4. Validasi Estimasi Waktu Grid Scan (persamaan 3–4)\n")
    R.append("| Grid | Tile | Prediksi (s) | Aktual rata-rata (s) | SD | MAPE (%) | Overhead/tile (s) |")
    R.append("|---|---|---|---|---|---|---|")
    for r in res["per_grid"]:
        sd = r['actual_sd'] if r['actual_sd'] == r['actual_sd'] else 0.0
        R.append(f"| {r['grid']} | {r['n_tiles']} | {r['pred']:.1f} | {r['actual']:.1f} | {sd:.1f} | {r['mape']:.1f} | {r['ovh']:.2f} |")
    R.append(f"\nMAPE persamaan (3)–(4) saja: **{res['MAPE_total_pct']:.2f}%**. "
             f"Overhead capture+unggah rata-rata **{res['t_capture_mean_s']:.2f} s/tile**; "
             f"model diperluas `T = T_pred + n·t_capture` memberi MAPE **{res['MAPE_with_capture_pct']:.2f}%** "
             f"(t_capture diestimasi dari data yang sama — untuk validasi yang adil, estimasi dari grid kecil lalu uji di grid besar).\n")
    return res


# ------------------------------------------------------------------ D5 rate limit
def ratelimit(df, out, R):
    res = {}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for ch, g in df.groupby("channel"):
        t0 = g["t_send_ns"].min()
        sec = ((g["t_send_ns"] - t0) / 1e9).astype(int)
        acc = g[g["result"] == "ACCEPTED"].groupby(sec[g["result"] == "ACCEPTED"]).size()
        sent = g.groupby(sec).size()
        rate = acc.reindex(sent.index, fill_value=0)
        res[ch] = {"sent_per_s": float(sent.mean()), "accepted_per_s_mean": float(rate.mean()),
                   "accepted_per_s_max": int(rate.max()), "rejected_total": int((g["result"] != "ACCEPTED").sum())}
        ax.plot(rate.index, rate.values, marker="o", label=f"{ch.upper()} diterima/detik")
    ax.axhline(2, color=C2, ls="--", label="batas REST 2/detik (Δt_min = 0,5 s)")
    ax.axhline(4, color=C3, ls=":", label="batas WS 4/detik (Δt_min = 0,25 s)")
    ax.set_xlabel("Detik ke-"); ax.set_ylabel("Perintah diterima"); ax.set_title("Uji rate-limit (burst)")
    ax.legend()
    _save(fig, out, "D5_ratelimit.png")
    R.append("## D5. Uji Rate-Limit (persamaan 6)\n")
    R.append("| Kanal | Dikirim/detik | Diterima/detik (rata-rata) | Diterima/detik (maks) | Total ditolak |")
    R.append("|---|---|---|---|---|")
    for ch, v in res.items():
        R.append(f"| {ch} | {v['sent_per_s']:.1f} | {v['accepted_per_s_mean']:.2f} | {v['accepted_per_s_max']} | {v['rejected_total']} |")
    R.append("")
    return res


# ------------------------------------------------------------------ D6 termal & resource
def telemetry(df, out, R):
    t = (df["t_ns"] - df["t_ns"].iloc[0]) / 60e9
    ram_pct = df["ram_used_gb"] / df["ram_total_gb"] * 100
    res = {"cpu_temp_max": float(df["cpu_temp"].max()), "gpu_temp_max": float(df["gpu_temp"].max()),
           "ram_pct_max": float(ram_pct.max()), "ram_pct_mean": float(ram_pct.mean()),
           "cpu_pct_mean": float(df["cpu"].mean()), "over_75pct_ram_s": float((ram_pct > 75).sum() * np.median(np.diff(df["t_ns"])) / 1e9)}
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].plot(t, df["cpu_temp"], color=C2, label="CPU")
    axs[0].plot(t, df["gpu_temp"], color="#f97316", label="GPU")
    axs[0].set_ylabel("Suhu (°C)"); axs[0].legend(); axs[0].set_title("Termal & sumber daya Jetson selama grid scan")
    axs[1].plot(t, ram_pct, color=C1, label="RAM (%)")
    axs[1].plot(t, df["cpu"], color=C3, label="CPU (%)", alpha=0.7)
    axs[1].axhline(75, color=C2, ls="--", label="Batas RAM 75% (proposal)")
    axs[1].set_ylabel("Penggunaan (%)"); axs[1].set_xlabel("Waktu (menit)"); axs[1].legend(loc="lower right")
    _save(fig, out, "D6_thermal_resource.png")
    R.append("## D6. Termal & Sumber Daya Jetson\n")
    R.append("| Suhu CPU maks (°C) | Suhu GPU maks (°C) | RAM maks (%) | RAM rata-rata (%) | CPU rata-rata (%) | Durasi RAM > 75% (s) |")
    R.append("|---|---|---|---|---|---|")
    R.append(f"| {res['cpu_temp_max']:.1f} | {res['gpu_temp_max']:.1f} | {res['ram_pct_max']:.1f} | {res['ram_pct_mean']:.1f} | {res['cpu_pct_mean']:.1f} | {res['over_75pct_ram_s']:.0f} |\n")
    return res


# ------------------------------------------------------------------ D7 watchdog
def watchdog(wd, tel, out, R):
    trig = wd[wd["event"] == "WATCHDOG_TRIGGERED"].copy()
    # Pemicu dengan umur pesan < 1 s terjadi tepat setelah link pulih lalu putus lagi (Wi-Fi baru
    # menyala); motor sudah Hold dari pemicu sebelumnya, jadi bukan percobaan baru.
    flaps = int((trig["last_rx_age_ms"] < 1000).sum())
    trig = trig[trig["last_rx_age_ms"] >= 1000]
    if trig.empty:
        return None
    holds = wd[wd["event"] == "FEED_HOLD_SENT"]["t_ns"].sort_values().to_numpy()
    rows = []
    for _, r in trig.iterrows():
        t_last_rx = r["t_ns"] - r["last_rx_age_ms"] * 1e6
        # Waktu berhenti diukur dari FEED_HOLD_SENT: telemetri tidak tercatat selama link putus.
        nxt = holds[holds >= r["t_ns"]]
        hold_ms = (nxt[0] - t_last_rx) / 1e6 if len(nxt) else np.nan
        rows.append({"detect_ms": r["last_rx_age_ms"], "hold_ms": hold_ms, "moving": r.get("motor_state") == "Run"})
    d = pd.DataFrame(rows)
    res = {"n": int(len(d)), "n_moving": int(d["moving"].sum()), "flaps_excluded": flaps,
           "detect_ms": _stats(d["detect_ms"]), "hold_ms": _stats(d["hold_ms"])}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(d["detect_ms"], bins=12, color=C1, alpha=0.85, label="Pesan terakhir → watchdog terpicu")
    ax.axvline(3000, color=C2, ls="--", lw=1, label="WATCHDOG_TIMEOUT = 3 s")
    ax.set_xlabel("Waktu (ms)"); ax.set_ylabel("Jumlah percobaan"); ax.set_title(f"Uji watchdog keep-alive (n = {len(d)})")
    ax.legend()
    _save(fig, out, "D7_watchdog.png")
    R.append("## D7. Uji Watchdog Keep-Alive (fail-safe)\n")
    R.append(f"{res['n']} percobaan ({res['n_moving']} saat motor bergerak). {flaps} pemicu ulang saat Wi-Fi "
             "tersambung-putus kembali tidak dihitung. Waktu berhenti diukur dari `FEED_HOLD_SENT` karena "
             "telemetri tidak tercatat selama koneksi putus.\n")
    R.append("| Metrik | n | Rata-rata (ms) | SD | p95 | Maks |")
    R.append("|---|---|---|---|---|---|")
    for k, lab in [("detect_ms", "Pesan terakhir → watchdog terpicu"), ("hold_ms", "Pesan terakhir → feed hold terkirim")]:
        s_ = res[k]
        if s_:
            R.append(f"| {lab} | {s_['n']} | {s_['mean']:.0f} | {s_['sd']:.0f} | {s_['p95']:.0f} | {s_['max']:.0f} |")
    R.append("")
    return res


# ------------------------------------------------------------------ D8 settle / ketajaman
def settle(df, out, R):
    """Ketajaman (variance of Laplacian) vs jeda setelah gerak. Jeda yang cukup =
    jeda terkecil yang ketajamannya >= 95% dari ketajaman maksimum (plateau)."""
    res = {}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for (mt, col) in [("in_row", C1), ("row_change", C2)]:
        g = df[df["move_type"] == mt]
        if g.empty:
            continue
        agg = g.groupby("delay_s")["sharpness"].agg(["mean", "std"]).reset_index()
        ref = agg["mean"].max()
        ok = agg[agg["mean"] >= 0.95 * ref]
        res[mt] = {"min_delay_s": float(ok["delay_s"].min()), "table": agg.to_dict("records")}
        ax.errorbar(agg["delay_s"], agg["mean"] / ref * 100, yerr=agg["std"] / ref * 100, fmt="o-",
                    color=col, capsize=4, label="gerak dalam baris" if mt == "in_row" else "pindah baris")
    ax.axhline(95, color="#6b7280", ls="--", label="95% ketajaman maks")
    ax.set_xlabel("Jeda setelah gerak selesai (s)"); ax.set_ylabel("Ketajaman relatif (%)")
    ax.set_title("Ketajaman citra vs jeda settle"); ax.legend()
    _save(fig, out, "D8_settle_sharpness.png")
    R.append("## D8. Jeda Settle vs Ketajaman Citra\n")
    R.append("| Jenis gerak | Jeda minimum agar ketajaman ≥ 95% maks (s) |")
    R.append("|---|---|")
    for mt, v in res.items():
        R.append(f"| {'Dalam baris' if mt == 'in_row' else 'Pindah baris'} | {v['min_delay_s']:.2f} |")
    R.append("")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "..", "test_logs"))
    ap.add_argument("--out", default=os.path.join(here, "..", "test_results"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    print(f"Membaca data dari {os.path.abspath(a.data)}")
    R = ["# Ringkasan Hasil Pengujian Hardware\n", f"Sumber data: `{os.path.abspath(a.data)}`\n"]
    S = {}
    rd = lambda n: _read(a.data, n)
    pos = rd("positioning.csv")
    if pos is not None: S["positioning"] = positioning(pos, a.out, R)
    loc, cli, edg = rd("cmd_local.csv"), rd("cmd_client.csv"), rd("cmd.csv")
    if loc is not None or (cli is not None and edg is not None): S["command_latency"] = command_latency(loc, cli, edg, a.out, R)
    vid = rd("video.csv")
    if vid is not None: S["video"] = video(vid, a.out, R)
    et = rd("eta.csv")
    if et is not None: S["eta"] = eta(et, a.out, R)
    rl = rd("ratelimit.csv")
    if rl is not None: S["ratelimit"] = ratelimit(rl, a.out, R)
    tel = rd("telemetry.csv")
    if tel is not None: S["telemetry"] = telemetry(tel, a.out, R)
    wd = rd("watchdog_log.csv")
    if wd is not None: S["watchdog"] = watchdog(wd, tel, a.out, R)
    st = rd("settle.csv")
    if st is not None: S["settle"] = settle(st, a.out, R)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(R))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=float)
    print(f"Selesai → {os.path.abspath(a.out)}/summary.md dan figures/")


if __name__ == "__main__":
    main()
