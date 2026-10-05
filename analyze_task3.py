#!/usr/bin/env python
"""
Task 3: Compile wall-clock time + peak memory across all 4 datasets.
Run AFTER all netmf runs are done (including Flickr from ARC).
    python analyze_task3.py
Produces:
    results/scalability_table.csv
    results/scalability_plot.png
"""

import re, os
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.style.use("seaborn-v0_8-whitegrid")

# ── Dataset metadata ──────────────────────────────────────────────────────
DATASETS = {
    "ppi":         {"nodes": 3890,  "edges": 76584},
    "wikipedia":   {"nodes": 4777,  "edges": 184812},
    "blogcatalog": {"nodes": 10312, "edges": 333983},
}

LOGS = {
    ("ppi",         1):  "results/ppi_T1.log",
    ("ppi",         10): "results/ppi_T10.log",
    ("wikipedia",   1):  "results/wikipedia_T1.log",
    ("wikipedia",   10): "results/wikipedia_T10.log",
    ("blogcatalog", 1):  "results/blogcatalog_T1.log",
    ("blogcatalog", 10): "results/blogcatalog_T10.log",
}

def parse_timing_from_log(path):
    """Extract wall-clock time from log timestamps (start -> 'Save embedding').
    Handles UTF-8, UTF-16 (PowerShell Tee-Object), and latin-1 fallback."""
    if not os.path.exists(path):
        return None, None

    raw = open(path, "rb").read()
    # Detect BOM for UTF-16
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        # Try UTF-8, fallback to latin-1
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")

    # Find all timestamps
    times = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)", text)
    if len(times) < 2:
        return None, None

    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    t_start = datetime.strptime(times[0],  fmt)
    t_end   = datetime.strptime(times[-1], fmt)
    wall    = (t_end - t_start).total_seconds()

    # Parse peak memory if logged by arc_flickr.sbatch
    mem_match = re.search(r"peak RAM:\s*([\d.]+)\s*MB", text)
    mem_mb    = float(mem_match.group(1)) if mem_match else None

    return wall, mem_mb

# ── Collect results ───────────────────────────────────────────────────────
rows = []
for (ds, T), log_path in sorted(LOGS.items()):
    wall, mem_mb = parse_timing_from_log(log_path)
    info = DATASETS[ds]
    rows.append({
        "dataset":  ds,
        "T":        T,
        "nodes":    info["nodes"],
        "edges":    info["edges"],
        "wall_sec": round(wall, 1) if wall is not None else "N/A",
        "peak_ram_mb": round(mem_mb, 0) if mem_mb is not None else "N/A",
    })
    status = f"{wall:.1f}s" if wall else "MISSING LOG"
    mem_s  = f"{mem_mb:.0f} MB" if mem_mb else "not recorded"
    print(f"  {ds:12s} T={T:<2d}  {status:>10}   RAM: {mem_s}")

# ── Write CSV ─────────────────────────────────────────────────────────────
with open("results/scalability_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("\nTable saved -> results/scalability_table.csv")

# ── Plot ──────────────────────────────────────────────────────────────────
node_counts = sorted(set(r["nodes"] for r in rows))
ds_order    = ["ppi", "wikipedia", "blogcatalog"]
colors      = {"1": "#4C72B0", "10": "#DD8452"}
markers     = {"1": "o", "10": "s"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for T in [1, 10]:
    xs, ys_time, ys_mem = [], [], []
    labels = []
    for ds in ds_order:
        r = next((x for x in rows if x["dataset"] == ds and x["T"] == T), None)
        if r and isinstance(r["wall_sec"], float):
            xs.append(r["nodes"])
            ys_time.append(r["wall_sec"])
            labels.append(ds)
        if r and isinstance(r.get("peak_ram_mb"), float):
            ys_mem.append(r["peak_ram_mb"])

    if xs:
        axes[0].plot(xs, ys_time, marker=markers[str(T)], color=colors[str(T)],
                     label=f"T={T}", lw=2, markersize=8)
        for x, y, lbl in zip(xs, ys_time, labels):
            axes[0].annotate(lbl, (x, y), textcoords="offset points",
                             xytext=(4, 4), fontsize=8)
    if ys_mem and len(ys_mem) == len(xs):
        axes[1].plot(xs[:len(ys_mem)], ys_mem, marker=markers[str(T)],
                     color=colors[str(T)], label=f"T={T}", lw=2, markersize=8)

axes[0].set_xlabel("Number of Nodes (N)", fontsize=12)
axes[0].set_ylabel("Wall-Clock Time (seconds)", fontsize=12)
axes[0].set_title("NetMF: Graph Scale vs. Execution Time", fontsize=13)
if axes[0].get_lines(): axes[0].legend()
axes[0].set_xscale("log")
axes[0].set_yscale("log")

axes[1].set_xlabel("Number of Nodes (N)", fontsize=12)
axes[1].set_ylabel("Peak RAM Usage (MB)", fontsize=12)
axes[1].set_title("NetMF: Graph Scale vs. Memory Usage", fontsize=13)
if axes[1].get_lines(): axes[1].legend()
axes[1].set_xscale("log")

fig.tight_layout()
fig.savefig("results/scalability_plot.png", dpi=150)
plt.close(fig)
print("Scalability plot saved -> results/scalability_plot.png")
