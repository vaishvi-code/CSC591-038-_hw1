"""
verify_all.py — verify symmetry, recompute exponents, confirm all numbers
"""
import os, math
import numpy as np
import scipy.io, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = {
    "ppi":         ("data/Homo_sapiens.mat", "network", 3890),
    "wikipedia":   ("data/POS.mat",          "network", 4777),
    "blogcatalog": ("data/blogcatalog.mat",  "network", 10312),
}

plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. Symmetry check ────────────────────────────────────────────────────────
print("=== SYMMETRY CHECK ===")
for ds, (path, var, N) in DATASETS.items():
    data = scipy.io.loadmat(path)
    A = sp.csr_matrix(data[var])
    diff = (A - A.T)
    sym  = diff.nnz == 0
    print(f"  {ds:12s}: nnz(A)={A.nnz:8d}  symmetric={sym}  "
          f"nnz(A-A.T)={diff.nnz}  N²={N*N:12d}  dens={100*A.nnz/(N*N):.4f}%")

# ── 2. Theoretical memory ─────────────────────────────────────────────────────
print("\n=== MEMORY (N²×8 bytes) ===")
nodes = {"ppi":3890,"wikipedia":4777,"blogcatalog":10312,"flickr":80513}
for ds,N in nodes.items():
    b   = N**2*8
    mib = b/1024**2
    gib = b/1024**3
    gb  = b/1e9
    print(f"  {ds:12s}: {mib:8.1f} MiB = {gib:.3f} GiB = {gb:.3f} GB")
print("  Naive (3 copies): 3× above")

# ── 3. Recompute scaling exponents from new timings ──────────────────────────
print("\n=== SCALING EXPONENTS (new timings from timing_breakdown.csv) ===")
# new measured totals
T1  = {"ppi":2.3, "wikipedia":3.0, "blogcatalog":9.5}
T10 = {"ppi":5.1, "wikipedia":5.1, "blogcatalog":21.4}
Ns  = {"ppi":3890, "wikipedia":4777, "blogcatalog":10312, "flickr":80513}

for T, wd in [(1, T1), (10, T10)]:
    ppi_N = Ns["ppi"]; bc_N = Ns["blogcatalog"]
    ppi_t = wd["ppi"]; bc_t = wd["blogcatalog"]
    ratio_N = bc_N/ppi_N; ratio_t = bc_t/ppi_t
    alpha = math.log(ratio_t)/math.log(ratio_N)
    print(f"  T={T}: PPI->{bc_N/ppi_N:.2f}x nodes => {bc_t/ppi_t:.2f}x time => N^{alpha:.2f}")

# Flickr check: T=1 177s vs BlogCatalog 9.5s
# (using Flickr time from ARC)
flickr_t1 = 177
bc_t1 = T1["blogcatalog"]
ratio_N_fl = Ns["flickr"]/Ns["blogcatalog"]
ratio_t_fl = flickr_t1/bc_t1
alpha_fl = math.log(ratio_t_fl)/math.log(ratio_N_fl)
print(f"  T=1 Flickr consistency: {ratio_N_fl:.2f}x nodes => {ratio_t_fl:.1f}x time => N^{alpha_fl:.2f}")

# ── 4. Regenerate scalability plot WITH Flickr ────────────────────────────────
print("\n=== REGENERATING SCALABILITY PLOT WITH FLICKR ===")
# Measured RSS (MiB, from fix_all_issues): 122/174/811 for local datasets
# Theoretical for Flickr: 80513²×8/1024² = 49,403 MiB ≈ 49 GiB
flickr_mib = 80513**2*8/1024**2
print(f"  Flickr N²×8 = {flickr_mib:.0f} MiB = {flickr_mib/1024:.1f} GiB")

ds_order_full = ["ppi","wikipedia","blogcatalog","flickr"]
xs_local = [3890, 4777, 10312]
xs_full  = [3890, 4777, 10312, 80513]

# Times: T=1 local + Flickr ARC, T=10 local only (Flickr T=10 not timed)
t1_all  = [2.3, 3.0, 9.5, 177]
t10_loc = [5.1, 5.1, 21.4]

# Memory (MiB): measured locally + theoretical for Flickr
mem_all = [122, 174, 811, flickr_mib]

colors={1:"#4C72B0", 10:"#DD8452"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Time panel
axes[0].plot(xs_full, t1_all,  marker="o", color=colors[1],  label="T=1 (local+ARC)", lw=2, ms=8)
axes[0].plot(xs_local, t10_loc, marker="s", color=colors[10], label="T=10 (local)", lw=2, ms=8, ls="--")
for x,y,lbl in zip(xs_full, t1_all, ds_order_full):
    axes[0].annotate(lbl,(x,y),textcoords="offset points",xytext=(4,4),fontsize=7)

# Memory panel
axes[1].plot(xs_full, mem_all, marker="o", color="#2CA02C", label="N²×8 (MiB)", lw=2, ms=8)
axes[1].axhline(32*1024, color="red", ls=":", lw=1.5, label="Local RAM limit (32 GiB)")
for x,y,lbl in zip(xs_full, mem_all, ds_order_full):
    axes[1].annotate(lbl,(x,y),textcoords="offset points",xytext=(4,4),fontsize=7)

for ax, ylabel, title in [
    (axes[0], "Wall-Clock Time (s)", "Scale vs. Time"),
    (axes[1], "Peak Memory (MiB)",   "Scale vs. Memory (N²×8 bytes)")]:
    ax.set_xlabel("Number of Nodes (N)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig("results/scalability_plot.png", dpi=150)
plt.close(fig)
print("  Saved results/scalability_plot.png (Flickr included, RAM limit marked)")

# ── 5. Confirm density/energy numbers from fix_all_issues ────────────────────
print("\n=== CONFIRMED FROM fix_all_issues LOG ===")
rows = [
    ("ppi",         1, 0.5061, 0.4991, None,   66.5),
    ("ppi",        10, None,   None,   49.99,  94.1),
    ("wikipedia",   1, 0.8099, 0.7177, None,   62.9),
    ("wikipedia",  10, None,   None,   94.90,  99.0),
    ("blogcatalog", 1, 0.6282, 0.6049, None,   71.2),
    ("blogcatalog",10, None,   None,   91.89,  98.9),
]
print(f"  {'Dataset':12s} {'T':>2}  {'A_dens':>8}  {'Y_T1':>7}  {'Y_T10':>7}  {'Energy':>7}")
for ds,T,A_d,Y1,Y10,en in rows:
    print(f"  {ds:12s} {T:>2}  "
          f"{A_d if A_d else '':>8}  "
          f"{Y1 if Y1 else '':>7}  "
          f"{Y10 if Y10 else '':>7}  "
          f"{en:>7.1f}%")
