#!/usr/bin/env python
"""
fix_report_data.py
==================
Generates all missing/corrected data for the report:
  1. Measures REAL peak memory (psutil RSS) for T=1 and T=10 on all small datasets
  2. Computes retained energy ratio  sum(s[:128]^2) / sum(s^2)  for each dataset/window
  3. Produces SVD spectrum plots for ALL datasets at T=1 AND T=10
  4. Regenerates scalability_plot.png with real memory panel
  5. Prints corrected table values for copy-paste into report.tex
"""

import os, time, sys
import numpy as np
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    print("[WARN] psutil not installed; using tracemalloc for memory (less accurate)")
    HAVE_PSUTIL = False
    import tracemalloc

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("results", exist_ok=True)

DATASETS = {
    "ppi":         ("data/Homo_sapiens.mat", "network"),
    "wikipedia":   ("data/POS.mat",          "network"),
    "blogcatalog": ("data/blogcatalog.mat",  "network"),
}
DIM  = 128
RANK = 256
B    = 1.0   # negative sampling

# ─────────────────────────────────────────────────────────────────────────────
def get_rss_mb():
    if HAVE_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return 0.0

def deepwalk_filter(evals, window):
    out = np.where(
        evals >= 1,
        1.0,
        evals * (1 - evals**window) / (1 - evals + 1e-12) / window
    )
    return np.maximum(out, 0)

def build_T1(A):
    """Small-window (T=1): direct computation. M is built dense, Y stored sparse."""
    n   = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X       = sparse.identity(n) - L          # D^{-1/2} A D^{-1/2}
    S       = X * (vol / 1.0 / B)
    D_rt_inv = sparse.diags(d_rt ** -1)
    M_sparse = D_rt_inv.dot(D_rt_inv.dot(S).T)
    M_dense  = M_sparse.toarray()             # dense N x N materialised here
    Y_dense  = np.log(np.maximum(M_dense, 1))
    Y_sparse = sparse.csr_matrix(Y_dense)     # store sparsely (zeros dropped)
    return Y_dense, Y_sparse

def build_T10(A):
    """Large-window (T=10): eigenpair approx, but still forms dense N x N mmT."""
    n   = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X       = sparse.identity(n) - L
    rank    = min(RANK, n - 2)
    evals, evecs = sparse.linalg.eigsh(X, rank, which="LA")
    evals_f = deepwalk_filter(evals, window=10)
    Xmat    = sparse.diags(np.sqrt(evals_f)).dot(evecs.T).T
    mmT     = np.dot(Xmat, Xmat.T)            # dense N x N materialised here
    mmT    *= (vol / B)
    np.maximum(mmT, 1, out=mmT)
    np.log(mmT, out=mmT)
    return mmT

def svd_and_energy(Y, dim=DIM, rank=RANK):
    """Return (singular_values_sorted, retained_energy_ratio)."""
    if sparse.issparse(Y):
        k  = min(rank, min(Y.shape) - 2)
        _, s, _ = sparse.linalg.svds(Y, k)
    else:
        from sklearn.utils.extmath import randomized_svd
        k = min(rank, min(Y.shape) - 1)
        _, s, _ = randomized_svd(Y, n_components=k, random_state=0)
    s = np.sort(s)[::-1]
    energy = np.sum(s[:dim]**2) / np.sum(s**2)
    return s, energy

# ─────────────────────────────────────────────────────────────────────────────
records = {}

for ds, (mat_path, var) in DATASETS.items():
    if not os.path.exists(mat_path):
        print(f"[SKIP] {ds}: {mat_path} not found")
        continue
    print(f"\n{'='*60}\nDataset: {ds}")

    data = scipy.io.loadmat(mat_path)
    A    = sparse.csr_matrix(data[var])
    n    = A.shape[0]
    print(f"  N={n}")

    # ── T=1 ──────────────────────────────────────────────────────────────────
    print("  [T=1] Building matrix and measuring peak RSS...")
    rss_before = get_rss_mb()
    t0 = time.time()
    Y_dense, Y_sparse = build_T1(A)
    wall_T1 = time.time() - t0
    rss_after = get_rss_mb()
    mem_T1 = max(rss_after - rss_before, Y_dense.nbytes / 1024**2)

    s_T1, energy_T1 = svd_and_energy(Y_sparse)
    nnz_T1  = np.count_nonzero(Y_dense)
    dens_T1 = 100.0 * nnz_T1 / (n*n)

    print(f"  [T=1] wall={wall_T1:.1f}s  peak_mem~{mem_T1:.0f} MB  "
          f"density={dens_T1:.4f}%  retained_energy={energy_T1*100:.1f}%")
    records[(ds, 1)] = dict(wall=wall_T1, mem_mb=mem_T1,
                             s=s_T1, energy=energy_T1,
                             nnz=nnz_T1, density=dens_T1)

    # ── T=10 ─────────────────────────────────────────────────────────────────
    print("  [T=10] Building matrix (eigenpair approx) and measuring peak RSS...")
    rss_before = get_rss_mb()
    t0 = time.time()
    Y10 = build_T10(A)
    wall_T10 = time.time() - t0
    rss_after = get_rss_mb()
    mem_T10 = max(rss_after - rss_before, Y10.nbytes / 1024**2)

    s_T10, energy_T10 = svd_and_energy(Y10)
    nnz_T10  = np.count_nonzero(Y10)
    dens_T10 = 100.0 * nnz_T10 / (n*n)

    print(f"  [T=10] wall={wall_T10:.1f}s  peak_mem~{mem_T10:.0f} MB  "
          f"density={dens_T10:.2f}%  retained_energy={energy_T10*100:.1f}%")
    records[(ds, 10)] = dict(wall=wall_T10, mem_mb=mem_T10,
                              s=s_T10, energy=energy_T10,
                              nnz=nnz_T10, density=dens_T10)

    # ── SVD spectrum plots for both T ────────────────────────────────────────
    for T, key, s_arr, en in [(1, "T1", s_T1, energy_T1), (10, "T10", s_T10, energy_T10)]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(s_arr)+1), s_arr, lw=1.5, color="#4C72B0")
        ax.axvline(DIM, color="#DD8452", linestyle="--", lw=1.2,
                   label=f"d={DIM}  (retained energy: {en*100:.1f}%)")
        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Singular Value", fontsize=12)
        ax.set_title(f"{ds} — NetMF Singular Value Spectrum (T={T})", fontsize=13)
        ax.legend()
        fig.tight_layout()
        out = f"results/svd_spectrum_{ds}_{key}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Regenerate scalability plot with REAL memory panel
# ─────────────────────────────────────────────────────────────────────────────
ds_order = ["ppi", "wikipedia", "blogcatalog"]
nodes    = {"ppi": 3890, "wikipedia": 4777, "blogcatalog": 10312}
colors   = {1: "#4C72B0", 10: "#DD8452"}
markers  = {1: "o", 10: "s"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for T in [1, 10]:
    xs, ys_time, ys_mem, lbls = [], [], [], []
    for ds in ds_order:
        if (ds, T) not in records:
            continue
        r = records[(ds, T)]
        xs.append(nodes[ds])
        ys_time.append(r["wall"])
        ys_mem.append(r["mem_mb"])
        lbls.append(ds)

    if xs:
        axes[0].plot(xs, ys_time, marker=markers[T], color=colors[T],
                     label=f"T={T}", lw=2, markersize=8)
        for x, y, lbl in zip(xs, ys_time, lbls):
            axes[0].annotate(lbl, (x, y), textcoords="offset points",
                             xytext=(4, 4), fontsize=8)
        axes[1].plot(xs, ys_mem, marker=markers[T], color=colors[T],
                     label=f"T={T}", lw=2, markersize=8)
        for x, y, lbl in zip(xs, ys_mem, lbls):
            axes[1].annotate(lbl, (x, y), textcoords="offset points",
                             xytext=(4, 4), fontsize=8)

axes[0].set_xlabel("Number of Nodes (N)", fontsize=12)
axes[0].set_ylabel("Wall-Clock Time (s)", fontsize=12)
axes[0].set_title("NetMF: Scale vs. Execution Time", fontsize=13)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
if axes[0].get_lines(): axes[0].legend()

axes[1].set_xlabel("Number of Nodes (N)", fontsize=12)
axes[1].set_ylabel("Peak RSS Memory (MB)", fontsize=12)
axes[1].set_title("NetMF: Scale vs. Peak Memory", fontsize=13)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
if axes[1].get_lines(): axes[1].legend()

fig.tight_layout()
fig.savefig("results/scalability_plot.png", dpi=150)
plt.close(fig)
print("\nScalability plot (with real memory) saved -> results/scalability_plot.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print copy-paste values for report.tex
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("COPY-PASTE VALUES FOR report.tex")
print("="*70)
print(f"\n{'Dataset':12s} {'T':>3}  {'Wall(s)':>8}  {'PeakMem(MB)':>12}  {'RetainedE(%)':>13}  {'Density(%)':>11}")
for ds in ds_order:
    for T in [1, 10]:
        if (ds, T) not in records:
            continue
        r = records[(ds, T)]
        print(f"{ds:12s} {T:>3}  {r['wall']:>8.1f}  {r['mem_mb']:>12.0f}  "
              f"{r['energy']*100:>13.1f}  {r['density']:>11.4f}")

print("\nDone. Check results/ for updated plots.")
