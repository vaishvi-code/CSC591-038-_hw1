#!/usr/bin/env python
"""
Task 2: Sparsity analysis + singular value spectrum for each dataset.
Run from the NetMF root directory:
    python analyze_task2.py
Produces:
    results/sparsity_table.csv
    results/svd_spectrum_<dataset>_T<window>.png
"""

import numpy as np
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import csv, os, time

plt.style.use("seaborn-v0_8-whitegrid")

DATASETS = {
    "blogcatalog": "data/blogcatalog.mat",
    "ppi":         "data/Homo_sapiens.mat",
    "wikipedia":   "data/POS.mat",
    "flickr":      "data/flickr.mat",      # skip if not present
}
DIM   = 128   # embedding dim used in paper
RANK  = 256   # eigenpairs for large-window approx

os.makedirs("results", exist_ok=True)

rows = []   # for sparsity_table.csv

for ds, mat_path in DATASETS.items():
    if not os.path.exists(mat_path):
        print(f"[SKIP] {ds}: {mat_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"Dataset: {ds}")
    data = scipy.io.loadmat(mat_path)
    A = sparse.csr_matrix(data["network"])
    n = A.shape[0]
    total_entries = n * n

    # ── Adjacency sparsity ────────────────────────────────────────────
    A_nnz = A.nnz
    A_sparsity = 100.0 * A_nnz / total_entries
    print(f"  Nodes N={n},  A.nnz={A_nnz},  sparsity={A_sparsity:.4f}%")

    # ── Build NetMF matrix (small window T=1 for memory) ─────────────
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L        # D^{-1/2} A D^{-1/2}

    print(f"  Building NetMF matrix (T=1, direct)...")
    t0 = time.time()
    # Matrix power sum (T=1 → just X itself)
    S = X.copy()
    S = S * (vol / 1.0 / 1.0)        # vol/T/b, b=1
    D_rt_inv = sparse.diags(d_rt ** -1)
    M_sparse = D_rt_inv.dot(D_rt_inv.dot(S).T)
    M_dense  = M_sparse.toarray() if n <= 20000 else None

    if M_dense is not None:
        M_log = np.log(np.maximum(M_dense, 1))
        netmf_nnz      = np.count_nonzero(M_log)
        netmf_sparsity = 100.0 * netmf_nnz / total_entries
    else:
        # For very large graphs just report approximate
        netmf_nnz      = "N/A (too large)"
        netmf_sparsity = "N/A"

    build_time = time.time() - t0
    print(f"  NetMF matrix build: {build_time:.1f}s")
    print(f"  NetMF nnz={netmf_nnz}, sparsity={netmf_sparsity}")

    # ── SVD on NetMF matrix (sparse version for spectrum) ────────────
    # Use the sparse approximate pipeline (large-window style) for spectrum
    print(f"  Computing top-{RANK} eigenpairs for spectrum...")
    t0 = time.time()
    evals, evecs = sparse.linalg.eigsh(X, min(RANK, n - 2), which="LA")
    eig_time = time.time() - t0
    print(f"  Eigsh done in {eig_time:.1f}s")

    # Filter eigenvalues (deepwalk_filter, T=1)
    window = 1
    evals_filt = np.array([
        1.0 if ev >= 1 else ev * (1 - ev**window) / (1 - ev) / window
        for ev in evals
    ])
    evals_filt = np.maximum(evals_filt, 0)

    Xmat = sparse.diags(np.sqrt(evals_filt)).dot(evecs.T).T
    # Approximate singular values of NetMF via svds on the rank-RANK approx
    # (full SVD only for smaller graphs)
    if n <= 20000 and M_dense is not None:
        M_log_sp = sparse.csr_matrix(M_log)
        k = min(RANK, n - 2)
        u, s, v = sparse.linalg.svds(M_log_sp, k)
        s_sorted = np.sort(s)[::-1]
    else:
        # Approximation: eigenvalues of X̂X̂ᵀ ≈ singular values of NetMF
        s_sorted = np.sort(evals_filt)[::-1]

    # ── Plot singular value spectrum ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(s_sorted) + 1), s_sorted, lw=1.5, color="#4C72B0")
    ax.axvline(DIM, color="#DD8452", linestyle="--", lw=1.2,
               label=f"d={DIM} (embedding dim)")
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Singular Value", fontsize=12)
    ax.set_title(f"{ds} — NetMF Singular Value Spectrum (T=1)", fontsize=13)
    ax.legend()
    fig.tight_layout()
    out_fig = f"results/svd_spectrum_{ds}_T1.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"  Saved spectrum plot -> {out_fig}")

    rows.append({
        "dataset":         ds,
        "nodes_N":         n,
        "A_nnz":           A_nnz,
        "A_density_pct":   round(A_sparsity, 6),
        "netmf_nnz":       netmf_nnz,
        "netmf_density_pct": netmf_sparsity if isinstance(netmf_sparsity, str) else round(netmf_sparsity, 2),
    })

# ── Write sparsity table ──────────────────────────────────────────────────
with open("results/sparsity_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\n\nSparsity table saved to results/sparsity_table.csv")  # no unicode needed
print("\nSummary:")
for r in rows:
    print(f"  {r['dataset']:12s}  N={r['nodes_N']:6d}  "
          f"A_density={r['A_density_pct']}%  "
          f"NetMF_density={r['netmf_density_pct']}%")
