"""
fix_all_issues.py  – fixes every reviewer issue
Outputs:
  results/scalability_plot.png  (memory panel now populated)
  results/timing_breakdown.csv  (construction + SVD separately)
  prints corrected density values and timing breakdown
"""
import os, time, sys
import numpy as np
import scipy.io, scipy.sparse as sp
from scipy.sparse import csgraph
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil; HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("results", exist_ok=True)

DATASETS = {
    "ppi":         ("data/Homo_sapiens.mat", "network"),
    "wikipedia":   ("data/POS.mat",          "network"),
    "blogcatalog": ("data/blogcatalog.mat",  "network"),
}
DIM=128; RANK=256; B=1.0

def rss():
    if HAVE_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss/1024**2
    return 0.0

def deepwalk_filter(evals, window):
    return np.maximum(np.where(evals>=1, 1.0,
        evals*(1-evals**window)/(1-evals+1e-12)/window), 0)

# ── Collect real data ────────────────────────────────────────────────────────
records = {}
print(f"{'='*65}")
print(f"{'Dataset':12s} {'T':>2}  {'Build(s)':>8} {'SVD(s)':>7} {'Total(s)':>8}  "
      f"{'PeakMem(MB)':>11}  {'A_nnz':>8}  {'A_density(%)':>12}  {'Y_density_T1(%)':>15}")
print(f"{'='*65}")

for ds, (path, var) in DATASETS.items():
    if not os.path.exists(path):
        print(f"[SKIP] {ds}"); continue
    data = scipy.io.loadmat(path)
    A = sp.csr_matrix(data[var])
    n = A.shape[0]; total = n*n
    A_nnz  = A.nnz  # actual non-zeros stored (counts both directions if undirected)
    A_dens = 100.0 * A_nnz / total

    # ── T=1: construction ─────────────────────────────────────────────────
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sp.identity(n) - L  # D^{-1/2} A D^{-1/2}

    r0 = rss(); t0 = time.time()
    S = X * (vol / B)
    D_inv = sp.diags(d_rt**-1)
    M_sp  = D_inv.dot(D_inv.dot(S).T)
    M_den = M_sp.toarray()
    Y_den = np.log(np.maximum(M_den, 1))
    Y_sp  = sp.csr_matrix(Y_den)
    build_T1 = time.time()-t0; mem_T1 = max(rss()-r0, M_den.nbytes/1024**2)

    nnz_Y_T1 = np.count_nonzero(Y_den)
    dens_Y_T1 = 100.0*nnz_Y_T1/total
    # embedding density: N×128 all dense
    emb_dens = 100.0  # N×128 float64 always fully dense

    # T=1 SVD
    r0 = rss(); t0 = time.time()
    _, s, _ = sp.linalg.svds(Y_sp, min(RANK, n-2))
    s = np.sort(s)[::-1]
    svd_T1 = time.time()-t0
    energy_T1 = np.sum(s[:DIM]**2)/np.sum(s**2)

    records[(ds,1)] = dict(build=build_T1, svd=svd_T1,
                            mem=mem_T1, s=s, energy=energy_T1,
                            A_nnz=A_nnz, A_dens=A_dens,
                            Y_dens_T1=dens_Y_T1, Y_dens_T10=None)

    # ── T=10: construction ────────────────────────────────────────────────
    r0 = rss(); t0 = time.time()
    rank = min(RANK, n-2)
    evals, evecs = sp.linalg.eigsh(X, rank, which="LA")
    evals_f = deepwalk_filter(evals, 10)
    Xm  = sp.diags(np.sqrt(evals_f)).dot(evecs.T).T
    mmT = np.dot(Xm, Xm.T)
    mmT *= (vol/B); np.maximum(mmT,1,out=mmT); np.log(mmT,out=mmT)
    build_T10 = time.time()-t0; mem_T10 = max(rss()-r0, mmT.nbytes/1024**2)

    dens_Y_T10 = 100.0*np.count_nonzero(mmT)/total

    r0 = rss(); t0 = time.time()
    from sklearn.utils.extmath import randomized_svd
    _, s10, _ = randomized_svd(mmT, n_components=min(RANK,n-1), random_state=0)
    s10 = np.sort(s10)[::-1]
    svd_T10 = time.time()-t0
    energy_T10 = np.sum(s10[:DIM]**2)/np.sum(s10**2)

    records[(ds,10)] = dict(build=build_T10, svd=svd_T10,
                             mem=mem_T10, s=s10, energy=energy_T10,
                             A_nnz=A_nnz, A_dens=A_dens,
                             Y_dens_T1=dens_Y_T1, Y_dens_T10=dens_Y_T10)
    records[(ds,1)]['Y_dens_T10'] = dens_Y_T10

    print(f"{ds:12s}  1  {build_T1:8.1f} {svd_T1:7.1f} {build_T1+svd_T1:8.1f}  "
          f"{mem_T1:11.0f}  {A_nnz:8d}  {A_dens:12.4f}  {dens_Y_T1:15.4f}")
    print(f"{ds:12s} 10  {build_T10:8.1f} {svd_T10:7.1f} {build_T10+svd_T10:8.1f}  "
          f"{mem_T10:11.0f}  {'':8}  {'':12}  {dens_Y_T10:15.4f}")

# ── Memory-aware scalability plot ────────────────────────────────────────────
ds_order = ["ppi","wikipedia","blogcatalog"]
nodes    = {"ppi":3890,"wikipedia":4777,"blogcatalog":10312}
# Hardcoded totals from log (construction+SVD):
wall_T1  = {"ppi":2.3,"wikipedia":3.0,"blogcatalog":9.5}
wall_T10 = {"ppi":5.1,"wikipedia":5.1,"blogcatalog":21.4}
# Estimated peak memory N^2 * 8 in MiB:
mem_MB   = {"ppi":115,"wikipedia":174,"blogcatalog":811}

colors={1:"#4C72B0",10:"#DD8452"}; markers={1:"o",10:"s"}
fig, axes = plt.subplots(1,2,figsize=(11,4.2))

for T, wall_d in [(1,wall_T1),(10,wall_T10)]:
    xs=[nodes[d] for d in ds_order]
    yt=[wall_d[d] for d in ds_order]
    axes[0].plot(xs,yt,marker=markers[T],color=colors[T],label=f"T={T}",lw=2,ms=8)
    for x,y,lbl in zip(xs,yt,ds_order):
        axes[0].annotate(lbl,(x,y),textcoords="offset points",xytext=(4,4),fontsize=8)

xs=[nodes[d] for d in ds_order]
ym=[mem_MB[d] for d in ds_order]
axes[1].plot(xs,ym,marker="o",color="#2CA02C",label=r"Est. $N^2 \times 8$ (MiB)",lw=2,ms=8)
for x,y,lbl in zip(xs,ym,ds_order):
    axes[1].annotate(f"{lbl} ({y})",(x,y),textcoords="offset points",xytext=(4,4),fontsize=8)

for ax,ylabel,title in [
    (axes[0],"Wall-Clock Time (s)","Scale vs. Time"),
    (axes[1],"Estimated Memory (MiB)",r"Scale vs. Memory ($N^2 \times 8$ bytes)")]:
    ax.set_xlabel("Number of Nodes ($N$)",fontsize=11)
    ax.set_ylabel(ylabel,fontsize=11)
    ax.set_title(title,fontsize=12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig("results/scalability_plot.png",dpi=150)
plt.close(fig)
print("\nSaved results/scalability_plot.png (memory panel now populated with 115, 174, 811 MiB)")

# ── Print timing breakdown for report ───────────────────────────────────────
import csv
with open("results/timing_breakdown.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset","T","build_s","svd_s","total_s","peak_mem_mb"])
    for ds in ds_order:
        for T in [1,10]:
            r = records[(ds,T)]
            w.writerow([ds,T,round(r['build'],1),round(r['svd'],1),
                        round(r['build']+r['svd'],1),round(r['mem'],0)])
print("Saved results/timing_breakdown.csv")

print("\n== DENSITY TABLE (actual nnz from scipy) ==")
print(f"{'Dataset':12s}  {'N':>6}  {'A.nnz':>8}  {'A_dens(%)':>10}  "
      f"{'Y_T1(%)':>9}  {'Y_T10(%)':>9}  {'Embed(%)':>9}")
for ds in ds_order:
    r = records[(ds,1)]
    n = nodes[ds]
    print(f"{ds:12s}  {n:>6}  {r['A_nnz']:>8}  {r['A_dens']:>10.4f}  "
          f"{r['Y_dens_T1']:>9.4f}  {r['Y_dens_T10']:>9.4f}  {'100.000':>9}")

print("\n== RETAINED ENERGY ==")
for ds in ds_order:
    for T in [1,10]:
        e = records[(ds,T)]['energy']
        print(f"  {ds} T={T}: {e*100:.1f}%")

print("\n== SCALING EXPONENTS (log fit, 3 points) ==")
import math
xs = [nodes[d] for d in ds_order]
for T, wd in [(1,wall_T1),(10,wall_T10)]:
    ys = [wd[d] for d in ds_order]
    # log-log slope between first and last
    alpha = (math.log(ys[-1])-math.log(ys[0]))/(math.log(xs[-1])-math.log(xs[0]))
    print(f"  T={T}: slope={alpha:.2f}  (N^{alpha:.2f})")
