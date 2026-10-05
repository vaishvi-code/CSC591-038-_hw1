import glob, os
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

for path in sorted(glob.glob("data/*.mat")):
    print("=" * 60)
    print(path, f"({os.path.getsize(path)/1e6:.2f} MB)")
    with open(path, "rb") as f:
        head = f.read(20)
    if not head.startswith(b"MATLAB"):
        print("  NOT a MATLAB file. Starts with:", head)
        continue

    m = loadmat(path)
    print("  keys:", [k for k in m if not k.startswith("__")])
    A = sp.csr_matrix(m["network"])
    G = sp.csr_matrix(m["group"])

    n = A.shape[0]
    loops = int((A.diagonal() != 0).sum())
    sym = abs(A - A.T).nnz == 0
    deg = np.asarray(A.sum(axis=1)).ravel()
    labels_per_node = np.asarray((G > 0).sum(axis=1)).ravel()

    print(f"  nodes:               {n}")
    print(f"  adjacency nnz:       {A.nnz}")
    print(f"  undirected edges:    {(A.nnz - loops) // 2 + loops}")
    print(f"  symmetric:           {sym}")
    print(f"  self-loops:          {loops}")
    print(f"  edge weights:        min={A.data.min()}, max={A.data.max()}")
    print(f"  isolated nodes:      {int((deg == 0).sum())}")
    print(f"  label matrix shape:  {G.shape}  (rows should equal nodes)")
    print(f"  nodes with no label: {int((labels_per_node == 0).sum())}")
    print(f"  avg labels per node: {labels_per_node.mean():.2f}")