import subprocess, sys
import numpy as np

PY = sys.executable
for tag in ("a", "b"):
    subprocess.run([PY, "netmf.py", "--input", "data/POS.mat",
                    "--output", f"results/wiki_det_{tag}", "--small",
                    "--window", "1", "--dim", "128"],
                   check=True, capture_output=True)

A = np.load("results/wiki_det_a.npy")
B = np.load("results/wiki_det_b.npy")

# Gram matrices are invariant to the sign flips SVD is allowed to make
diff = np.abs(A @ A.T - B @ B.T).max()
scale = np.abs(A @ A.T).max()
print(f"max |Gram_a - Gram_b| = {diff:.3e}  (relative to max entry: {diff/scale:.3e})")
print("identical" if diff < 1e-8 * scale else "DIFFERENT between runs")