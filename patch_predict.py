import re

src = open("predict_original.py").read()

# Fix 1: np.int was removed in NumPy 1.24
src, n = re.subn(r"\bnp\.int\b", "int", src)
assert n == 3, f"expected 3 np.int uses, found {n}"

# Fix 2: multi_class was removed from LogisticRegression in scikit-learn 1.8
# (redundant anyway: liblinear is one-vs-rest, and it is wrapped in OneVsRestClassifier)
src, n = re.subn(r',\s*multi_class="ovr"', "", src)
assert n == 1, f"expected 1 multi_class argument, found {n}"

open("predict.py", "w").write(src)
print("predict.py patched (np.int -> int, removed multi_class)")