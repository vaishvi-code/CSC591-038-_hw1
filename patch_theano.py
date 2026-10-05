src = open("netmf_original.py").read()

edits = [
    # 1. remove theano imports and config line
    ("import theano\nfrom theano import tensor as T\n", ""),
    ("theano.config.exception_verbosity='high'\n", ""),

    # 2. approximate_deepwalk_matrix (used by --large)
    ("    m = T.matrix()\n"
     "    mmT = T.dot(m, m.T) * (vol/b)\n"
     "    f = theano.function([m], T.log(T.maximum(mmT, 1)))\n"
     "    Y = f(X.astype(theano.config.floatX))\n",
     "    mmT = np.dot(X, X.T) * (vol/b)\n"
     "    Y = np.log(np.maximum(mmT, 1))\n"),

    # 3. direct_compute_deepwalk_matrix (used by --small)
    ("    m = T.matrix()\n"
     "    f = theano.function([m], T.log(T.maximum(m, 1)))\n"
     "    Y = f(M.todense().astype(theano.config.floatX))\n",
     "    Y = np.log(np.maximum(M.toarray(), 1))\n"),
]

for old, new in edits:
    assert src.count(old) == 1, f"expected exactly one match for:\n{old}"
    src = src.replace(old, new)

assert "theano" not in src and "T." not in src.replace("np.", "")
open("netmf.py", "w").write(src)
print("netmf.py patched")