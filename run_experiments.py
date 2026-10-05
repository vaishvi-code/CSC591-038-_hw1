import csv, os, re, subprocess, sys

PY = sys.executable
os.makedirs("results", exist_ok=True)

DATASETS = {
    "blogcatalog": "data/blogcatalog.mat",
    "ppi": "data/Homo_sapiens.mat",
    "wikipedia": "data/POS.mat",
}
SETTINGS = {
    1: ["--small", "--window", "1"],
    10: ["--large", "--window", "10", "--rank", "256"],
}
# Table 3 of the paper, 10% training: (micro, macro)
PAPER = {
    ("blogcatalog", 1): (33.04, 14.86), ("blogcatalog", 10): (38.36, 22.90),
    ("ppi", 1): (16.01, 12.10),         ("ppi", 10): (18.16, 14.32),
    ("wikipedia", 1): (49.90, 9.25),    ("wikipedia", 10): (46.21, 8.38),
}

def run(cmd, log):
    try:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    except Exception:
        if os.path.exists(log):
            os.remove(log)          # don't leave a half-written log behind
        raise

def read_log(path):
    # PowerShell 5.1's Tee-Object writes UTF-16, so handle both encodings
    raw = open(path, "rb").read()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return raw.decode(enc, errors="replace")

def parse(path):
    text = read_log(path)
    ratios = [float(x) for x in re.findall(r"training ratio ([\d.]+)", text)]
    avgs = re.findall(r"Average micro ([\d.]+), Average macro ([\d.]+)", text)
    assert len(ratios) == len(avgs) > 0, f"could not parse {path}"
    return [(round(r * 100), float(a), float(b)) for r, (a, b) in zip(ratios, avgs)]

rows = []
for ds, mat in DATASETS.items():
    for T, flags in SETTINGS.items():
        tag = f"results/{ds}_T{T}"
        if not os.path.exists(tag + ".npy"):
            print(f"[netmf]   {ds} T={T}", flush=True)
            run([PY, "netmf.py", "--input", mat, "--output", tag,
                 "--dim", "128", *flags], tag + ".log")
        if not os.path.exists(tag + "_predict.log"):
            print(f"[predict] {ds} T={T}", flush=True)
            run([PY, "predict.py", "--label", mat, "--embedding", tag + ".npy",
                 "--seed", "0", "--start-train-ratio", "10",
                 "--stop-train-ratio", "90", "--num-train-ratio", "9"],
                tag + "_predict.log")
        for ratio, mi, ma in parse(tag + "_predict.log"):
            rows.append((ds, T, ratio, mi, ma))

with open("results/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "T", "train_ratio_pct", "micro_f1", "macro_f1"])
    w.writerows(rows)

print("\n10% training: yours vs paper (micro / macro)")
for (ds, T), (pm, pM) in PAPER.items():
    mine = [r for r in rows if r[0] == ds and r[1] == T and r[2] == 10][0]
    print(f"{ds:12s} T={T:<2d} yours {mine[3]:6.2f} / {mine[4]:6.2f}   paper {pm:6.2f} / {pM:6.2f}")
print("\nFull sweep saved to results/summary.csv")