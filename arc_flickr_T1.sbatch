#!/bin/bash
#SBATCH --job-name=netmf_flickr_T1
#SBATCH --output=results/flickr_T1_slurm_%j.log
#SBATCH --error=results/flickr_T1_slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

# ── Load modules ─────────────────────────────────────────────────────────────
module load gnu12 openmpi4
module load py3-numpy py3-scipy
export PYTHONPATH="$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

cd ~/NetMF
mkdir -p results

# ── Peak memory tracker ───────────────────────────────────────────────────────
peak_mem_mb() {
    local pid=$1 max=0 rss
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
        [[ $rss -gt $max ]] && max=$rss
        sleep 2
    done
    echo $((max / 1024))
}

# ── NetMF T=1 (small window — direct dense N×N matrix build, ~52GB for Flickr)
# NOTE: This is expected to OOM on most nodes (N=80513, matrix = ~52GB).
#       The OOM itself is the key Task 3 finding: closed-form factorization
#       fails for large graphs. Document the failure in your report.
echo "=== Flickr T=1 (small window) ===" | tee results/flickr_T1.log
echo "WARNING: N=80513, dense matrix requires ~52GB RAM. May OOM." | tee -a results/flickr_T1.log
T_START=$(date +%s%3N)

python3 netmf.py \
    --input  data/flickr.mat \
    --output results/flickr_T1 \
    --small  --window 1 --dim 128 \
    2>&1 | tee -a results/flickr_T1.log &

PID=$!
MEM=$(peak_mem_mb $PID)
wait $PID
EXIT=$?

T_END=$(date +%s%3N)
T_SEC=$(echo "scale=1; ($T_END - $T_START) / 1000" | bc)

if [ $EXIT -ne 0 ]; then
    echo "RESULT: OOM or error (exit code $EXIT) after ${T_SEC}s, peak RAM: ${MEM} MB" | tee -a results/flickr_T1.log
    echo "This confirms that direct matrix factorization is infeasible for N=80513."
    exit 0   # exit 0 so SLURM marks job as complete, not failed
fi

echo "Flickr T=1 wall-clock: ${T_SEC}s, peak RAM: ${MEM} MB" | tee -a results/flickr_T1.log

# ── Classification (only if embedding was produced) ──────────────────────────
if [ -f results/flickr_T1.npy ]; then
    echo "=== Predicting T=1 ===" | tee -a results/flickr_T1.log
    python3 predict.py \
        --label data/flickr.mat \
        --embedding results/flickr_T1.npy \
        --seed 0 \
        --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 \
        > results/flickr_T1_predict.log 2>&1
fi
