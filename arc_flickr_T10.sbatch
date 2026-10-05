#!/bin/bash
#SBATCH --job-name=netmf_flickr_T10
#SBATCH --output=results/flickr_T10_slurm_%j.log
#SBATCH --error=results/flickr_T10_slurm_%j.err
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

# ── NetMF T=10 (large window, eigenpair approximation) ───────────────────────
echo "=== Flickr T=10 (large window) ===" | tee results/flickr_T10.log
T_START=$(date +%s%3N)

python3 netmf.py \
    --input  data/flickr.mat \
    --output results/flickr_T10 \
    --large  --window 10 --rank 256 --dim 128 \
    2>&1 | tee -a results/flickr_T10.log &

PID=$!
MEM=$(peak_mem_mb $PID)
wait $PID
EXIT=$?

T_END=$(date +%s%3N)
T_SEC=$(echo "scale=1; ($T_END - $T_START) / 1000" | bc)
echo "Flickr T=10 wall-clock: ${T_SEC}s, peak RAM: ${MEM} MB" | tee -a results/flickr_T10.log

if [ $EXIT -ne 0 ]; then
    echo "ERROR: netmf.py exited with code $EXIT" | tee -a results/flickr_T10.log
    exit $EXIT
fi

# ── Classification ────────────────────────────────────────────────────────────
echo "=== Predicting T=10 ===" | tee -a results/flickr_T10.log
python3 predict.py \
    --label data/flickr.mat \
    --embedding results/flickr_T10.npy \
    --seed 0 \
    --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 \
    > results/flickr_T10_predict.log 2>&1

echo "Done. T=10 wall-clock: ${T_SEC}s, peak RAM: ${MEM} MB"
