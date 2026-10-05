#!/bin/bash
#SBATCH --job-name=netmf_flickr
#SBATCH --output=results/flickr_slurm_%j.log
#SBATCH --error=results/flickr_slurm_%j.err
#SBATCH --time=04:00:00
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

# ── Flickr T=1 (small window) ────────────────────────────────────────────────
if [ -f results/flickr_T1.npy ]; then
    echo "=== Flickr T=1: embedding already exists, skipping ===" | tee results/flickr_T1.log
    T1_SEC=177   # recorded from job 297509 (21:16:02 -> 21:18:59)
else
    echo "=== Flickr T=1 (small) ===" | tee results/flickr_T1.log
    T1_START=$SECONDS

    python3 netmf.py \
        --input  data/flickr.mat \
        --output results/flickr_T1 \
        --small  --window 1 --dim 128 \
        2>&1 | tee -a results/flickr_T1.log

    T1_SEC=$(( SECONDS - T1_START ))
    echo "Flickr T=1 wall-clock: ${T1_SEC}s" | tee -a results/flickr_T1.log
fi

# ── Flickr T=10 (large window) ───────────────────────────────────────────────
echo "=== Flickr T=10 (large) ===" | tee results/flickr_T10.log
T10_START=$SECONDS

python3 netmf.py \
    --input  data/flickr.mat \
    --output results/flickr_T10 \
    --large  --window 10 --rank 256 --dim 128 \
    2>&1 | tee -a results/flickr_T10.log

T10_SEC=$(( SECONDS - T10_START ))
echo "Flickr T=10 wall-clock: ${T10_SEC}s" | tee -a results/flickr_T10.log

# ── Classification ────────────────────────────────────────────────────────────
if [ -f results/flickr_T1.npy ]; then
    echo "=== Predicting T=1 ==="
    python3 predict.py \
        --label data/flickr.mat \
        --embedding results/flickr_T1.npy \
        --seed 0 \
        --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 \
        > results/flickr_T1_predict.log 2>&1
    echo "Prediction T=1 done."
fi

if [ -f results/flickr_T10.npy ]; then
    echo "=== Predicting T=10 ==="
    python3 predict.py \
        --label data/flickr.mat \
        --embedding results/flickr_T10.npy \
        --seed 0 \
        --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 \
        > results/flickr_T10_predict.log 2>&1
    echo "Prediction T=10 done."
fi

echo ""
echo "=== All done ==="
echo "  T=1  wall-clock: ${T1_SEC}s"
echo "  T=10 wall-clock: ${T10_SEC}s"
