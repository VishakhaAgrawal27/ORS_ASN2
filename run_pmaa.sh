#!/bin/bash
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode045
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/%u/pmaa_job_%j.log

# ═══════════════════════════════════════════════════════════════
# PMAA Cloud Removal — Belgaum — Ada HPC
# ═══════════════════════════════════════════════════════════════
# /share1 = login node only
# /home2  = login + compute nodes (25 GB quota)
# /scratch = compute nodes only (fast, auto-purged 7 days)
# ═══════════════════════════════════════════════════════════════

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ── 1. Load modules ──────────────────────────────────────────
module load u18/cuda/11.6
module load u18/cudnn/8.4.0-cuda-11.6
module load u18/python/3.10.2

echo "CUDA: $(nvcc --version 2>&1 | tail -1)"
echo "Python: $(python3 --version)"
nvidia-smi

# ── 2. Set up scratch workspace ──────────────────────────────
WORK=/scratch/pmaa_${SLURM_JOB_ID}
mkdir -p $WORK/results

DATA_HOME=/home2/$USER/pmaa_data
REPO_HOME=/home2/$USER/PMAA
SCRIPT_HOME=/home2/$USER/pmaa_ada_inference.py

echo ""
echo "=== Copying data to scratch ==="
cp $DATA_HOME/*.tif $WORK/
echo "Files copied: $(ls $WORK/*.tif | wc -l)"
echo "Size: $(du -sh $WORK/ | cut -f1)"

# Copy repo + script
cp -r $REPO_HOME $WORK/PMAA
cp $SCRIPT_HOME $WORK/
echo "Repo + script copied."

# ── 3. Install packages ──────────────────────────────────────
echo ""
echo "=== Installing packages ==="
pip install --user rasterio tifffile timm==0.6.12 scikit-image scipy matplotlib 2>&1 | tail -3

# ── 4. Run inference ──────────────────────────────────────────
echo ""
echo "=== Starting inference ==="
cd $WORK

python3 pmaa_ada_inference.py \
    --data_dir $WORK \
    --output_dir $WORK/results \
    --repo_dir $WORK/PMAA \
    --device cuda \
    --batch_size 8

# ── 5. Copy results to home ──────────────────────────────────
echo ""
echo "=== Copying results ==="
rm -rf /home2/$USER/pmaa_results
mkdir -p /home2/$USER/pmaa_results
cp $WORK/results/* /home2/$USER/pmaa_results/
ls -lh /home2/$USER/pmaa_results/

# ── 6. Cleanup ────────────────────────────────────────────────
rm -rf $WORK
echo "Scratch cleaned."
echo "=== Job finished at $(date) ==="
echo "Results in: /home2/$USER/pmaa_results/"
