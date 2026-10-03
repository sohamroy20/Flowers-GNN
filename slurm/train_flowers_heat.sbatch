#!/bin/bash
#SBATCH --job-name=flowers_heat
#SBATCH --qos=h200-1day
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/flowers_heat.%j.out
#SBATCH --error=logs/flowers_heat.%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, started $(date)"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
cd ~/thesis/Flowers-GNN
source .venv/bin/activate
mkdir -p outputs
nvidia-smi

LATEST_CKPT=""
CKPT_DIR=$(ls -td outputs/flowers_heat_v1_heat_l_sines_* 2>/dev/null | head -1 || true)
if [ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR/checkpoints" ]; then
    LATEST_CKPT=$(ls -t "$CKPT_DIR/checkpoints"/epoch_*.pt 2>/dev/null | head -1 || true)
fi
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    RESUME_ARG="resume_from=$LATEST_CKPT"
else
    echo "Starting fresh run (no checkpoint found)"
    RESUME_ARG=""
fi

python -u train.py \
    model=flowers_heat \
    data=heat_l_sines \
    train.epochs=35 \
    data.batch_size=2 \
    data.num_workers=4 \
    train.optimizer.lr=1e-3 \
    data.train_noise_std=0.0018 \
    "train.rollout.horizons=[1,5,10,20]" \
    train.rollout.max_trajectories=32 \
    experiment.wandb_mode=online \
    $RESUME_ARG

echo "Finished $(date)"
