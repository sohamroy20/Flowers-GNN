#!/bin/bash
#SBATCH --job-name=debug_mgn
#SBATCH --qos=rtx4090-30min
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug_mgn.%j.out
#SBATCH --error=logs/debug_mgn.%j.err

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

cd ~/thesis/Flowers-GNN
source .venv/bin/activate
mkdir -p outputs
python -u train.py \
    model=mgn \
    train.epochs=2 \
    data.max_train_trajectories=100 \
    data.max_valid_trajectories=10 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.train_in_memory=true \
    train.optimizer.lr=1e-4 \
    train.rollout.max_trajectories=5 \
    experiment.wandb_mode=disabled
