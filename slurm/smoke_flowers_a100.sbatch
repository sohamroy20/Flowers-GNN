#!/bin/bash
#SBATCH --job-name=smoke_flowers
#SBATCH --qos=a100-30min
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_flowers.%j.out
#SBATCH --error=logs/smoke_flowers.%j.err

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
cd ~/thesis/Flowers-GNN
source .venv/bin/activate
mkdir -p outputs

python -u train.py \
    model=flowers_mesh \
    train.epochs=3 \
    data.max_train_trajectories=50 \
    data.max_valid_trajectories=10 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.train_in_memory=true \
    train.rollout.max_trajectories=5 \
    experiment.wandb_mode=disabled

