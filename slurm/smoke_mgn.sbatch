#!/bin/bash
#SBATCH --job-name=smoke_mgn
#SBATCH --qos=h200-30min
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_mgn.%j.out
#SBATCH --error=logs/smoke_mgn.%j.err

set -euo pipefail

echo "Job started on $(date)"
nvidia-smi

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

cd ~/thesis/Flowers-GNN
source .venv/bin/activate

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

python train.py \
    model=mgn \
    train.epochs=3 \
    data.max_train_trajectories=50 \
    data.max_valid_trajectories=10 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.train_in_memory=true \
    train.rollout.max_trajectories=2 \
    experiment.wandb_mode=online

echo "Job finished on $(date)"
