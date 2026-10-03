#!/bin/bash
#SBATCH --job-name=smoke_wave
#SBATCH --qos=h200-6hours
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/smoke_wave.%j.out
#SBATCH --error=logs/smoke_wave.%j.err

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
cd ~/thesis/Flowers-GNN
source .venv/bin/activate
mkdir -p outputs
nvidia-smi

for M in mgn_wave flowers_wave; do
  echo "=============== $M ==============="
  python -u train.py \
      model=$M \
      data=wave_c_sines \
      train.epochs=10 \
      data.max_train_trajectories=128 \
      data.max_valid_trajectories=8 \
      data.batch_size=2 \
      data.num_workers=4 \
      train.optimizer.lr=1e-3 \
      "train.rollout.horizons=[1,5,10,20]" \
      train.rollout.max_trajectories=8 \
      data.train_noise_std=0.001 \
      experiment.wandb_mode=disabled
done
