#!/bin/bash
#SBATCH --job-name=noise_abl
#SBATCH --qos=h200-1day
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/noise_abl.%j.out
#SBATCH --error=logs/noise_abl.%j.err

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
cd ~/thesis/Flowers-GNN
source .venv/bin/activate
mkdir -p outputs

for S in 0.0 0.001 0.005 0.02; do
  echo "=============== noise_std=$S ==============="
  python -u train.py \
      model=flowers_wave \
      data=wave_c_sines \
      train.epochs=10 \
      data.max_train_trajectories=128 \
      data.max_valid_trajectories=32 \
      data.batch_size=2 \
      data.num_workers=4 \
      train.optimizer.lr=1e-3 \
      data.train_noise_std=$S \
      "train.rollout.horizons=[1,5,10,20]" \
      train.rollout.max_trajectories=16 \
      experiment.wandb_mode=disabled
done
