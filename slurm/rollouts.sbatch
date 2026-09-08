#!/bin/bash
#SBATCH --job-name=rollouts
#SBATCH --qos=rtx4090-6hours
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rollouts.%j.out
#SBATCH --error=logs/rollouts.%j.err

set -euo pipefail
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
cd ~/thesis/Flowers-GNN
source .venv/bin/activate

python -u scripts/generate_rollouts.py \
    --checkpoint outputs/mgn_cylinder_20260810_085613/checkpoints/epoch_30.pt \
    --model mgn \
    --out artifacts/rollouts/mgn_ep30.npz

python -u scripts/generate_rollouts.py \
    --checkpoint outputs_archive/flowers_v1_prefix_bug/flowers_mesh_cylinder_20260810_085434/checkpoints/epoch_30.pt \
    --model flowers \
    --out artifacts/rollouts/flowers_v1_ep30.npz

echo "done"
