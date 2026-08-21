#!/bin/bash
#SBATCH --job-name=flowers_cylinder
#SBATCH --qos=h200-1day
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=logs/flowers.%j.out
#SBATCH --error=logs/flowers.%j.err

set -euo pipefail

echo "Job started on $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

cd ~/thesis/Flowers-GNN
source .venv/bin/activate

mkdir -p outputs

echo "GPU info:"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# Auto-resume: finding the latest checkpoint from any previous flowers_mesh_cylinder run
LATEST_CKPT=""
CKPT_DIR=$(ls -td outputs/flowers_mesh_cylinder_* 2>/dev/null | head -1 || true)
if [ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR/checkpoints" ]; then
    LATEST_CKPT=$(ls -t "$CKPT_DIR/checkpoints"/epoch_*.pt 2>/dev/null | head -1 || true)
fi

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="resume_from=$LATEST_CKPT"
else
    echo "Starting fresh run (no checkpoint found)"
    RESUME_ARG=""
fi

python -u train.py \
    model=flowers_mesh \
    train.epochs=30 \
    data.batch_size=8 \
    data.num_workers=8 \
    data.train_in_memory=true \
    train.optimizer.lr=1e-4 \
    train.rollout.max_trajectories=20 \
    train.rollout.eval_every_n_epochs=1 \
    experiment.wandb_mode=online \
    $RESUME_ARG

echo "Job finished on $(date)"
