#!/bin/bash
#SBATCH --job-name=mgn_cylinder
#SBATCH --qos=rtx4090-1day
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=logs/mgn.%j.out
#SBATCH --error=logs/mgn.%j.err

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

# Auto resume: we can find the latest checkpoint from any previous mgn_cylinder run
LATEST_CKPT=""
CKPT_DIR=$(ls -td outputs/mgn_cylinder_* 2>/dev/null | head -1 || true)
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
    model=mgn \
    train.epochs=30 \
    data.batch_size=8 \
    data.num_workers=4 \
    data.train_in_memory=true \
    train.optimizer.lr=1e-4 \
    train.rollout.max_trajectories=32 \
    experiment.wandb_mode=online \
    $RESUME_ARG

echo "Job finished on $(date)
