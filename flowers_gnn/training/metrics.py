"""Evaluation metrics: rollout MSE / RMSE at multiple horizons.

Mirrors MGN's cfd_eval.py convention exactly:
  - error = mean over channels of (pred - target)**2  ->  shape (T,)
  - for each horizon h, MSE_h = mean of error[1:h+1]
  - RMSE_h = sqrt(MSE_h)

Averaging is over ALL nodes (not restricted to loss-mask nodes). This is how
MGN reports numbers in the paper, so we match it for direct comparison.
"""

from typing import Dict, List

import torch


def rollout_mse_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizons: List[int],
) -> Dict[str, float]:
    """Compute {mse_h, rmse_h} for each horizon h.

    Args:
        pred:   (T+1, N, 2)  predicted trajectory (includes step 0)
        target: (T+1, N, 2)  ground-truth trajectory (includes step 0)
        horizons: list of ints, e.g. [1, 10, 20, 50, 100, 200]

    Returns:
        dict with keys like 'mse_1_steps', 'rmse_1_steps', 'mse_10_steps', ...
    """
    # Per-frame error: mean over nodes and channels
    err = ((pred - target) ** 2).mean(dim=(1, 2))   # (T+1,)

    out: Dict[str, float] = {}
    max_available = err.shape[0] - 1                # -1 because frame 0 is the init
    for h in horizons:
        if h > max_available:
            continue
        mse = err[1 : h + 1].mean().item()
        out[f"mse_{h}_steps"] = mse
        out[f"rmse_{h}_steps"] = mse ** 0.5
    return out