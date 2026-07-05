"""Precompute normalization statistics for CylinderFlow (fast version).

Computes mean and std over TRAIN split for:
  - velocity   (input)
  - delta      (target = velocity(t+1) - velocity(t))
Both restricted to NORMAL and OUTFLOW nodes (matching the loss mask).

Fast: one trajectory loaded per iteration instead of one frame per iteration.

Output: data/<dataset>/torch/norm_stats.pt

Usage:
    python scripts/compute_norm_stats.py --dataset cylinder_flow
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from flowers_gnn.data.cylinder_dataset import NodeType


def welford_update(count, mean, m2, new_values):
    """Numerically stable running mean/variance via Welford's algorithm.

    Args:
        count:  scalar, running sample count
        mean:   (D,) running mean
        m2:     (D,) running sum of squared deviations from the mean
        new_values: (K, D) batch of new samples

    Returns:
        Updated (count, mean, m2).
    """
    k = new_values.shape[0]
    if k == 0:
        return count, mean, m2

    new_mean = new_values.mean(dim=0)
    new_m2 = ((new_values - new_mean) ** 2).sum(dim=0)

    total = count + k
    delta = new_mean - mean
    combined_mean = mean + delta * (k / total)
    combined_m2 = m2 + new_m2 + (delta ** 2) * (count * k / total)
    return total, combined_mean, combined_m2


def compute_stats(train_dir: Path):
    """Iterate trajectory-by-trajectory, accumulating Welford stats over masked nodes."""
    traj_paths = sorted(train_dir.glob("traj_*.pt"))
    if not traj_paths:
        raise RuntimeError(f"No traj_*.pt files in {train_dir}")

    v_count, v_mean, v_m2 = 0, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    d_count, d_mean, d_m2 = 0, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)

    total_frames = 0
    for p in tqdm(traj_paths, desc="scanning trajectories"):
        blob = torch.load(p, weights_only=True)
        node_type = blob["node_type"][0, :, 0].long()                # (N,)
        mask = (node_type == NodeType.NORMAL) | (node_type == NodeType.OUTFLOW)  # (N,)
        velocity = blob["velocity"].double()                         # (T, N, 2)

        # Only masked nodes
        v_masked = velocity[:, mask, :]                              # (T,   K, 2)
        # All frames' velocities: (T*K, 2)
        v_flat = v_masked.reshape(-1, 2)
        v_count, v_mean, v_m2 = welford_update(v_count, v_mean, v_m2, v_flat)

        # Deltas: shape (T-1, K, 2), flatten to ((T-1)*K, 2)
        d_masked = v_masked[1:] - v_masked[:-1]
        d_flat = d_masked.reshape(-1, 2)
        d_count, d_mean, d_m2 = welford_update(d_count, d_mean, d_m2, d_flat)

        total_frames += velocity.shape[0] - 1

    v_std = torch.sqrt(v_m2 / max(v_count - 1, 1))
    d_std = torch.sqrt(d_m2 / max(d_count - 1, 1))

    return {
        "velocity_mean": v_mean.float(),
        "velocity_std": v_std.float(),
        "delta_mean": d_mean.float(),
        "delta_std": d_std.float(),
        "n_frames": total_frames,
        "n_masked_nodes_velocity_total": v_count,
        "n_masked_nodes_delta_total": d_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cylinder_flow", "airfoil"], required=True)
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args()

    dataset_root = Path(args.data_root) / args.dataset
    train_dir = dataset_root / "torch" / "train"
    out_path = dataset_root / "torch" / "norm_stats.pt"

    print(f"Computing normalization stats for {args.dataset} (train split)")
    stats = compute_stats(train_dir)

    print()
    print("Statistics (over NORMAL + OUTFLOW nodes, train split):")
    print(f"  velocity_mean: [{stats['velocity_mean'][0]:+.6f}, {stats['velocity_mean'][1]:+.6f}]")
    print(f"  velocity_std:  [{stats['velocity_std'][0]:+.6f}, {stats['velocity_std'][1]:+.6f}]")
    print(f"  delta_mean:    [{stats['delta_mean'][0]:+.8f}, {stats['delta_mean'][1]:+.8f}]")
    print(f"  delta_std:     [{stats['delta_std'][0]:+.8f}, {stats['delta_std'][1]:+.8f}]")
    print(f"  n_frames:      {stats['n_frames']}")
    print()

    torch.save(stats, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()