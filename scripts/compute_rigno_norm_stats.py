"""Precompute normalization statistics for a RIGNO dataset."""

import argparse
from pathlib import Path

import torch

from flowers_gnn.data.rigno_dataset import RignoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=1024)
    args = ap.parse_args()

    ds = RignoDataset(args.nc_path, split="train", n_train=args.n_train)
    u = ds.u                                  # (n_traj, T, N, F)
    delta = u[:, 1:] - u[:, :-1]

    stats = {
        "velocity_mean": u.reshape(-1, ds.n_fields).mean(0),
        "velocity_std":  u.reshape(-1, ds.n_fields).std(0).clamp_min(1e-8),
        "delta_mean":    delta.reshape(-1, ds.n_fields).mean(0),
        "delta_std":     delta.reshape(-1, ds.n_fields).std(0).clamp_min(1e-8),
        "n_fields":      torch.tensor(ds.n_fields),
    }
    for k, v in stats.items():
        print(f"  {k}: {v.tolist() if v.ndim else v.item()}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
