"""Convert MGN TFRecord trajectories to per-trajectory torch .pt files.

One-time conversion. Run once per dataset split.
Usage:
    python scripts/convert_tfrecord_to_torch.py --dataset cylinder_flow --split train
    python scripts/convert_tfrecord_to_torch.py --dataset cylinder_flow --split valid
    python scripts/convert_tfrecord_to_torch.py --dataset cylinder_flow --split test

After all three splits run, we can ignore the TFRecord files for training.
"""

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm


def parse_trajectory(proto, meta):
    """Parse one trajectory from a tf.Example record (mirrors MGN's dataset.py:_parse)."""
    feature_lists = {name: tf.io.VarLenFeature(tf.string) for name in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)

    out = {}
    for key, field in meta["features"].items():
        raw = features[key].values
        data = tf.io.decode_raw(raw, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        # Note: we deliberately DO NOT tile static fields here.
        # Static = stored once, applies at every timestep. Tiling 600x wastes memory.
        # The dataset class will handle the "static vs dynamic" semantics.
        out[key] = data
    return out


def trajectory_to_torch(traj, meta):
    """Convert one parsed trajectory (TF tensors) to a dict of torch tensors.

    Compresses static fields to their leading-1 form:
        mesh_pos:  (1, N, 2)  not (T, N, 2)
        cells:     (1, M, 3)
        node_type: (1, N, 1)
        velocity:  (T, N, 2)   (dynamic, kept full)
        pressure:  (T, N, 1)   (dynamic, kept full)  -- ignored for cylinder model but stored
    """
    out = {}
    for key, field in meta["features"].items():
        arr = traj[key].numpy()
        out[key] = torch.from_numpy(arr)
    return out


def convert_split(dataset_dir: Path, split: str, output_dir: Path):
    """Read all trajectories from one TFRecord split and write one .pt per trajectory."""
    with open(dataset_dir / "meta.json") as f:
        meta = json.load(f)

    tfrecord_path = dataset_dir / f"{split}.tfrecord"
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"Missing {tfrecord_path}")

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.map(functools.partial(parse_trajectory, meta=meta))

    n_written = 0
    sizes = []
    for traj in tqdm(ds, desc=f"converting {split}", unit="traj"):
        traj_torch = trajectory_to_torch(traj, meta)
        out_path = split_out / f"traj_{n_written:04d}.pt"
        torch.save(traj_torch, out_path)
        sizes.append({k: tuple(v.shape) for k, v in traj_torch.items()})
        n_written += 1

    print(f"  Wrote {n_written} trajectories to {split_out}")
    if sizes:
        # Sanity: show shape range across trajectories
        N_vals = [s["mesh_pos"][1] for s in sizes]
        M_vals = [s["cells"][1] for s in sizes]
        T_vals = [s["velocity"][0] for s in sizes]
        print(f"  Node count N: min={min(N_vals)}, max={max(N_vals)}, "
              f"unique={len(set(N_vals))}")
        print(f"  Cell count M: min={min(M_vals)}, max={max(M_vals)}")
        print(f"  Timesteps T:  {set(T_vals)}")
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cylinder_flow", "airfoil"], required=True)
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    output_dir = dataset_dir / "torch"
    print(f"Converting {args.dataset} / {args.split}")
    print(f"  input:  {dataset_dir / f'{args.split}.tfrecord'}")
    print(f"  output: {output_dir / args.split}/")
    convert_split(dataset_dir, args.split, output_dir)


if __name__ == "__main__":
    main()