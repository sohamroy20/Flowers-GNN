"""Inspect a single CylinderFlow trajectory: load, print shapes, plot one frame.

Run from the project root:
    python scripts/inspect_cylinder.py
"""

import functools
import json
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import tensorflow as tf


DATA_DIR = "data/cylinder_flow"


def parse_trajectory(proto, meta):
    """Parse one trajectory from a tf.Example record.

    Mirrors meshgraphnets/dataset.py:_parse, simplified for our needs.
    """
    feature_lists = {
        name: tf.io.VarLenFeature(tf.string) for name in meta["field_names"]
    }
    features = tf.io.parse_single_example(proto, feature_lists)

    out = {}
    for key, field in meta["features"].items():
        raw = features[key].values
        data = tf.io.decode_raw(raw, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            # Static fields are stored once but apply at every timestep.
            # We tile to (T, N, C) so downstream code treats them uniformly.
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        out[key] = data
    return out


def load_n_trajectories(split: str, n: int = 5):
    """Loading the first n trajectories from a split."""
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)

    ds = tf.data.TFRecordDataset(os.path.join(DATA_DIR, f"{split}.tfrecord"))
    ds = ds.map(functools.partial(parse_trajectory, meta=meta))

    trajs = []
    for traj in ds.take(n):
        trajs.append({k: v.numpy() for k, v in traj.items()})
    return trajs, meta


def print_summary(traj, meta):
    print("=" * 60)
    print(f"meta:")
    print(f"  simulator: {meta['simulator']}")
    print(f"  dt: {meta['dt']}")
    print(f"  trajectory length: {meta['trajectory_length']}")
    print("=" * 60)
    print("trajectory shapes and dtypes:")
    for key, arr in traj.items():
        print(f"  {key:12s} shape={str(arr.shape):20s} dtype={arr.dtype}")
    print("=" * 60)
    print("node_type distribution (counts at t=0):")
    nt = traj["node_type"][0, :, 0]
    types_meanings = {
        0: "NORMAL",
        1: "OBSTACLE",
        2: "AIRFOIL",
        3: "HANDLE",
        4: "INFLOW",
        5: "OUTFLOW",
        6: "WALL_BOUNDARY",
    }
    for t, count in zip(*np.unique(nt, return_counts=True)):
        name = types_meanings.get(int(t), "UNKNOWN")
        print(f"  type {t} ({name:14s}): {count} nodes")
    print("=" * 60)
    print("velocity range (across all timesteps, all nodes):")
    v = traj["velocity"]
    print(f"  vx: min={v[..., 0].min():+.4f}  max={v[..., 0].max():+.4f}  mean={v[..., 0].mean():+.4f}")
    print(f"  vy: min={v[..., 1].min():+.4f}  max={v[..., 1].max():+.4f}  mean={v[..., 1].mean():+.4f}")
    print("=" * 60)


def plot_frame(traj, t: int = 0, save_path: str = "scripts/inspect_cylinder_frame.png"):
    mesh_pos = traj["mesh_pos"][0]   # (N, 2) — static, same at every t
    cells = traj["cells"][0]         # (M, 3) — triangles
    velocity = traj["velocity"][t]   # (N, 2)
    node_type = traj["node_type"][0, :, 0]

    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], cells)
    speed = np.linalg.norm(velocity, axis=-1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top: mesh with node-type coloring
    ax = axes[0]
    ax.triplot(triang, color="lightgray", linewidth=0.3)
    sc = ax.scatter(
        mesh_pos[:, 0], mesh_pos[:, 1],
        c=node_type, cmap="tab10", s=4, vmin=0, vmax=9,
    )
    ax.set_aspect("equal")
    ax.set_title(f"Mesh + node types (N={mesh_pos.shape[0]}, M={cells.shape[0]} cells)")
    plt.colorbar(sc, ax=ax, label="node_type", shrink=0.8)

    # Bottom: velocity magnitude at t
    ax = axes[1]
    tpc = ax.tripcolor(triang, speed, shading="gouraud", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title(f"|velocity| at t={t}")
    plt.colorbar(tpc, ax=ax, label="speed", shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved figure to {save_path}")


def main():
    trajs, meta = load_n_trajectories("valid", n=5)

    print("=" * 60)
    print(f"Loaded {len(trajs)} trajectories from valid")
    print("=" * 60)
    print(f"{'idx':>4s}  {'N (nodes)':>10s}  {'M (cells)':>10s}  {'T':>5s}")
    for i, traj in enumerate(trajs):
        N = traj["mesh_pos"].shape[1]
        M = traj["cells"].shape[1]
        T = traj["velocity"].shape[0]
        print(f"{i:>4d}  {N:>10d}  {M:>10d}  {T:>5d}")
    print("=" * 60)

    # Detailed look at the first one
    print("First trajectory in detail:")
    print_summary(trajs[0], meta)
    plot_frame(trajs[0], t=0)


if __name__ == "__main__":
    main()