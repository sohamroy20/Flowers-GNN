"""Generate and save full rollout trajectories for visualisation.

Loads a checkpoint, rolls out on chosen TEST trajectories, and saves the
predicted and ground-truth velocity fields plus the static mesh geometry.
Plotting is done separately so the GPU step and the matplotlib step stay
decoupled and you can re-plot without re-running rollouts.

Usage:
    python scripts/generate_rollouts.py \
        --checkpoint outputs/mgn_cylinder_.../checkpoints/epoch_30.pt \
        --model mgn \
        --out artifacts/rollouts/mgn_ep30.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset
from flowers_gnn.training.rollout import rollout_trajectory


# Fixed across every figure in the thesis. Chosen once, then frozen.
TEST_TRAJECTORIES = [0, 7, 23, 41]
MAX_STEPS = 200


def build_model(model_name: str, ckpt: dict) -> torch.nn.Module:
    """Instantiate using the architecture recorded in the checkpoint's cfg."""
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise ValueError("checkpoint has no cfg; cannot infer architecture")

    net = cfg["model"]["net"]

    if model_name == "mgn":
        from flowers_gnn.models.mgn import MGN
        return MGN(latent_size=net["latent_size"],
                   num_mp_steps=net["num_mp_steps"])

    from flowers_gnn.models.flowers_mesh import FlowersMesh
    return FlowersMesh(latent_dim=net["latent_dim"],
                       num_blocks=net["num_blocks"],
                       num_heads=net["num_heads"],
                       k_interp=net["k_interp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True, choices=["mgn", "flowers"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/cylinder_flow")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    print(f"checkpoint epoch: {ckpt['epoch']}, val_loss: {ckpt.get('val_loss')}")

    model = build_model(args.model, ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model.__class__.__name__}, params: {n_params:,}")

    root = Path(args.data_root)
    stats = torch.load(root / "torch" / "norm_stats.pt", weights_only=True)
    dataset = CylinderFlowDataset(root=root, split="test", in_memory=False)
    print(f"test trajectories available: {len(dataset.traj_paths)}")

    out = {}
    for tid in TEST_TRAJECTORIES:
        print(f"rolling out test trajectory {tid} ...", flush=True)

        first_frame_idx = dataset._cum_offsets[tid]
        static = dataset.get(first_frame_idx)

        blob = torch.load(dataset.traj_paths[tid], weights_only=True)
        traj = {"velocity": blob["velocity"].float()}

        res = rollout_trajectory(
            model=model, traj=traj, stats=stats, static=static,
            max_steps=MAX_STEPS, device=device,
        )

        out[f"pred_{tid}"] = res["pred"].cpu().numpy().astype(np.float32)
        out[f"target_{tid}"] = res["target"].cpu().numpy().astype(np.float32)
        out[f"pos_{tid}"] = dataset._traj_static[tid]["pos"].numpy().astype(np.float32)
        out[f"cells_{tid}"] = dataset._traj_static[tid]["cells"].numpy().astype(np.int32)
        out[f"node_type_{tid}"] = dataset._traj_static[tid]["node_type"].numpy().astype(np.int32)

    out["trajectories"] = np.array(TEST_TRAJECTORIES)
    out["epoch"] = np.array(ckpt["epoch"])
    out["n_params"] = np.array(n_params)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
