"""Verifying rollout runs end-to-end with the dummy model and produces sane metrics."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset
from flowers_gnn.data.datamodule import CylinderDataModule
from flowers_gnn.models.dummy import DummyModel
from flowers_gnn.training.metrics import rollout_mse_rmse
from flowers_gnn.training.rollout import rollout_trajectory


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")   # small enough test to run on CPU

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    dm = CylinderDataModule(cfg.data)
    ds = CylinderFlowDataset(root=cfg.data.root, split="valid", in_memory=True,
                             max_trajectories=1)

    # Load the raw trajectory (velocity across all timesteps)
    traj_blob = torch.load(ds.traj_paths[0], weights_only=True)
    traj = {"velocity": traj_blob["velocity"].float()}
    static = ds.get(0)   # gives us pos/node_type/edge_index/loss_mask/cells for this traj

    model = DummyModel().to(device)
    out = rollout_trajectory(
        model=model,
        traj=traj,
        stats=dm.stats,
        static=static,
        max_steps=200,
        device=device,
    )

    print(f"pred shape:   {tuple(out['pred'].shape)}")
    print(f"target shape: {tuple(out['target'].shape)}")

    metrics = rollout_mse_rmse(out["pred"], out["target"],
                               horizons=list(cfg.train.rollout.horizons))
    print()
    print("Rollout metrics with DUMMY MODEL (predicts zero delta = 'velocity frozen'):")
    for k, v in metrics.items():
        print(f"  {k:20s}  {v:.6f}")


if __name__ == "__main__":
    main()