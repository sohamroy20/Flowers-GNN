"""Rollout -> npz for the RIGNO datasets (Heat-L-Sines, Wave-C-Sines).

Separate from generate_rollouts.py, which is CylinderFlow-specific: it
hardcodes CylinderFlowDataset and its build_model() drops the n_fields /
use_node_type / max_disp kwargs the RIGNO models need. Here the net is
built with hydra.instantiate from the recorded cfg, so every kwarg is
carried over whatever the config contains.

Writes the contract make_rigno_gifs.py expects: coords, gt, pred.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from flowers_gnn.data.datamodule import CylinderDataModule
from flowers_gnn.training.rollout import rollout_trajectory

MAX_STEPS = 20          # RIGNO trajectories hold 21 frames


def build_model(ckpt, model_cfg_name):
    """Prefer the architecture recorded in the checkpoint; fall back to the
    config file. instantiate() passes through every kwarg, including the
    ones the CylinderFlow builder omits."""
    net = (ckpt.get("cfg", {}) or {}).get("model", {}).get("net", None)
    if net is None:
        print("  ckpt has no cfg; falling back to configs/model/"
              f"{model_cfg_name}.yaml")
        net = OmegaConf.load(f"configs/model/{model_cfg_name}.yaml").net
    return instantiate(OmegaConf.create(OmegaConf.to_container(
        OmegaConf.create(net), resolve=True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model config name, e.g. mgn_heat")
    ap.add_argument("--data", required=True, help="data config name, e.g. heat_l_sines")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--traj-idx", type=int, default=0)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    print(f"checkpoint epoch {ckpt.get('epoch')}  val_loss {ckpt.get('val_loss')}")

    model = build_model(ckpt, a.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"model {model.__class__.__name__}  "
          f"params {sum(p.numel() for p in model.parameters()):,}")

    dcfg = OmegaConf.load(f"configs/data/{a.data}.yaml")
    dm = CylinderDataModule(dcfg)
    ds = dm.valid_dataset() if a.split == "valid" else dm.test_dataset()
    if not hasattr(ds, "trajectory"):
        raise SystemExit(f"{a.data} is not a RIGNO dataset; use generate_rollouts.py")
    print(f"{a.split} trajectories: {ds.n_traj}")

    traj = ds.trajectory(a.traj_idx)
    static = ds.static()
    res = rollout_trajectory(model=model, traj=traj, stats=dm.stats,
                             static=static, max_steps=a.max_steps, device=device)

    pred = res["pred"].cpu().numpy().astype(np.float32)
    gt = res["target"].cpu().numpy().astype(np.float32)
    coords = ds.pos.cpu().numpy().astype(np.float32)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, coords=coords, gt=gt, pred=pred,
                        epoch=np.array(ckpt.get("epoch", -1)))
    print(f"saved {a.out}   coords {coords.shape}  gt {gt.shape}  pred {pred.shape}")
    rmse = float(np.sqrt(((pred - gt) ** 2).mean()))
    print(f"  whole-rollout RMSE {rmse:.6f}")


if __name__ == "__main__":
    main()
