"""Diagnose whether learned warps are local or long-range.

Registers hooks on each SelfWarpMesh block to capture the predicted
displacements, then compares their magnitudes against the mesh's own
length scales (nearest-neighbour spacing, domain extent).

If displacements are comparable to mesh spacing, the warp is effectively
local and multiscale machinery would help. If they span a meaningful
fraction of the domain, the warp is already doing long-range routing.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset
from flowers_gnn.models.flowers_mesh import FlowersMesh
from flowers_gnn.models.selfwarp_mesh import SelfWarpMesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/cylinder_flow")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    net = ckpt["cfg"]["model"]["net"]
    print(f"checkpoint epoch {ckpt['epoch']}, config: "
          f"latent={net['latent_dim']} blocks={net['num_blocks']} heads={net['num_heads']}")

    model = FlowersMesh(latent_dim=net["latent_dim"],
                        num_blocks=net["num_blocks"],
                        num_heads=net["num_heads"],
                        k_interp=net["k_interp"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Capture the flow tensor each block computes.
    captured = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            h, pos, batch_idx = inputs
            x = module.norm_pre_warp(h)
            flow = module.flow_head(x).view(-1, module.num_heads, 2)
            captured[idx] = flow.detach().cpu()
        return hook

    handles = [blk.register_forward_hook(make_hook(i))
               for i, blk in enumerate(model.blocks)]

    root = Path(args.data_root)
    stats = torch.load(root / "torch" / "norm_stats.pt", weights_only=True)
    ds = CylinderFlowDataset(root=root, split=args.split, in_memory=False)

    v_mean, v_std = stats["velocity_mean"], stats["velocity_std"]

    per_block = {i: [] for i in range(len(model.blocks))}
    mesh_spacings, domain_extents = [], []

    from torch_geometric.data import Batch
    for f in range(args.n_frames):
        data = ds.get(f)
        data.x = ((data.x - v_mean) / v_std)
        batch = Batch.from_data_list([data]).to(device)

        with torch.no_grad():
            model(batch)

        pos = data.pos.numpy()
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        mesh_spacings.append(np.median(d.min(axis=1)))
        domain_extents.append(float(np.ptp(pos, axis=0).max()))

        for i in range(len(model.blocks)):
            per_block[i].append(captured[i].norm(dim=-1).flatten().numpy())

    for h in handles:
        h.remove()

    spacing = float(np.median(mesh_spacings))
    extent = float(np.median(domain_extents))
    print(f"\nmedian nearest-neighbour spacing : {spacing:.5f}")
    print(f"domain extent (longest side)     : {extent:.5f}")
    print(f"\n{'block':>5}  {'median':>9} {'p90':>9} {'max':>9}   "
          f"{'/spacing':>9} {'/extent':>9}")
    for i in range(len(model.blocks)):
        m = np.concatenate(per_block[i])
        med = np.median(m)
        print(f"{i:>5}  {med:9.5f} {np.percentile(m, 90):9.5f} {m.max():9.5f}   "
              f"{med / spacing:9.2f} {med / extent:9.4f}")

    print("\nInterpretation: displacement/spacing near 1 means the warp reaches "
          "about one mesh cell (effectively local). Values in the tens, or "
          "displacement/extent above ~0.1, mean genuine long-range routing.")


if __name__ == "__main__":
    main()
