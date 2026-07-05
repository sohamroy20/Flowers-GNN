"""Smoke test the CylinderFlowDataset on the valid split."""
import torch
from torch_geometric.loader import DataLoader

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset


def main():
    ds = CylinderFlowDataset(
        root="data/cylinder_flow",
        split="valid",
        in_memory=True,  # valid is small
        max_trajectories=5,  # quick test on 5 trajectories
    )
    print(f"Dataset has {len(ds)} frames across 5 trajectories")
    print(f"Expected: ~5 * 599 = ~2995 frames")
    print()

    sample = ds.get(0)
    print("Single sample (frame 0 of trajectory 0):")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} shape={tuple(v.shape)}  dtype={v.dtype}")
    print()

    # PyG batching: stacks variable-size graphs into one disjoint batch
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print("Batch of 4 (PyG disjoint stacking):")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s} shape={tuple(v.shape)}  dtype={v.dtype}")
    print(f"  batch index ('batch' attribute): shape={tuple(batch.batch.shape)}")
    print(f"  num_graphs: {batch.num_graphs}")


if __name__ == "__main__":
    main()