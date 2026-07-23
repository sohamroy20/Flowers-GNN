"""MGN-style training noise augmentation.

Applies Gaussian noise to the input velocity on NORMAL nodes only, during
training. Boundary nodes are left clean because their values are dictated by
the simulation setup, not predicted by the model.

For cfd (gamma=1.0), the target is left unchanged: the model is trained to
predict the CLEAN next-step delta from a NOISED current velocity. This is the
rollout-stability trick from MeshGraphNets — it teaches the model to be robust
to its own future prediction errors.

Applied identically to every model in the pipeline (config-controlled). This is
one of the confounder controls: if MGN gets the trick, FLOWERS gets it too.
"""

import torch
from torch_geometric.data import Batch

from flowers_gnn.data.cylinder_dataset import NodeType


def apply_training_noise(
    batch: Batch,
    std: float,
    gamma: float = 1.0,
) -> Batch:
    """Add Gaussian noise to batch.x on NORMAL nodes and adjust batch.y accordingly.

    Modifies the batch in-place AND returns it. Skip entirely if std <= 0.

    Args:
        batch: PyG Batch with x [total_N, 2], y [total_N, 2], node_type [total_N].
        std: Gaussian noise std in *physical* velocity units (MGN uses 0.02 for cfd).
             Applied AFTER the batch is loaded but BEFORE the model sees it.
             Note: our x is in physical units at this point (dataset returns raw
             velocity), so std is in the same units.
        gamma: How much of the noise "leaks" into the target.
               gamma=1.0 (MGN cfd default) -> target unchanged. Model learns to
               predict the clean delta from a noised input.
               gamma<1.0 -> some of the noise is added to the target too. Used for
               cloth in MGN; not relevant for us but kept for completeness.

    Returns:
        The same batch object (modified).
    """
    if std <= 0.0:
        return batch

    # Sample noise for every node, then zero it out for non-NORMAL nodes.
    noise = torch.randn_like(batch.x) * std
    normal_mask = (batch.node_type == NodeType.NORMAL).unsqueeze(-1)  # [total_N, 1]
    noise = noise * normal_mask.to(noise.dtype)

    batch.x = batch.x + noise
    if gamma < 1.0:
        batch.y = batch.y + (1.0 - gamma) * noise

    return batch