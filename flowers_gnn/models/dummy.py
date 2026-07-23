"""Dummy model: returns zeros for every node.
Purpose: smoke-test the end-to-end pipeline before we build MGN or FLOWERS.
If this model trains, validates, rolls out, and logs to W&B without errors,
the pipeline is proven correct. The actual predictions will be trash (constant
zeros), but that's fine — we're testing plumbing, not learning.

Contract for every model in this pipeline:
    forward(batch: torch_geometric.data.Batch) -> torch.Tensor of shape [total_N, 2]
    returns the predicted NORMALIZED velocity delta.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A single learnable parameter so the optimizer has something to update.
        # Without any params, .backward() would still work but there'd be nothing
        # to test the optimizer path against.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Batch) -> torch.Tensor:
        # batch.x is [total_N, 2]. Return zeros of the same shape.
        # We multiply by dummy_param so gradients can flow — otherwise the
        # optimizer step is a no-op and we can't verify it works.
        return torch.zeros_like(batch.x) + self.dummy_param * 0.0