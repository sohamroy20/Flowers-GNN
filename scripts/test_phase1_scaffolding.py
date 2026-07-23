"""Smoke test: config loads, model instantiates, datamodule yields batches, model forwards."""
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def main():
    # Manually compose config (not using @hydra.main so we can run inline)
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg: DictConfig = compose(config_name="config")

    print("=" * 60)
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Instantiate the model
    #model = hydra.utils.instantiate(cfg.model)
    model = hydra.utils.instantiate(cfg.model.net)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__} with {n_params} params")

    # Instantiate the datamodule
    from flowers_gnn.data.datamodule import CylinderDataModule
    dm = CylinderDataModule(cfg.data)
    print(f"Normalization stats loaded, keys: {list(dm.stats.keys())}")

    # Grab one batch and forward it
    loader = dm.valid_loader()
    batch = next(iter(loader))
    print(f"Batch: {batch.num_graphs} graphs, x shape {tuple(batch.x.shape)}")

    with torch.no_grad():
        pred = model(batch)
    print(f"Prediction shape: {tuple(pred.shape)}  (should match batch.x)")
    print("=" * 60)
    print("Phase 1 scaffolding OK.")


if __name__ == "__main__":
    main()