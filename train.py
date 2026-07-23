"""Hydra entry point for training.

Usage:
    python train.py                             # dummy model, defaults
    python train.py model=mgn                   # (when we add MGN)
    python train.py train.epochs=50 data.batch_size=8   # CLI overrides
"""

from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from flowers_gnn.training.trainer import Trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Print the resolved config once at the top so the run log has everything.
    print(OmegaConf.to_yaml(cfg))

    # W&B init
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        mode=cfg.experiment.wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(Path(cfg.output_dir)),
    )

    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()