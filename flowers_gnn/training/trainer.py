"""Shared trainer for MGN, FLOWERS-mesh, and any other model satisfying our contract.

The pipeline is deliberately model-agnostic. The only thing that varies between
runs is the model itself (via cfg.model.net._target_). Everything below —
normalization, noise, loss, rollout, metrics, logging — is made identical across
models to analyze and compare them. This is what makes the model-vs-model comparison scientifically valid.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset
from flowers_gnn.data.datamodule import CylinderDataModule
from flowers_gnn.training.metrics import rollout_mse_rmse
from flowers_gnn.training.noise import apply_training_noise
from flowers_gnn.training.rollout import rollout_trajectory


# ---------------------------------------------------------------------- utility

def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def normalize_batch(batch: Batch, stats: Dict[str, torch.Tensor]) -> Batch:
    """Normalize x (velocity) and y (delta). Runs on the batch's current device."""
    v_mean = stats["velocity_mean"].to(batch.x.device)
    v_std = stats["velocity_std"].to(batch.x.device)
    d_mean = stats["delta_mean"].to(batch.x.device)
    d_std = stats["delta_std"].to(batch.x.device)
    batch.x = (batch.x - v_mean) / v_std
    batch.y = (batch.y - d_mean) / d_std
    return batch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over masked nodes and both channels."""
    diff2 = (pred - target) ** 2                 # [total_N, 2]
    per_node = diff2.mean(dim=-1)                # [total_N]
    return per_node[mask].mean()


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    eta_min_factor: float,
) -> LambdaLR:
    """Linear warmup for `warmup_epochs`, then cosine decay to lr * eta_min_factor."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        # Cosine decay from 1.0 down to eta_min_factor
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine
    return LambdaLR(optimizer, lr_lambda)


def param_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5


# ---------------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        print(f"Using device: {self.device}")

        torch.manual_seed(cfg.experiment.seed)

        # Data
        self.dm = CylinderDataModule(cfg.data)
        self.stats = self.dm.stats
        self.train_loader = self.dm.train_loader()
        self.valid_loader = self.dm.valid_loader()

        # Model
        self.model: torch.nn.Module = hydra.utils.instantiate(cfg.model.net).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model {cfg.model.name}: {n_params:,} params")

        # Optimizer + scheduler
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            cfg.train.optimizer, params=self.model.parameters()
        )
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            total_epochs=cfg.train.epochs,
            warmup_epochs=cfg.train.lr_scheduler.warmup_epochs,
            eta_min_factor=cfg.train.lr_scheduler.eta_min_factor,
        )

        # Output dir
        self.output_dir = Path(cfg.output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # Rollout support: eager-loaded valid trajectories for rollout eval
        self.rollout_dataset = CylinderFlowDataset(
            root=cfg.data.root,
            split="valid",
            in_memory=True,
            max_trajectories=cfg.train.rollout.max_trajectories,
        )

        # Bookkeeping
        self.best_val_loss = float("inf")
        self.checkpoint_paths: List[Path] = []

    # ---------- one step ----------

    def train_step(self, batch: Batch) -> float:
        batch = batch.to(self.device)
        batch = apply_training_noise(
            batch,
            std=self.cfg.data.train_noise_std,
            gamma=self.cfg.data.train_noise_gamma,
        )
        batch = normalize_batch(batch, self.stats)

        pred = self.model(batch)
        loss = masked_mse(pred, batch.y, batch.loss_mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.inference_mode()
    def valid_step(self, batch: Batch) -> float:
        batch = batch.to(self.device)
        batch = normalize_batch(batch, self.stats)
        pred = self.model(batch)
        return masked_mse(pred, batch.y, batch.loss_mask).item()

    # ---------- one epoch ----------

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []
        t0 = time.time()
        for batch in self.train_loader:
            losses.append(self.train_step(batch))
        mean = sum(losses) / max(1, len(losses))
        print(f"[epoch {epoch}] train loss {mean:.6f}  ({time.time() - t0:.1f}s)")
        return mean

    def validate(self, epoch: int) -> float:
        self.model.eval()
        losses = []
        for batch in self.valid_loader:
            losses.append(self.valid_step(batch))
        mean = sum(losses) / max(1, len(losses))
        print(f"[epoch {epoch}] valid loss {mean:.6f}")
        return mean

    def rollout_eval(self, epoch: int) -> Dict[str, float]:
        """Roll out N valid trajectories, average metrics."""
        self.model.eval()
        horizons = list(self.cfg.train.rollout.horizons)
        max_steps = max(horizons)

        all_metrics: Dict[str, List[float]] = {}
        for i in range(len(self.rollout_dataset.traj_paths)):
            traj_blob = torch.load(self.rollout_dataset.traj_paths[i], weights_only=True)
            traj = {"velocity": traj_blob["velocity"].float()}
            # get(idx=0 of this traj) gives us the static fields for this specific traj
            first_frame_idx = self.rollout_dataset._cum_offsets[i]
            static = self.rollout_dataset.get(first_frame_idx)

            out = rollout_trajectory(
                model=self.model,
                traj=traj,
                stats=self.stats,
                static=static,
                max_steps=max_steps,
                device=self.device,
            )
            m = rollout_mse_rmse(out["pred"], out["target"], horizons=horizons)
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)

        # Average across trajectories
        avg = {k: sum(vs) / len(vs) for k, vs in all_metrics.items()}
        print(f"[epoch {epoch}] rollout: "
              + ", ".join(f"{k}={v:.4f}" for k, v in avg.items() if k.startswith("rmse")))
        return avg

    # ---------- checkpoints ----------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "epoch"):
        path = self.output_dir / "checkpoints" / f"{tag}_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "val_loss": val_loss,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }, path)
        if tag == "epoch":
            self.checkpoint_paths.append(path)
            keep = self.cfg.train.checkpoint.keep_last
            while len(self.checkpoint_paths) > keep:
                self.checkpoint_paths.pop(0).unlink(missing_ok=True)
        return path

    # ---------- main loop ----------

    def train(self):
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "lr": self.lr_scheduler.get_last_lr()[0],
                "param_norm": param_norm(self.model),
            }

            if epoch % self.cfg.train.rollout.eval_every_n_epochs == 0:
                for k, v in self.rollout_eval(epoch).items():
                    log[f"rollout/{k}"] = v

            wandb.log(log, step=epoch)

            self.lr_scheduler.step()

            # Checkpoint
            if epoch % self.cfg.train.checkpoint.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_loss, tag="epoch")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")

        print("Training complete.")