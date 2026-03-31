"""Training loop for galaxy morphology classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import get_train_dataloader, get_val_dataloader
from src.evaluation.metrics import MetricsTracker


class Trainer:
    """Training orchestrator for galaxy morphology model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        checkpoint_dir: Path | str = "./models/checkpoints",
        log_dir: Path | str = "./logs",
        class_names: list = None,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: PyTorch optimizer
            loss_fn: Loss function (e.g., CrossEntropyLoss with weights)
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            class_names: Names of classes for reporting
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or ["Elliptical", "Spiral", "Barred_Spiral", "Edge_on", "Irregular_Merger"]
        
        # History tracking
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1_macro": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
            "best_val_f1_macro": 0.0,
            "best_epoch": 0,
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        total_loss = 0.0
        metrics_tracker = MetricsTracker(
            num_classes=len(self.class_names),
            class_names=self.class_names,
        )
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch+1}",
            leave=False,
        )
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            metrics_tracker.update(logits.detach(), targets)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics_tracker.compute()
        epoch_metrics["loss"] = avg_loss
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        metrics_tracker = MetricsTracker(
            num_classes=len(self.class_names),
            class_names=self.class_names,
        )
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Val Epoch {epoch+1}",
            leave=False,
        )
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)
            
            total_loss += loss.item()
            metrics_tracker.update(logits, targets)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        epoch_metrics = metrics_tracker.compute()
        epoch_metrics["loss"] = avg_loss
        epoch_metrics["confusion_matrix"] = metrics_tracker.confusion_matrix().tolist()
        
        return epoch_metrics
    
    def fit(
        self,
        num_epochs: int,
        save_interval: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
            verbose: Print progress
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_metrics.get("loss", 0.0))
            self.history["train_accuracy"].append(train_metrics.get("accuracy", 0.0))
            self.history["train_f1_macro"].append(train_metrics.get("f1_macro", 0.0))
            
            self.history["val_loss"].append(val_metrics.get("loss", 0.0))
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_f1_macro"].append(val_metrics.get("f1_macro", 0.0))
            self.history["val_f1_weighted"].append(val_metrics.get("f1_weighted", 0.0))
            
            # Check for best model
            if val_metrics.get("f1_macro", 0.0) > self.history["best_val_f1_macro"]:
                self.history["best_val_f1_macro"] = val_metrics.get("f1_macro", 0.0)
                self.history["best_epoch"] = epoch
                self._save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch)
            
            # Log progress
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_metrics.get('loss', 0.0):.4f} | "
                      f"Acc: {train_metrics.get('accuracy', 0.0):.4f} | "
                      f"F1: {train_metrics.get('f1_macro', 0.0):.4f}")
                print(f"Val Loss:   {val_metrics.get('loss', 0.0):.4f} | "
                      f"Acc: {val_metrics.get('accuracy', 0.0):.4f} | "
                      f"F1: {val_metrics.get('f1_macro', 0.0):.4f}")
                print(f"Best F1 (Val): {self.history['best_val_f1_macro']:.4f} "
                      f"(Epoch {self.history['best_epoch']+1})")
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best F1 (macro): {self.history['best_val_f1_macro']:.4f}")
        print(f"Best epoch: {self.history['best_epoch']+1}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            print(f"✓ Saved best model: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})
        
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_file = self.log_dir / "training_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            elif isinstance(value, list):
                history_serializable[key] = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in value
                ]
            else:
                history_serializable[key] = value
        
        with history_file.open("w") as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"✓ Saved training history: {history_file}")
