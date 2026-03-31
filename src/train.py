#!/usr/bin/env python
"""Train galaxy morphology classifier.

Usage:
    python -m src.train [--config CONFIG] [--device DEVICE] [--epochs EPOCHS]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.config import load_config
from src.data import get_train_dataloader, get_val_dataloader, get_class_weights
from src.models import create_model
from src.training import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_str: str) -> Path:
    """Resolve path relative to project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main(
    config_file: str = "configs/default.yaml",
    device: str = "cpu",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    save_interval: int = 5,
) -> None:
    """Main training script.
    
    Args:
        config_file: Path to config file
        device: Device to train on ('cpu' or 'cuda')
        num_epochs: Number of epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        save_interval: Save checkpoint every N epochs
    """
    
    # Load configs
    model_cfg = load_yaml(resolve_path("configs/default.yaml")).get("model", {})
    dataset_cfg = load_yaml(resolve_path("configs/dataset.yaml")).get("dataset", {})
    training_cfg = load_yaml(resolve_path("configs/default.yaml")).get("training", {})
    
    # Override with CLI args
    num_epochs = num_epochs or training_cfg.get("num_epochs", 50)
    batch_size = batch_size or training_cfg.get("batch_size", 32)
    learning_rate = learning_rate or training_cfg.get("learning_rate", 1e-4)
    
    print("\n" + "="*60)
    print("GalaxyMorph Training Pipeline")
    print("="*60)
    print(f"Config: {config_file}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Resolve dataset path
    dataset_file = resolve_path(dataset_cfg["final_dataset_file"])
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    print(f"✓ Dataset: {dataset_file}")
    
    # Set device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
    
    device = torch.device(device)
    print(f"✓ Using device: {device}\n")
    
    # Create model
    num_classes = model_cfg.get("num_classes", 5)
    model = create_model(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        device=str(device),
    )
    print(f"✓ Created {model_cfg.get('name', 'ResNet50')} model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Create dataloaders
    # Note: Use num_workers=0 in Docker environments to avoid shared memory issues
    print("Creating dataloaders...")
    num_workers = training_cfg.get("num_workers", 0)
    
    train_loader = get_train_dataloader(
        dataset_file=str(dataset_file),
        batch_size=batch_size,
        num_workers=training_cfg.get("num_workers", 0),
        input_size=model_cfg.get("input_size", 224),
        balanced=True,
    )
    print(f"✓ Train loader: {len(train_loader)} batches")
    
    val_loader = get_val_dataloader(
        dataset_file=str(dataset_file),
        batch_size=batch_size,
        num_workers=training_cfg.get("num_workers", 0),
        input_size=model_cfg.get("input_size", 224),
    )
    print(f"✓ Val loader: {len(val_loader)} batches\n")
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_cfg.get("weight_decay", 1e-5),
    )
    print(f"✓ Optimizer: Adam (lr={learning_rate})")
    
    # Setup loss function with class weights
    class_weights = get_class_weights(str(dataset_file))
    class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    print(f"✓ Loss: CrossEntropyLoss with class weights")
    print(f"  Weights: {class_weights.cpu().numpy()}\n")
    
    # Setup trainer
    class_names = ["Elliptical", "Spiral", "Barred_Spiral", "Edge_on", "Irregular_Merger"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=str(device),
        checkpoint_dir="models/checkpoints",
        log_dir="logs",
        class_names=class_names,
    )
    
    # Train
    history = trainer.fit(
        num_epochs=num_epochs,
        save_interval=save_interval,
        verbose=True,
    )
    
    # Save history
    trainer.save_history()
    
    print("\n✓ Training complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train galaxy morphology classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        config_file=args.config,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
    )
