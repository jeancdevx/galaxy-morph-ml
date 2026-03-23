"""Galaxy morphology dataset and dataloaders for PyTorch.

This module provides:
- GalaxyMorphDataset: PyTorch Dataset for loading images and labels
- get_train_dataloader: DataLoader with augmentation and class balancing
- get_val_dataloader, get_test_dataloader: DataLoaders for evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as transforms


class GalaxyMorphDataset(Dataset):
    """PyTorch Dataset for Galaxy Zoo galaxy morphology classification.
    
    Loads images from disk and returns (image, label_idx) tuples.
    """

    def __init__(
        self,
        dataset_file: str,
        images_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize dataset.
        
        Args:
            dataset_file: Path to galaxy_dataset.csv manifest
            images_dir: Optional override for images directory
            split: One of 'train', 'val', 'test'
            transform: Optional torchvision transforms
        """
        self.dataset_file = Path(dataset_file)
        self.split = split
        self.transform = transform

        # Load manifest
        self.manifest = pd.read_csv(self.dataset_file)

        # Filter by split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)

        if len(self.manifest) == 0:
            raise ValueError(f"No samples found for split '{split}' in {dataset_file}")

        # Use provided images_dir or infer from first row
        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            first_image_path = Path(self.manifest.iloc[0]["image_path"])
            self.images_dir = first_image_path.parent

        print(
            f"Loaded {split} dataset: {len(self.manifest)} samples "
            f"from {self.dataset_file}"
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]

        # Load image
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label index
        label_idx = int(row["label_idx"])

        return image, label_idx


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Get augmentation transforms for training.
    
    Args:
        input_size: Input size for ResNet (typically 224)
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """Get transforms for validation/testing (no augmentation).
    
    Args:
        input_size: Input size for ResNet (typically 224)
    
    Returns:
        Composed transforms for evaluation
    """
    return transforms.Compose(
        [
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_class_weights(dataset_file: str) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.
    
    Uses inverse frequency weighting: w_c = N / (n_classes * n_c)
    
    Args:
        dataset_file: Path to galaxy_dataset.csv
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    manifest = pd.read_csv(dataset_file)

    # Filter to training split only
    manifest = manifest[manifest["split"] == "train"]

    class_counts = manifest["label_idx"].value_counts().sort_index()
    num_classes = len(class_counts)
    total_samples = len(manifest)

    # w_c = N / (k * n_c) where N=total, k=num_classes, n_c=count_class_c
    class_weights = total_samples / (num_classes * class_counts.values)

    # Normalize to mean=1 for numerical stability
    class_weights = class_weights / class_weights.mean()

    return torch.tensor(class_weights, dtype=torch.float32)


def get_sample_weights(dataset_file: str, split: str = "train") -> np.ndarray:
    """Compute sample weights for WeightedRandomSampler.
    
    Args:
        dataset_file: Path to galaxy_dataset.csv
        split: One of 'train', 'val', 'test'
    
    Returns:
        Array of weights for each sample in the split
    """
    manifest = pd.read_csv(dataset_file)
    manifest = manifest[manifest["split"] == split].reset_index(drop=True)

    class_counts = manifest["label_idx"].value_counts()
    num_classes = len(class_counts)
    total_samples = len(manifest)

    # Weight per class
    class_weights = total_samples / (num_classes * class_counts)

    # Assign weight to each sample based on its class
    sample_weights = manifest["label_idx"].map(class_weights).values
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    return sample_weights


def get_train_dataloader(
    dataset_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
    balanced: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create training DataLoader with augmentation and optional balancing.
    
    Args:
        dataset_file: Path to galaxy_dataset.csv
        batch_size: Batch size
        num_workers: Number of workers for data loading
        input_size: Input size for transforms
        balanced: Use WeightedRandomSampler for class balancing
        seed: Random seed
    
    Returns:
        PyTorch DataLoader for training
    """
    dataset = GalaxyMorphDataset(
        dataset_file=dataset_file,
        split="train",
        transform=get_train_transforms(input_size),
    )

    if balanced:
        sample_weights = get_sample_weights(dataset_file, split="train")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )


def get_val_dataloader(
    dataset_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
) -> DataLoader:
    """Create validation DataLoader.
    
    Args:
        dataset_file: Path to galaxy_dataset.csv
        batch_size: Batch size
        num_workers: Number of workers
        input_size: Input size for transforms
    
    Returns:
        PyTorch DataLoader for validation
    """
    dataset = GalaxyMorphDataset(
        dataset_file=dataset_file,
        split="val",
        transform=get_val_transforms(input_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_test_dataloader(
    dataset_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
) -> DataLoader:
    """Create test DataLoader.
    
    Args:
        dataset_file: Path to galaxy_dataset.csv
        batch_size: Batch size
        num_workers: Number of workers
        input_size: Input size for transforms
    
    Returns:
        PyTorch DataLoader for testing
    """
    dataset = GalaxyMorphDataset(
        dataset_file=dataset_file,
        split="test",
        transform=get_val_transforms(input_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Quick test
    dataset_file = "data/processed/galaxy_dataset.csv"
    
    print("\n=== Class weights ===")
    class_weights = get_class_weights(dataset_file)
    print(f"Class weights: {class_weights}")
    
    print("\n=== Train DataLoader (balanced) ===")
    train_loader = get_train_dataloader(
        dataset_file,
        batch_size=8,
        num_workers=0,
        balanced=True,
    )
    batch = next(iter(train_loader))
    print(f"Batch shape: images {batch[0].shape}, labels {batch[1].shape}")
    print(f"Label distribution in batch: {np.bincount(batch[1].numpy())}")
    
    print("\n=== Val DataLoader ===")
    val_loader = get_val_dataloader(dataset_file, batch_size=8, num_workers=0)
    batch = next(iter(val_loader))
    print(f"Batch shape: images {batch[0].shape}, labels {batch[1].shape}")
    
    print("\n=== Test DataLoader ===")
    test_loader = get_test_dataloader(dataset_file, batch_size=8, num_workers=0)
    batch = next(iter(test_loader))
    print(f"Batch shape: images {batch[0].shape}, labels {batch[1].shape}")
