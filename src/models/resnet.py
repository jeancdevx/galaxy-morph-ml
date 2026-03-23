"""ResNet50 model wrapper for galaxy morphology classification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class GalaxyMorphResNet50(nn.Module):
    """ResNet50 backbone with custom head for 4-class galaxy morphology classification.
    
    Architecture:
    - ResNet50 pretrained backbone
    - Global Average Pooling
    - Fully connected layer: 2048 -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """Initialize ResNet50 classifier.
        
        Args:
            num_classes: Number of output classes (default: 4)
            pretrained: Use ImageNet pretrained weights (default: True)
            dropout: Dropout rate before final layer (default: 0.5)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet50 backbone
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Identity()  # Remove original fc layer
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Backbone -> (B, 2048, 1, 1) after global avg pool
        backbone_out = self.backbone(x)
        
        # Classification head -> (B, num_classes)
        logits = self.head(backbone_out)
        
        return logits
    
    def save_checkpoint(self, path: Path | str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Checkpoint loaded: {path}")


def create_model(
    num_classes: int = 4,
    pretrained: bool = True,
    device: str = "cpu",
) -> GalaxyMorphResNet50:
    """Factory function to create and prepare model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to move model to
    
    Returns:
        Model on specified device
    """
    model = GalaxyMorphResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Quick test
    print("=== ResNet50 Model Test ===")
    model = create_model(num_classes=4, pretrained=True)
    print(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test checkpoint save/load
    checkpoint_path = "/tmp/test_checkpoint.pt"
    model.save_checkpoint(checkpoint_path)
    
    model2 = create_model(num_classes=4, pretrained=False)
    model2.load_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint save/load works")
