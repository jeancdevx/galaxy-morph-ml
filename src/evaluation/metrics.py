"""Evaluation metrics for galaxy morphology classification."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsTracker:
    """Track and compute classification metrics."""

    def __init__(self, num_classes: int = 4, class_names: Optional[list] = None):
        """Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes for reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update tracker with new predictions.
        
        Args:
            logits: Model logits of shape (B, num_classes)
            targets: Ground truth labels of shape (B,)
        """
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts = targets.cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(tgts)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        if len(self.predictions) == 0:
            return {}
        
        preds = np.array(self.predictions)
        tgts = np.array(self.targets)
        
        metrics = {
            "accuracy": accuracy_score(tgts, preds),
            "f1_macro": f1_score(tgts, preds, average="macro"),
            "f1_weighted": f1_score(tgts, preds, average="weighted"),
            "precision_macro": precision_score(tgts, preds, average="macro"),
            "recall_macro": recall_score(tgts, preds, average="macro"),
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            mask = tgts == i
            if mask.sum() > 0:
                class_preds = preds[mask]
                class_targets = tgts[mask]
                
                # Calculate per-class recall and precision
                per_class_recall = recall_score(
                    class_targets,
                    class_preds,
                    average=None,
                    zero_division=0,
                    labels=[i],
                )
                per_class_precision = precision_score(
                    class_targets,
                    class_preds,
                    average=None,
                    zero_division=0,
                    labels=[i],
                )
                
                metrics[f"recall_{class_name}"] = float(per_class_recall[0])
                metrics[f"precision_{class_name}"] = float(per_class_precision[0])
        
        return metrics
    
    def confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.
        
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        if len(self.predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        preds = np.array(self.predictions)
        tgts = np.array(self.targets)
        
        return confusion_matrix(tgts, preds, labels=list(range(self.num_classes)))
    
    def format_metrics(self, metrics: Dict[str, float], prefix: str = "") -> str:
        """Format metrics for printing.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for each line (e.g., "val_")
        
        Returns:
            Formatted string
        """
        lines = []
        
        # Main metrics
        if "accuracy" in metrics:
            val = float(metrics["accuracy"])
            lines.append(f"{prefix}Accuracy: {val:.4f}")
        if "f1_macro" in metrics:
            val = float(metrics["f1_macro"])
            lines.append(f"{prefix}F1 (macro): {val:.4f}")
        if "f1_weighted" in metrics:
            val = float(metrics["f1_weighted"])
            lines.append(f"{prefix}F1 (weighted): {val:.4f}")
        
        # Per-class recall
        recall_keys = [k for k in metrics.keys() if k.startswith("recall_")]
        if recall_keys:
            lines.append(f"{prefix}Per-class Recall:")
            for key in sorted(recall_keys):
                val = float(metrics[key])
                lines.append(f"  {key}: {val:.4f}")
        
        return "\n".join(lines)


def compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.
    
    Args:
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(targets, return_counts=True)
    total = len(targets)
    
    weights = np.zeros(num_classes)
    for cls, count in zip(unique, counts):
        weights[cls] = total / (num_classes * count)
    
    # Normalize to mean=1
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    print("=== Metrics Test ===")
    
    # Dummy data
    logits = torch.randn(16, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1, 1, 2, 2, 3, 3, 0, 0])
    
    tracker = MetricsTracker(num_classes=4, class_names=["Spiral", "Elliptical", "Lenticular", "Irregular"])
    tracker.update(logits, targets)
    
    metrics = tracker.compute()
    print("\nMetrics:")
    print(tracker.format_metrics(metrics, prefix="  "))
    
    print("\nConfusion Matrix:")
    cm = tracker.confusion_matrix()
    print(cm)
