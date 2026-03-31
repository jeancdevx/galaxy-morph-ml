"""Build a unified training dataset manifest for GalaxyMorph.

Uses the `gz2_class` column from Hart et al. (2016) to assign morphological
labels based on the Hubble-de Vaucouleurs classification scheme.

5 classes:
    0 - Elliptical      (gz2_class starts with 'E')
    1 - Spiral           (gz2_class starts with 'S', not 'SB'/'Se')
    2 - Barred_Spiral    (gz2_class starts with 'SB')
    3 - Edge_on          (gz2_class starts with 'Se')
    4 - Irregular_Merger (gz2_class contains '(i)', '(d)', or '(m)')

Outputs a CSV manifest with image paths, labels, and train/val/test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Label scheme ──────────────────────────────────────────────────────────

LABEL_MAP = {
    "Elliptical": 0,
    "Spiral": 1,
    "Barred_Spiral": 2,
    "Edge_on": 3,
    "Irregular_Merger": 4,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def classify_gz2(gz2_class: str) -> Optional[str]:
    """Map a gz2_class string to one of our 5 morphological labels.

    Priority: Irregular/Merger markers take precedence over base type,
    because visually these galaxies look irregular regardless of their
    underlying morphology.

    Args:
        gz2_class: Classification string from Hart et al. 2016.

    Returns:
        Label string, or None if the galaxy should be excluded.
    """
    cls = str(gz2_class).strip()

    # Exclude artifacts
    if cls == "A" or cls == "nan" or not cls:
        return None

    # Irregular/Merger: parenthetical markers
    if "(i)" in cls or "(d)" in cls or "(m)" in cls:
        return "Irregular_Merger"

    # Base type from prefix (order matters: SB before S)
    if cls.startswith("E"):
        return "Elliptical"
    if cls.startswith("SB"):
        return "Barred_Spiral"
    if cls.startswith("Se"):
        return "Edge_on"
    if cls.startswith("S"):
        return "Spiral"

    return None


# ── Helpers ───────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# ── Data loading ──────────────────────────────────────────────────────────

def load_sources(
    mapping_path: Path,
    metadata_path: Path,
    images_dir: Path,
) -> pd.DataFrame:
    """Load and merge the image mapping and metadata tables."""
    mapping = pd.read_csv(
        mapping_path, dtype={"objid": str, "asset_id": str}
    )
    metadata = pd.read_csv(
        metadata_path, compression="infer", dtype={"dr7objid": str}
    )

    if "objid" not in mapping.columns or "asset_id" not in mapping.columns:
        raise ValueError("Mapping file must contain columns: objid, asset_id")
    if "dr7objid" not in metadata.columns:
        raise ValueError("Metadata file must contain column: dr7objid")

    mapping = mapping.rename(columns={"objid": "object_id"})
    metadata = metadata.rename(columns={"dr7objid": "object_id"})

    # Ensure same dtype for merge
    mapping["object_id"] = mapping["object_id"].astype(str)
    metadata["object_id"] = metadata["object_id"].astype(str)

    # Build image path
    mapping["filename"] = mapping["asset_id"].astype(str) + ".jpg"
    mapping["image_path"] = mapping["filename"].apply(
        lambda name: str(images_dir / name)
    )

    merged = mapping.merge(
        metadata, on="object_id", how="inner", validate="many_to_one"
    )
    return merged


# ── Label assignment ──────────────────────────────────────────────────────

def assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign 5-class labels using gz2_class column."""
    if "gz2_class" not in frame.columns:
        raise ValueError(
            "Metadata must contain 'gz2_class' column (from Hart et al. 2016)"
        )

    frame["label"] = frame["gz2_class"].apply(classify_gz2)

    # Drop unclassifiable rows
    n_before = len(frame)
    frame = frame.dropna(subset=["label"]).copy()
    n_dropped = n_before - len(frame)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} unclassifiable rows (artifacts, NaN)")

    # Assign label indices
    frame["label_idx"] = frame["label"].map(LABEL_MAP)

    return frame


# ── Splitting ─────────────────────────────────────────────────────────────

def stratified_split(
    frame: pd.DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> pd.DataFrame:
    """Create stratified train/val/test splits."""
    if round(train_split + val_split + test_split, 6) != 1.0:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    train_df, temp_df = train_test_split(
        frame,
        test_size=(1.0 - train_split),
        random_state=seed,
        stratify=frame["label_idx"],
    )

    val_ratio_in_temp = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_df["label_idx"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main(config_file: str) -> None:
    cfg = load_yaml(resolve_path(config_file))

    dataset_cfg = cfg.get("dataset", {})
    build_cfg = cfg.get("build", {})

    images_dir = resolve_path(dataset_cfg["images_dir"])
    mapping_path = resolve_path(dataset_cfg["mapping_file"])
    metadata_path = resolve_path(dataset_cfg["metadata_file"])
    output_path = resolve_path(dataset_cfg["final_dataset_file"])

    train_split = float(build_cfg.get("train_split", 0.70))
    val_split = float(build_cfg.get("val_split", 0.15))
    test_split = float(build_cfg.get("test_split", 0.15))
    random_seed = int(build_cfg.get("random_seed", 42))

    # ── Validate paths ──
    for path in [images_dir, mapping_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    # ── Load & merge ──
    print("Loading data sources...")
    frame = load_sources(mapping_path, metadata_path, images_dir)
    print(f"  Merged rows: {len(frame):,}")

    # ── Assign labels from gz2_class ──
    print("Assigning labels from gz2_class...")
    frame = assign_labels(frame)

    # ── Validate images exist ──
    print("Validating image paths...")
    n_before = len(frame)
    frame = frame[frame["image_path"].map(lambda p: Path(p).exists())].copy()
    n_missing = n_before - len(frame)
    if n_missing > 0:
        print(f"  Dropped {n_missing:,} rows with missing images")

    if frame.empty:
        raise RuntimeError("No valid samples after filtering.")

    # ── Select output columns ──
    base_cols = [
        "image_path",
        "object_id",
        "asset_id",
        "gz2_class",
        "label",
        "label_idx",
    ]
    optional_cols = ["sample", "total_votes"]
    selected_cols = base_cols + [c for c in optional_cols if c in frame.columns]

    manifest = frame[selected_cols].copy()

    # ── Stratified split ──
    print("Creating stratified splits...")
    manifest = stratified_split(
        manifest,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=random_seed,
    )

    # ── Save ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Total samples: {len(manifest):,}")
    print(f"\nClass distribution:")
    for label_name, label_idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        count = (manifest["label"] == label_name).sum()
        pct = count / len(manifest) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label_idx} {label_name:<20s} {count:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"\nSplit distribution:")
    print(manifest["split"].value_counts().to_string())
    print(f"{'='*60}")


def parse_args() -> Tuple[str]:
    parser = argparse.ArgumentParser(
        description="Build unified GalaxyMorph dataset manifest (5 classes)"
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset config YAML (default: configs/dataset.yaml)",
    )
    args = parser.parse_args()
    return (args.config,)


if __name__ == "__main__":
    (config_path,) = parse_args()
    main(config_path)
