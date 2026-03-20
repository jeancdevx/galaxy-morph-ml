"""Build a unified training dataset manifest for GalaxyMorph.

This script merges:
- image filename mapping (objid <-> asset_id)
- Galaxy Zoo metadata table (hart16)

It outputs a CSV manifest with image paths, labels, and data split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def safe_col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def build_scores(frame: pd.DataFrame) -> pd.DataFrame:
    smooth = safe_col(frame, "t01_smooth_or_features_a01_smooth_debiased")
    features = safe_col(frame, "t01_smooth_or_features_a02_features_or_disk_debiased")
    edgeon = safe_col(frame, "t02_edgeon_a04_yes_debiased")
    spiral = safe_col(frame, "t04_spiral_a08_spiral_debiased")

    bulge_noticeable = safe_col(frame, "t05_bulge_prominence_a11_just_noticeable_debiased")
    bulge_obvious = safe_col(frame, "t05_bulge_prominence_a12_obvious_debiased")
    bulge_dominant = safe_col(frame, "t05_bulge_prominence_a13_dominant_debiased")
    bulge_score = (bulge_noticeable + bulge_obvious + bulge_dominant).clip(upper=1.0)

    irregular = safe_col(frame, "t08_odd_feature_a22_irregular_debiased")
    disturbed = safe_col(frame, "t08_odd_feature_a23_disturbed_debiased")
    merger = safe_col(frame, "t08_odd_feature_a24_merger_debiased")
    odd_yes = safe_col(frame, "t06_odd_a14_yes_debiased")

    # Heuristic scores for 4-class setup.
    frame["score_spiral"] = (features * spiral).clip(lower=0.0, upper=1.0)
    frame["score_elliptical"] = smooth.clip(lower=0.0, upper=1.0)
    frame["score_lenticular"] = (features * edgeon * bulge_score).clip(lower=0.0, upper=1.0)
    frame["score_irregular"] = (
        odd_yes * pd.concat([irregular, disturbed, merger], axis=1).max(axis=1)
    ).clip(lower=0.0, upper=1.0)

    return frame


def assign_label(frame: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "score_spiral",
        "score_elliptical",
        "score_lenticular",
        "score_irregular",
    ]
    score_to_label = {
        "score_spiral": "Spiral",
        "score_elliptical": "Elliptical",
        "score_lenticular": "Lenticular",
        "score_irregular": "Irregular",
    }

    top_score_col = frame[score_cols].idxmax(axis=1)
    frame["label"] = top_score_col.map(score_to_label)
    frame["confidence"] = frame[score_cols].max(axis=1)
    frame["label_idx"] = frame["label"].map(
        {"Spiral": 0, "Elliptical": 1, "Lenticular": 2, "Irregular": 3}
    )

    return frame


def stratified_split(
    frame: pd.DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> pd.DataFrame:
    if round(train_split + val_split + test_split, 6) != 1.0:
        raise ValueError("train_split + val_split + test_split must be 1.0")

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


def load_sources(mapping_path: Path, metadata_path: Path, images_dir: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path, dtype={"objid": str, "asset_id": str})
    metadata = pd.read_csv(metadata_path, compression="infer", dtype={"dr7objid": str})

    if "objid" not in mapping.columns or "asset_id" not in mapping.columns:
        raise ValueError("Mapping file must contain columns: objid, asset_id")
    if "dr7objid" not in metadata.columns:
        raise ValueError("Metadata file must contain column: dr7objid")

    mapping = mapping.rename(columns={"objid": "object_id"})
    metadata = metadata.rename(columns={"dr7objid": "object_id"})

    mapping["filename"] = mapping["asset_id"].astype(str) + ".jpg"
    mapping["image_path"] = mapping["filename"].apply(lambda name: str(images_dir / name))

    merged = mapping.merge(metadata, on="object_id", how="inner", validate="many_to_one")
    return merged


def main(config_file: str) -> None:
    cfg = load_yaml(resolve_path(config_file))

    dataset_cfg = cfg.get("dataset", {})
    build_cfg = cfg.get("build", {})

    images_dir = resolve_path(dataset_cfg["images_dir"])
    mapping_path = resolve_path(dataset_cfg["mapping_file"])
    metadata_path = resolve_path(dataset_cfg["metadata_file"])
    output_path = resolve_path(dataset_cfg["final_dataset_file"])

    min_confidence = float(build_cfg.get("min_confidence", 0.40))
    train_split = float(build_cfg.get("train_split", 0.70))
    val_split = float(build_cfg.get("val_split", 0.15))
    test_split = float(build_cfg.get("test_split", 0.15))
    random_seed = int(build_cfg.get("random_seed", 42))

    for path in [images_dir, mapping_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    frame = load_sources(mapping_path, metadata_path, images_dir)

    frame = build_scores(frame)
    frame = assign_label(frame)

    frame = frame[frame["confidence"] >= min_confidence].copy()
    frame = frame[frame["image_path"].map(lambda p: Path(p).exists())].copy()

    if frame.empty:
        raise RuntimeError("No valid samples remained after filtering.")

    # Prefer mapping sample name when both mapping and metadata provide it.
    if "sample_x" in frame.columns:
        frame = frame.rename(columns={"sample_x": "sample"})
    elif "sample_y" in frame.columns:
        frame = frame.rename(columns={"sample_y": "sample"})

    base_cols = [
        "image_path",
        "object_id",
        "asset_id",
        "label",
        "label_idx",
        "confidence",
    ]
    optional_cols = ["sample", "gz2_class", "total_votes"]
    selected_cols = base_cols + [c for c in optional_cols if c in frame.columns]

    manifest = frame[selected_cols].copy()

    manifest = stratified_split(
        manifest,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=random_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print(f"Saved unified dataset: {output_path}")
    print(f"Total samples: {len(manifest):,}")
    print("Class counts:")
    print(manifest["label"].value_counts().to_string())
    print("Split counts:")
    print(manifest["split"].value_counts().to_string())


def parse_args() -> Tuple[str]:
    parser = argparse.ArgumentParser(description="Build unified GalaxyMorph dataset manifest")
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
