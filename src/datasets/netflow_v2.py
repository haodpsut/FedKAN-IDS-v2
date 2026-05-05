"""Loaders for the standardised NetFlow-v2 IDS datasets (Sarhan et al. 2022).

Three datasets share the same schema and therefore one loader:
    - NF-BoT-IoT-v2
    - NF-ToN-IoT-v2
    - NF-CSE-CIC-IDS2018-v2

Pipeline:
    raw CSV  ──▶  drop ID columns  ──▶  encode multiclass label
                ──▶  Min-Max scale features  ──▶  parquet cache.

The expensive cleanup runs once via `scripts/prepare_data.py`; downstream
training reads only the parquet cache.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CACHE_DIR = Path("data/cache")
RAW_DIR = Path("data/raw")
META_SUFFIX = ".meta.yaml"


@dataclass
class NetflowDataset:
    X: np.ndarray             # float32, shape (N, F), already scaled to [0,1]
    y_binary: np.ndarray      # int64, shape (N,)
    y_multiclass: np.ndarray  # int64, shape (N,)
    feature_names: list[str]
    class_names: list[str]    # multiclass label decoder; class_names[y_mc[i]] == name

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes_binary(self) -> int:
        return 2

    @property
    def n_classes_multiclass(self) -> int:
        return len(self.class_names)


def cache_paths(name: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{name}.parquet", CACHE_DIR / f"{name}{META_SUFFIX}"


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".csv", ".tsv"):
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(f"Unsupported extension {suf} for {path}")


def preprocess_raw_csv(csv_path: Path, ds_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply Sarhan-style cleanup; returns (parquet-ready df, meta dict)."""
    df = _read_table(csv_path)
    print(f"[netflow_v2] loaded {csv_path.name}: shape={df.shape}, "
          f"first cols={list(df.columns)[:8]}")
    pp = ds_cfg["preprocess"]

    drop = [c for c in pp.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop)

    bin_col = pp["binary_label_column"]
    mc_col = pp["multiclass_label_column"]
    if bin_col not in df.columns or mc_col not in df.columns:
        raise ValueError(
            f"Expected label columns {bin_col!r} and {mc_col!r} in {csv_path}; "
            f"all columns: {list(df.columns)}"
        )

    # Numeric features (drop labels).
    feature_cols = [c for c in df.columns if c not in (bin_col, mc_col)]
    # Some columns can be object dtype with stray strings; coerce numerics.
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # Min-Max scale per column to [0, 1].
    if pp.get("scaler", "minmax") == "minmax":
        col_min = df[feature_cols].min().to_numpy(dtype=np.float32)
        col_max = df[feature_cols].max().to_numpy(dtype=np.float32)
        denom = np.where((col_max - col_min) < 1e-9, 1.0, col_max - col_min)
        df[feature_cols] = (df[feature_cols].to_numpy() - col_min) / denom

    # Encode multiclass label.
    class_names = sorted(df[mc_col].astype(str).unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    df["__y_multiclass"] = df[mc_col].astype(str).map(cls_to_idx).astype(np.int64)
    df["__y_binary"] = df[bin_col].astype(np.int64)

    keep_cols = feature_cols + ["__y_binary", "__y_multiclass"]
    out = df[keep_cols].copy()

    meta = {
        "feature_names": feature_cols,
        "class_names": class_names,
        "n_samples": int(len(out)),
        "binary_class_balance": out["__y_binary"].value_counts().to_dict(),
        "multiclass_balance": out["__y_multiclass"].value_counts().to_dict(),
    }
    return out, meta


def write_cache(name: str, df: pd.DataFrame, meta: dict) -> Path:
    parq, meta_path = cache_paths(name)
    parq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parq, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        # Cast non-yaml-friendly numpy ints to plain ints.
        m = {**meta,
             "binary_class_balance": {int(k): int(v) for k, v in meta["binary_class_balance"].items()},
             "multiclass_balance": {int(k): int(v) for k, v in meta["multiclass_balance"].items()}}
        yaml.safe_dump(m, f, sort_keys=False)
    return parq


def load_cached(name: str) -> NetflowDataset:
    parq, meta_path = cache_paths(name)
    if not parq.exists():
        raise FileNotFoundError(
            f"Parquet cache missing for '{name}' at {parq}. "
            f"Run: python scripts/prepare_data.py --dataset {name}"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    df = pd.read_parquet(parq)
    feat = meta["feature_names"]
    return NetflowDataset(
        X=df[feat].to_numpy(dtype=np.float32),
        y_binary=df["__y_binary"].to_numpy(dtype=np.int64),
        y_multiclass=df["__y_multiclass"].to_numpy(dtype=np.int64),
        feature_names=feat,
        class_names=meta["class_names"],
    )


def maybe_downsample(
    X: np.ndarray, y: np.ndarray, target_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random downsample any class above target_per_class; smaller classes untouched."""
    rng = np.random.RandomState(seed)
    keep_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > target_per_class:
            idx = rng.choice(idx, target_per_class, replace=False)
        keep_idx.append(idx)
    out = np.concatenate(keep_idx)
    rng.shuffle(out)
    return X[out], y[out]
