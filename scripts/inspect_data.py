"""Sanity-check a parquet cache produced by scripts/prepare_data.py.

Usage:
    python scripts/inspect_data.py --dataset nf_botiot_v2

Prints:
  - shape, feature count
  - binary label distribution
  - multiclass label distribution + class name decoder
  - per-feature: min/max/mean/std/% zero/% NaN
  - flags suspicious conditions (e.g. 'binary labels are degenerate ALL ZERO').

This is the diagnostic to run when a model converges to acc=0.5 on a binary
task — most likely the binary label column came out all-0 or all-1.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.netflow_v2 import cache_paths, RAW_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["nf_botiot_v2", "nf_toniot_v2", "nf_cseciic_v2"])
    p.add_argument("--n-features", type=int, default=10,
                   help="How many feature stats to print")
    args = p.parse_args()

    parq, meta_path = cache_paths(args.dataset)
    if not parq.exists():
        sys.exit(f"Cache missing: {parq}\nRun: python scripts/prepare_data.py --dataset {args.dataset}")

    df = pd.read_parquet(parq)
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}

    print(f"\n=== {args.dataset} ===")
    print(f"shape: {df.shape}")
    print(f"feature count: {len(meta.get('feature_names', []))}")
    print()

    # Binary label distribution
    if "__y_binary" in df.columns:
        vc = df["__y_binary"].value_counts().sort_index()
        print("BINARY label distribution (__y_binary):")
        for k, v in vc.items():
            print(f"  class {int(k)}: {v:>12,} ({v / len(df) * 100:5.2f}%)")
        if len(vc) < 2:
            print(f"  WARNING: only {len(vc)} unique binary value(s). "
                  "Models will trivially predict the majority class.")
        elif vc.min() / vc.sum() < 0.005:
            print(f"  WARNING: minority class is < 0.5% of data — "
                  "extreme imbalance, may converge to majority predictions.")
    else:
        print("ERROR: __y_binary column missing from parquet!")

    # Multi-class label distribution
    if "__y_multiclass" in df.columns:
        print("\nMULTICLASS label distribution (__y_multiclass):")
        vc = df["__y_multiclass"].value_counts().sort_index()
        names = meta.get("class_names", [])
        for k, v in vc.items():
            name = names[int(k)] if int(k) < len(names) else f"class_{k}"
            print(f"  {int(k)} ({name}): {v:>12,} ({v / len(df) * 100:5.2f}%)")
    else:
        print("ERROR: __y_multiclass column missing!")

    # Feature stats
    feature_cols = meta.get("feature_names") or [c for c in df.columns
                                                  if not c.startswith("__y_")]
    print(f"\nFEATURE STATS (first {args.n_features} of {len(feature_cols)}):")
    print(f"{'col':<30s} {'min':>10s} {'max':>10s} {'mean':>10s} "
          f"{'std':>10s} {'zero%':>7s} {'nan%':>7s}")
    print("-" * 90)
    flags = []
    for c in feature_cols[: args.n_features]:
        col = df[c]
        nzero = (col == 0).sum()
        nnan = col.isna().sum()
        print(f"{c[:29]:<30s} {col.min():>10.4g} {col.max():>10.4g} "
              f"{col.mean():>10.4g} {col.std():>10.4g} "
              f"{nzero/len(col)*100:>6.1f}% {nnan/len(col)*100:>6.1f}%")
        if col.min() == col.max():
            flags.append(f"  - constant column: {c}")
        if nnan / len(col) > 0.5:
            flags.append(f"  - >50% NaN in column: {c}")
    if flags:
        print("\nFLAGGED COLUMNS:")
        for f in flags:
            print(f)

    # Diagnose acc=0.5 root cause
    print("\nDIAGNOSIS:")
    if "__y_binary" in df.columns:
        nu = df["__y_binary"].nunique()
        if nu == 1:
            print("  >>> ROOT CAUSE LIKELY: binary labels are degenerate "
                  "(all the same value). The Label column may have been "
                  "renamed in this Kaggle mirror or the binary encoding "
                  "in netflow_v2.py needs adjustment.")
        elif nu == 2 and df["__y_binary"].dtype != np.int64:
            print("  >>> binary labels are non-int64; check dtype.")
        else:
            print("  >>> binary labels look OK (2 distinct values, both populated).")

    # Show what the original CSV-side might still hold
    raw_dir = RAW_DIR / args.dataset
    if raw_dir.exists():
        any_csv = list(raw_dir.rglob("*.csv")) + list(raw_dir.rglob("*.parquet"))
        any_csv = [c for c in any_csv if "feature" not in c.name.lower()
                   and c.stat().st_size > 1_000_000]
        if any_csv:
            sample = any_csv[0]
            print(f"\nORIGINAL FILE COLUMNS (peek at {sample.name}):")
            try:
                if sample.suffix == ".parquet":
                    cols = list(pd.read_parquet(sample, columns=None).columns[:50])
                else:
                    cols = list(pd.read_csv(sample, nrows=1).columns[:50])
                for c in cols:
                    print(f"  {c}")
            except Exception as e:
                print(f"  (could not read: {e})")


if __name__ == "__main__":
    main()
