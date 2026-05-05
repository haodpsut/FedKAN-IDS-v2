"""One-shot data preparation: download (Kaggle) → preprocess → parquet cache.

Usage on Colab:
    # 1) Drop kaggle.json into ~/.kaggle/ once (see notebook cell)
    # 2) Run for each dataset:
    python scripts/prepare_data.py --dataset nf_botiot_v2
    python scripts/prepare_data.py --dataset nf_toniot_v2
    python scripts/prepare_data.py --dataset nf_cseciic_v2

Files land in:
    data/raw/<name>/      # original CSV (kept for traceability)
    data/cache/<name>.parquet
    data/cache/<name>.meta.yaml
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402
from src.datasets.netflow_v2 import (  # noqa: E402
    RAW_DIR,
    cache_paths,
    preprocess_raw_csv,
    write_cache,
)


def kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    home = Path.home() / ".kaggle" / "kaggle.json"
    if not home.exists():
        sys.exit(
            f"\n[prepare_data] kaggle.json missing at {home}.\n"
            "On Colab, run cell 'A1' in notebooks/10_run_batch.ipynb to copy "
            "it from your Drive, or upload it manually:\n"
            "    !mkdir -p ~/.kaggle && cp <path>/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
        )
    print(f"[prepare_data] kaggle datasets download -d {slug} -p {dest} --unzip")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)


def find_csv(dest: Path, expected: str) -> Path:
    cand = dest / expected
    if cand.exists():
        return cand
    matches = list(dest.rglob(expected))
    if matches:
        return matches[0]
    csvs = list(dest.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV under {dest} after Kaggle unzip; got: {list(dest.iterdir())}"
        )
    print(f"[prepare_data] expected '{expected}' missing, "
          f"using first CSV found: {csvs[0].name}")
    return csvs[0]


def prepare(name: str, force: bool) -> None:
    cfg = load_config(ROOT / "configs" / "datasets" / f"{name}.yaml")
    parq, _ = cache_paths(name)
    if parq.exists() and not force:
        print(f"[prepare_data] cache exists: {parq} (skip; pass --force to rebuild)")
        return

    raw_dir = RAW_DIR / name
    csv_name = cfg["source"].get("expected_csv")
    csv_in_raw = list(raw_dir.rglob(csv_name)) if csv_name else []

    if not csv_in_raw:
        if cfg["source"]["primary"] != "kaggle":
            sys.exit(f"Only kaggle source is wired up; got {cfg['source']}")
        kaggle_download(cfg["source"]["kaggle_slug"], raw_dir)
        csv_path = find_csv(raw_dir, csv_name)
    else:
        csv_path = csv_in_raw[0]
        print(f"[prepare_data] reuse existing raw CSV: {csv_path}")

    print(f"[prepare_data] preprocess {csv_path} ...")
    df, meta = preprocess_raw_csv(csv_path, cfg)
    out = write_cache(name, df, meta)
    print(f"[prepare_data] wrote {out}  ({meta['n_samples']:,} rows, "
          f"{len(meta['feature_names'])} features, "
          f"{len(meta['class_names'])} classes)")
    print(f"[prepare_data] class_names = {meta['class_names']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["nf_botiot_v2", "nf_toniot_v2", "nf_cseciic_v2"])
    p.add_argument("--force", action="store_true",
                   help="Re-download and re-preprocess even if cache exists.")
    args = p.parse_args()
    prepare(args.dataset, args.force)


if __name__ == "__main__":
    main()
