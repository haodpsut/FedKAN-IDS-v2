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
    kdir = Path.home() / ".kaggle"
    legacy = kdir / "kaggle.json"
    new_tok = kdir / "access_token"
    has_env_tok = bool(os.environ.get("KAGGLE_API_TOKEN"))
    if not (legacy.exists() or new_tok.exists() or has_env_tok):
        sys.exit(
            f"\n[prepare_data] No Kaggle credentials found.\n"
            "Provide ONE of:\n"
            f"  - {legacy}  (legacy format with username + key)\n"
            f"  - {new_tok}  (new format: KGAT_... single-line token)\n"
            "  - environment variable KAGGLE_API_TOKEN=KGAT_...\n"
            "On Colab use cell A1 in notebooks/10_run_batch.ipynb."
        )
    print(f"[prepare_data] kaggle datasets download -d {slug} -p {dest} --unzip")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)


_METADATA_KEYWORDS = ("feature", "readme", "metadata", "description",
                      "license", "schema", "columns", "_info")
_MIN_DATA_BYTES = 1_000_000  # files smaller than 1 MB are almost certainly metadata


def find_data_file(dest: Path, expected: str | None = None) -> Path:
    """Locate the actual dataset file in `dest`.

    Strategy:
      1. If `expected` filename is provided and exists → use it.
      2. Otherwise scan for parquet (preferred) then csv, skipping files
         whose name suggests documentation/metadata or whose size is below
         _MIN_DATA_BYTES.
      3. Pick the largest survivor.
      4. On failure, raise listing every file under `dest` for debugging.
    """
    if expected:
        cand = dest / expected
        if cand.exists():
            return cand
        for m in dest.rglob(expected):
            return m

    candidates: list[Path] = []
    for pattern in ("*.parquet", "*.csv"):
        for f in dest.rglob(pattern):
            name_lower = f.name.lower()
            if any(kw in name_lower for kw in _METADATA_KEYWORDS):
                continue
            try:
                if f.stat().st_size < _MIN_DATA_BYTES:
                    continue
            except OSError:
                continue
            candidates.append(f)
        if candidates:
            break  # parquet wins; only fall back to csv if no parquet

    if not candidates:
        listing = "\n".join(
            f"  {p.relative_to(dest)}  ({p.stat().st_size:,} B)"
            for p in sorted(dest.rglob("*")) if p.is_file()
        )
        raise FileNotFoundError(
            f"No data file found under {dest} (after skipping metadata < {_MIN_DATA_BYTES:,} B). "
            f"Files present:\n{listing}"
        )

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    chosen = candidates[0]
    print(f"[prepare_data] picked data file: {chosen.relative_to(dest)} "
          f"({chosen.stat().st_size:,} B)")
    return chosen


def prepare(name: str, force: bool) -> None:
    cfg = load_config(ROOT / "configs" / "datasets" / f"{name}.yaml")
    parq, _ = cache_paths(name)
    if parq.exists() and not force:
        print(f"[prepare_data] cache exists: {parq} (skip; pass --force to rebuild)")
        return

    raw_dir = RAW_DIR / name
    expected = cfg["source"].get("expected_csv")
    pre_existing = list(raw_dir.rglob(expected)) if expected else []

    if not pre_existing:
        if cfg["source"]["primary"] != "kaggle":
            sys.exit(f"Only kaggle source is wired up; got {cfg['source']}")
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            kaggle_download(cfg["source"]["kaggle_slug"], raw_dir)
        else:
            print(f"[prepare_data] {raw_dir} already populated, skipping download")
        data_path = find_data_file(raw_dir, expected)
    else:
        data_path = pre_existing[0]
        print(f"[prepare_data] reuse existing raw file: {data_path}")

    print(f"[prepare_data] preprocess {data_path} ...")
    df, meta = preprocess_raw_csv(data_path, cfg)
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
