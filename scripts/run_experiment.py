"""CLI entry point: run one (config, seed) experiment, save metrics + per-round CSV.

Usage:
    python scripts/run_experiment.py --config configs/experiments/smoke.yaml --seed 42
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `src` importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import (  # noqa: E402
    set_seed,
    get_device,
    load_config,
    save_json,
    save_yaml,
    append_csv_row,
    run_id_for,
)
from src.data import build_federated_split  # noqa: E402
from src.models import build_model, count_params  # noqa: E402
from src.fl import federated_train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-root", default=None,
                   help="Override the per-run output directory root.")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))

    out_root = Path(args.output_root or cfg["output"]["dir"])
    run_dir = out_root / run_id_for(cfg, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config for reproducibility.
    snap = {**cfg, "_resolved": {"seed": seed, "device": str(device)}}
    save_yaml(snap, run_dir / "config_snapshot.yaml")

    print(f"[run] device={device}  seed={seed}  out={run_dir}")

    split = build_federated_split(cfg["data"], seed=seed)

    def factory():
        return build_model(cfg["model"], in_dim=split.n_features,
                           out_dim=split.n_classes)

    n_params = count_params(factory())
    print(f"[run] model={cfg['model']['name']}  params={n_params:,}")

    train_loaders, test_loader = split.loaders(cfg["fl"]["batch_size"])

    csv_path = run_dir / "per_round.csv"

    def on_round_end(round_idx: int, metrics: dict):
        row = {"round": round_idx, **{k: v for k, v in metrics.items()
                                       if not isinstance(v, list)}}
        append_csv_row(row, csv_path)
        print(f"  round {round_idx:>3}/{cfg['fl']['rounds']}: "
              f"acc={metrics['accuracy']:.4f}  "
              f"f1m={metrics['f1_macro']:.4f}  "
              f"uplink_kB={metrics['comm_uplink_bytes']/1024:.1f}  "
              f"t={metrics['wallclock_s']:.2f}s")

    result = federated_train(
        model_factory=factory,
        train_loaders=train_loaders,
        test_loader=test_loader,
        cfg_fl=cfg["fl"],
        device=device,
        on_round_end=on_round_end,
    )

    summary = {
        "run_id": run_id_for(cfg, seed),
        "seed": seed,
        "device": str(device),
        "n_params": n_params,
        "bytes_per_round_uplink": result["bytes_per_round_uplink"],
        "total_uplink_bytes": int(sum(result["history"]["comm_uplink_bytes"])),
        "final_metrics": result["history"]["metrics"][-1],
        "best_accuracy": max(m["accuracy"] for m in result["history"]["metrics"]),
        "rounds": len(result["history"]["rounds"]),
    }
    save_json(summary, run_dir / "metrics.json")
    print(f"[run] DONE -> {run_dir}/metrics.json  "
          f"final_acc={summary['final_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
