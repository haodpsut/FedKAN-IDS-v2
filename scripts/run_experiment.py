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


def _parse_hidden(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-root", default=None,
                   help="Override the per-run output directory root.")
    p.add_argument("--exp-id", default=None,
                   help="Override experiment.id (used in the run_id).")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--mode", choices=["binary", "multiclass"], default=None,
                   help="Override data.mode for NetFlow datasets.")
    p.add_argument("--downsample", type=int, default=None,
                   help="Override data.downsample_per_class.")
    p.add_argument("--model-name", choices=["kan", "mlp"], default=None,
                   help="Override model.name.")
    p.add_argument("--hidden", default=None,
                   help="Override model.hidden, e.g. '8' or '16,16'.")
    p.add_argument("--grid-size", type=int, default=None,
                   help="KAN grid size.")
    p.add_argument("--spline-order", type=int, default=None,
                   help="KAN B-spline order.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Do not re-run if metrics.json already exists.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.exp_id is not None:
        cfg.setdefault("experiment", {})["id"] = args.exp_id
    if args.partition is not None:
        cfg["data"]["partition"] = args.partition
    if args.alpha is not None:
        cfg["data"]["alpha"] = args.alpha
    if args.rounds is not None:
        cfg["fl"]["rounds"] = args.rounds
    if args.mode is not None:
        cfg["data"]["mode"] = args.mode
    if args.downsample is not None:
        cfg["data"]["downsample_per_class"] = args.downsample
    if args.model_name is not None:
        cfg["model"]["name"] = args.model_name
    if args.hidden is not None:
        cfg["model"]["hidden"] = _parse_hidden(args.hidden)
    if args.grid_size is not None:
        cfg["model"]["grid_size"] = args.grid_size
    if args.spline_order is not None:
        cfg["model"]["spline_order"] = args.spline_order

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))

    out_root = Path(args.output_root or cfg["output"]["dir"])
    run_dir = out_root / run_id_for(cfg, seed)
    if args.skip_existing and (run_dir / "metrics.json").exists():
        print(f"[run] SKIP (already done): {run_dir}/metrics.json")
        return
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
        "model_name": cfg["model"]["name"],
        "model_hidden": cfg["model"].get("hidden"),
        "model_grid_size": cfg["model"].get("grid_size"),
        "model_spline_order": cfg["model"].get("spline_order"),
        "data_mode": cfg["data"].get("mode", "binary"),
        "data_partition": cfg["data"].get("partition", "dirichlet"),
        "data_alpha": cfg["data"].get("alpha"),
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
