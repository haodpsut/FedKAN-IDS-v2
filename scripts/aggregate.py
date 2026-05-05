"""Aggregate results/runs/*/metrics.json into a tidy CSV + print a summary table.

Output:
    results/aggregated/summary.csv  (one row per run)
    results/aggregated/summary_by_cell.csv  (mean/std across seeds)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import statistics as stats

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "aggregated"


def parse_run_id(rid: str) -> dict:
    """e1_mini_botiot_kan_binary__dir0.1__seed42 ->
       dict(exp=..., partition=..., alpha=..., seed=...)
    """
    parts = rid.split("__")
    if len(parts) < 3:
        return {"exp": rid, "partition": "?", "alpha": None, "seed": None}
    exp = parts[0]
    ptag = parts[1]
    seed = int(parts[2].replace("seed", ""))
    if ptag == "iid":
        partition, alpha = "iid", None
    elif ptag.startswith("dir"):
        partition = "dirichlet"
        try:
            alpha = float(ptag[3:])
        except ValueError:
            alpha = None
    else:
        partition, alpha = ptag, None
    return {"exp": exp, "partition": partition, "alpha": alpha, "seed": seed}


def load_run(run_dir: Path) -> dict | None:
    mj = run_dir / "metrics.json"
    cs = run_dir / "config_snapshot.yaml"
    if not mj.exists():
        return None
    metrics = json.loads(mj.read_text())
    cfg = yaml.safe_load(cs.read_text()) if cs.exists() else {}
    fm = metrics.get("final_metrics", {})
    return {
        "run_id": metrics["run_id"],
        **parse_run_id(metrics["run_id"]),
        "model": cfg.get("model", {}).get("name", "?"),
        "n_params": metrics.get("n_params"),
        "rounds": metrics.get("rounds"),
        "final_accuracy": fm.get("accuracy"),
        "final_f1_macro": fm.get("f1_macro"),
        "final_f1_weighted": fm.get("f1_weighted"),
        "best_accuracy": metrics.get("best_accuracy"),
        "bytes_per_round_uplink_per_client": metrics.get("bytes_per_round_uplink"),
        "total_uplink_bytes": metrics.get("total_uplink_bytes"),
    }


def main():
    runs = [load_run(d) for d in sorted(RUNS.iterdir())
            if d.is_dir() and not d.name.startswith("smoke")]
    runs = [r for r in runs if r is not None]
    if not runs:
        sys.exit(f"No runs under {RUNS}")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Per-run table.
    cols = list(runs[0].keys())
    with open(OUT / "summary.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in runs:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    # 2. Aggregate by (exp, partition, alpha, model) — mean/std over seeds.
    cells: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r["model"], r["partition"], r["alpha"])
        cells.setdefault(key, []).append(r)

    with open(OUT / "summary_by_cell.csv", "w", encoding="utf-8") as f:
        f.write("model,partition,alpha,n_seeds,n_params,acc_mean,acc_std,"
                "f1m_mean,f1m_std,total_mb_mean\n")
        for key, rs in sorted(cells.items()):
            accs = [r["final_accuracy"] for r in rs if r["final_accuracy"] is not None]
            f1ms = [r["final_f1_macro"] for r in rs if r["final_f1_macro"] is not None]
            mbs = [(r["total_uplink_bytes"] or 0) / 1024 / 1024 for r in rs]
            n = len(rs)
            f.write(",".join([
                key[0], key[1],
                "" if key[2] is None else f"{key[2]}",
                str(n),
                str(rs[0]["n_params"] or ""),
                f"{stats.mean(accs):.4f}" if accs else "",
                f"{stats.stdev(accs):.4f}" if len(accs) > 1 else "0",
                f"{stats.mean(f1ms):.4f}" if f1ms else "",
                f"{stats.stdev(f1ms):.4f}" if len(f1ms) > 1 else "0",
                f"{stats.mean(mbs):.2f}" if mbs else "",
            ]) + "\n")

    # 3. Console pretty-print (ASCII-only for Windows cp1252 consoles).
    print(f"\nResults aggregated from {len(runs)} runs.\n")
    print(f"{'Model':6s} {'Partition':12s} {'alpha':>6s} {'Seeds':>5s} "
          f"{'Params':>8s} {'Acc (mean+/-std) %':>22s} {'F1m (mean+/-std) %':>22s} "
          f"{'Uplink (MB)':>14s}")
    print("-" * 110)
    for key, rs in sorted(cells.items()):
        accs = [r["final_accuracy"] for r in rs if r["final_accuracy"] is not None]
        f1ms = [r["final_f1_macro"] for r in rs if r["final_f1_macro"] is not None]
        mbs = [(r["total_uplink_bytes"] or 0) / 1024 / 1024 for r in rs]
        a_mean = stats.mean(accs) if accs else 0
        a_std = stats.stdev(accs) if len(accs) > 1 else 0
        f_mean = stats.mean(f1ms) if f1ms else 0
        f_std = stats.stdev(f1ms) if len(f1ms) > 1 else 0
        m_mean = stats.mean(mbs) if mbs else 0
        alpha_str = "-" if key[2] is None else f"{key[2]}"
        print(f"{key[0]:6s} {key[1]:12s} {alpha_str:>6s} {len(rs):>5d} "
              f"{rs[0]['n_params'] or 0:>8d} "
              f"{a_mean*100:>10.4f} +/- {a_std*100:>6.4f}  "
              f"{f_mean*100:>10.4f} +/- {f_std*100:>6.4f}  "
              f"{m_mean:>12.2f}")
    print()
    print(f"Wrote {OUT/'summary.csv'} and {OUT/'summary_by_cell.csv'}")


if __name__ == "__main__":
    main()
