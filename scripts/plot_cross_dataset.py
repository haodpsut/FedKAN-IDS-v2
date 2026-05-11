"""Three new IoT-J-grade figures from the 3-dataset run set.

Fig A: results/figures/cross_dataset_convergence.pdf
   3-column grid: BoT-IoT | ToN-IoT | CSE-CIC. Each panel: convergence curves
   for 4 variants under binary Dir(alpha=0.1), seed-mean +/- 1 std band.

Fig B: results/figures/skew_advantage_gradient.pdf
   Bar chart of KAN-8 advantage (pp) over MLP-PM-80 per dataset, ordered by
   class-distribution skew. Shows BOTH mean-accuracy and worst-seed gaps.

Fig C: results/figures/seed_distribution.pdf
   3-column strip plot of per-seed binary Dir(0.1) accuracy. Each dot is one
   seed. Outliers (e.g. seed 17 on ToN-IoT) are visible at a glance.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"


mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

VARIANT_ORDER = ["mlp_h32", "mlp_h80", "kan_h8", "kan_h16"]
VARIANT_LABEL = {
    "mlp_h32":  "FedAvg-MLP (32h)",
    "mlp_h80":  "FedAvg-MLP-PM (80h)",
    "kan_h8":   "FedKAN (Ours, 8h)",
    "kan_h16":  "F-KAN (16h)",
}
VARIANT_COLOR = {
    "mlp_h32":  "#1f77b4",
    "mlp_h80":  "#9467bd",
    "kan_h8":   "#d62728",
    "kan_h16":  "#2ca02c",
}
VARIANT_MARKER = {
    "mlp_h32":  "o",
    "mlp_h80":  "^",
    "kan_h8":   "s",
    "kan_h16":  "D",
}
DATASET_ORDER = ["botiot", "toniot", "cseciic"]
DATASET_TITLE = {
    "botiot":  "NF-BoT-IoT-v2 (high skew)",
    "toniot":  "NF-ToN-IoT-v2 (mid skew)",
    "cseciic": "NF-CSE-CIC-IDS2018-v2 (low skew)",
}
DATASET_PANEL = {"botiot": "(a)", "toniot": "(b)", "cseciic": "(c)"}


def variant_of(metrics: dict) -> str | None:
    name = metrics.get("model_name")
    h = metrics.get("model_hidden")
    if not name or not h:
        return None
    return f"{name}_h{'x'.join(str(x) for x in h)}"


def dataset_of(run_dir_name: str) -> str | None:
    if not run_dir_name.startswith("e1_") or run_dir_name.startswith("e1_mini"):
        return None
    return run_dir_name[3:].split("_", 1)[0]


def gather_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text())
        if m.get("data_partition") != "dirichlet" or m.get("data_alpha") != 0.1:
            continue
        if m.get("data_mode") != "binary":
            continue
        ds = dataset_of(d.name)
        v = variant_of(m)
        if ds is None or v is None:
            continue
        per_round = d / "per_round.csv"
        if not per_round.exists():
            continue
        m["__df"] = pd.read_csv(per_round)
        m["__dataset"] = ds
        m["__variant"] = v
        out.append(m)
    return out


def fig_convergence(runs: list[dict]):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    n_per_panel: list[int] = []
    for ax, ds in zip(axes, DATASET_ORDER):
        ds_runs = [r for r in runs if r["__dataset"] == ds]
        if not ds_runs:
            ax.set_title(f"{DATASET_PANEL[ds]} {DATASET_TITLE[ds]} (pending)",
                         fontsize=10)
            n_per_panel.append(0)
            continue
        n_here = 0
        for v in VARIANT_ORDER:
            v_runs = [r for r in ds_runs if r["__variant"] == v]
            if not v_runs:
                continue
            n_here = max(n_here, len(v_runs))
            # Truncate to the common min length in case any run went past 50 rounds.
            min_len = min(len(r["__df"]) for r in v_runs)
            rounds = v_runs[0]["__df"]["round"].to_numpy()[:min_len]
            accs = np.stack([r["__df"]["accuracy"].to_numpy()[:min_len]
                             for r in v_runs], 0)
            mean = accs.mean(0)
            std = accs.std(0)
            ax.plot(rounds, mean, color=VARIANT_COLOR[v], marker=VARIANT_MARKER[v],
                    markevery=5, label=VARIANT_LABEL[v])
            ax.fill_between(rounds, mean - std, mean + std,
                            color=VARIANT_COLOR[v], alpha=0.15)
        n_per_panel.append(n_here)
        ax.set_xlabel("Communication Round")
        ax.set_title(f"{DATASET_PANEL[ds]} {DATASET_TITLE[ds]}  "
                     fr"$(n{{=}}{n_here})$", fontsize=10)
        ax.set_ylim(0.4, 1.005)
    axes[0].set_ylabel("Global Accuracy")
    axes[-1].legend(loc="lower right", framealpha=0.95, fontsize=8)

    out = OUT / "cross_dataset_convergence.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_gradient(runs: list[dict]):
    """KAN-8 - MLP-PM-80 advantage per dataset (mean + worst-seed)."""
    by = {(r["__dataset"], r["__variant"]):
              [(r2["final_metrics"]["accuracy"]) for r2 in runs
               if r2["__dataset"] == r["__dataset"]
               and r2["__variant"] == r["__variant"]]
          for r in runs}

    mean_gap = []
    worst_gap = []
    dataset_labels = []
    n_pairs = []
    for ds in DATASET_ORDER:
        ka = by.get((ds, "kan_h8"), [])
        mp = by.get((ds, "mlp_h80"), [])
        if not ka or not mp:
            continue
        # Pair by seed: same number of seeds usually, but be defensive.
        n = min(len(ka), len(mp))
        mean_gap.append((np.mean(ka) - np.mean(mp)) * 100)
        worst_gap.append((min(ka) - min(mp)) * 100)
        dataset_labels.append(DATASET_TITLE[ds].replace(" (", "\n("))
        n_pairs.append(n)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(dataset_labels))
    w = 0.38
    bars1 = ax.bar(x - w / 2, mean_gap, w, color="#d62728",
                   edgecolor="black", linewidth=0.6,
                   label=r"Mean-seed advantage")
    bars2 = ax.bar(x + w / 2, worst_gap, w, color="#fdcb6e",
                   edgecolor="black", linewidth=0.6,
                   label=r"Worst-seed advantage")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=9)
    ax.set_ylabel("FedKAN-8 advantage\nover MLP-PM-80 (pp)")
    ax.set_title("Cross-dataset accuracy gap vs class-distribution skew\n"
                 r"(binary Dir($\alpha=0.1$); positive = KAN better)",
                 fontsize=10)
    ax.legend(loc="upper right", framealpha=0.95)

    # Annotate bars
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            yoff = 0.5 if h >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width() / 2, h + yoff,
                    f"{h:+.1f}", ha="center", va=va, fontsize=8.5)

    # Annotate n
    ymin = ax.get_ylim()[0]
    for xi, n in zip(x, n_pairs):
        ax.text(xi, ymin * 0.92, f"$n_{{KAN}}{{=}}{n}$",
                ha="center", fontsize=7.5, color="gray")

    out = OUT / "skew_advantage_gradient.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_seed_distribution(runs: list[dict]):
    """Strip plot of per-seed accuracy, faceted by dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    for ax, ds in zip(axes, DATASET_ORDER):
        ds_runs = [r for r in runs if r["__dataset"] == ds]
        if not ds_runs:
            ax.set_title(f"{DATASET_PANEL[ds]} {DATASET_TITLE[ds]} (pending)",
                         fontsize=10)
            ax.set_xticks([])
            continue
        for i, v in enumerate(VARIANT_ORDER):
            v_runs = [r for r in ds_runs if r["__variant"] == v]
            if not v_runs:
                continue
            accs = np.array([r["final_metrics"]["accuracy"] for r in v_runs]) * 100
            rng = np.random.RandomState(0)
            xs = rng.normal(i, 0.06, len(accs))
            ax.scatter(xs, accs, s=60, alpha=0.85,
                       color=VARIANT_COLOR[v], edgecolor="black", linewidth=0.4,
                       marker=VARIANT_MARKER[v])
            # Mean line
            mean = accs.mean()
            ax.hlines(mean, i - 0.22, i + 0.22, color="black", lw=1.6)
            # Worst-seed marker
            worst = accs.min()
            ax.hlines(worst, i - 0.16, i + 0.16, color="red", lw=1.0,
                      linestyle=":")
        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels(["MLP-32", "MLP-PM-80", "KAN-8", "F-KAN-16"],
                           rotation=18, ha="right", fontsize=8.5)
        ax.set_title(f"{DATASET_PANEL[ds]} {DATASET_TITLE[ds]}", fontsize=10)
        ax.set_ylim(40, 102)

    axes[0].set_ylabel("Global Accuracy (\\%)")
    # Manual legend for the line markers
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", lw=1.6, label="seed mean"),
        Line2D([0], [0], color="red", lw=1.0, ls=":", label="worst seed"),
    ]
    axes[-1].legend(handles=handles, loc="lower right", framealpha=0.95,
                    fontsize=8)
    out = OUT / "seed_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = gather_runs()
    if not runs:
        print("no e1_* binary Dir(0.1) runs found")
        return
    print(f"loaded {len(runs)} runs")
    fig_convergence(runs)
    fig_gradient(runs)
    fig_seed_distribution(runs)


if __name__ == "__main__":
    main()
