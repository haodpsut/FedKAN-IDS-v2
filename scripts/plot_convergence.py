"""Render IEEE-style convergence plots from results/runs/*/per_round.csv.

Output:
    results/figures/convergence_<partition>.pdf       (per-partition panels)
    results/figures/convergence_grid.pdf              (3x1 grid panel)

Each curve is the seed-mean with a +/- 1 std shaded band.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"


# IEEE-friendly style.
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

MODEL_STYLE = {
    "kan": dict(color="#d62728", marker="s", label="FedKAN"),
    "mlp": dict(color="#1f77b4", marker="o", label="FedAvg-MLP"),
}


def parse(rid: str):
    parts = rid.split("__")
    if len(parts) < 3:
        return None
    exp, ptag, seedtag = parts[0], parts[1], parts[2]
    seed = int(seedtag.replace("seed", ""))
    if "kan" in exp:
        model = "kan"
    elif "mlp" in exp:
        model = "mlp"
    else:
        model = "?"
    return model, ptag, seed


def gather() -> dict:
    """{(model, ptag): [DataFrame seed1, DataFrame seed2, ...]}"""
    bucket: dict = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        meta = parse(d.name)
        if meta is None:
            continue
        model, ptag, _seed = meta
        csv = d / "per_round.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        bucket.setdefault((model, ptag), []).append(df)
    return bucket


def panel(ax, dfs_by_model: dict, title: str):
    for model, dfs in dfs_by_model.items():
        if not dfs:
            continue
        rounds = dfs[0]["round"].to_numpy()
        accs = np.stack([d["accuracy"].to_numpy() for d in dfs], 0)
        mean = accs.mean(0)
        std = accs.std(0)
        st = MODEL_STYLE[model]
        ax.plot(rounds, mean, color=st["color"], marker=st["marker"],
                markevery=5, label=f"{st['label']} (n={len(dfs)})")
        ax.fill_between(rounds, mean - std, mean + std,
                        color=st["color"], alpha=0.18)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.005)
    ax.legend(loc="lower right", framealpha=0.95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bucket = gather()
    if not bucket:
        sys.exit("No e1_mini runs found.")

    # 1) Three single-panel figures (one per partition).
    for ptag in ("iid", "dir1.0", "dir0.1"):
        dfs_by_model = {
            m: bucket.get((m, ptag), []) for m in ("kan", "mlp")
        }
        if not any(dfs_by_model.values()):
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        title = {"iid": "IID partition",
                 "dir1.0": r"Dirichlet $\alpha=1.0$ (mild non-IID)",
                 "dir0.1": r"Dirichlet $\alpha=0.1$ (extreme non-IID)"}[ptag]
        panel(ax, dfs_by_model, title)
        out = OUT / f"convergence_{ptag}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    # 2) 1x3 grid figure.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), sharey=True)
    titles = {
        "iid": "(a) IID",
        "dir1.0": r"(b) Dir($\alpha=1.0$)",
        "dir0.1": r"(c) Dir($\alpha=0.1$)",
    }
    for ax, ptag in zip(axes, ("iid", "dir1.0", "dir0.1")):
        dfs_by_model = {m: bucket.get((m, ptag), []) for m in ("kan", "mlp")}
        panel(ax, dfs_by_model, titles[ptag])
    for ax in axes[1:]:
        ax.set_ylabel("")
    out = OUT / "convergence_grid.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
