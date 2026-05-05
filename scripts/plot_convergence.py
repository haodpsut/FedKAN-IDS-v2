"""IEEE-style convergence plots from results/runs/*/per_round.csv.

Output: results/figures/convergence_<mode>_<partition>.pdf
        results/figures/convergence_<mode>_grid.pdf

Each curve is the seed-mean with a +/- 1 std shaded band. Variants
(kan_h8, mlp_h32, mlp_h80, kan_h16, ...) are read from metrics.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


# Stable visual mapping; new variants get a fallback colour/marker.
VARIANT_STYLE: dict[str, dict] = {
    "kan_h8":     dict(color="#d62728", marker="s", label="FedKAN (Ours, 8h)"),
    "mlp_h32":    dict(color="#1f77b4", marker="o", label="FedAvg-MLP (32h)"),
    "mlp_h80":    dict(color="#9467bd", marker="^", label="FedAvg-MLP-PM (80h)"),
    "kan_h16":    dict(color="#2ca02c", marker="D", label="F-KAN (16h)"),
    "kan_h16x16": dict(color="#ff7f0e", marker="v", label="F-KAN (16x16)"),
}
_FALLBACK = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _style_for(variant: str, idx: int) -> dict:
    if variant in VARIANT_STYLE:
        return VARIANT_STYLE[variant]
    return dict(color=_FALLBACK[idx % len(_FALLBACK)],
                marker="x", label=variant)


def parse(rid: str):
    parts = rid.split("__")
    if len(parts) < 3:
        return None
    seed = int(parts[2].replace("seed", ""))
    return parts[1], seed  # ptag, seed


def variant_of(run_dir: Path) -> str | None:
    mj = run_dir / "metrics.json"
    if not mj.exists():
        return None
    m = json.loads(mj.read_text())
    model = m.get("model_name")
    hidden = m.get("model_hidden")
    if model and hidden:
        return f"{model}_h{'x'.join(str(h) for h in hidden)}"
    return None


def gather() -> dict:
    """{(mode, partition_tag, variant): [DataFrame seed1, DataFrame seed2, ...]}"""
    bucket: dict = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        meta = parse(d.name)
        if meta is None:
            continue
        ptag, _seed = meta
        v = variant_of(d)
        if v is None:
            continue
        m = json.loads((d / "metrics.json").read_text())
        mode = m.get("data_mode", "binary")
        df = pd.read_csv(d / "per_round.csv")
        bucket.setdefault((mode, ptag, v), []).append(df)
    return bucket


def panel(ax, dfs_by_variant: dict, title: str):
    for i, (variant, dfs) in enumerate(sorted(dfs_by_variant.items())):
        if not dfs:
            continue
        rounds = dfs[0]["round"].to_numpy()
        accs = np.stack([d["accuracy"].to_numpy() for d in dfs], 0)
        mean = accs.mean(0); std = accs.std(0)
        st = _style_for(variant, i)
        ax.plot(rounds, mean, color=st["color"], marker=st["marker"],
                markevery=5, label=f"{st['label']} (n={len(dfs)})")
        ax.fill_between(rounds, mean - std, mean + std,
                        color=st["color"], alpha=0.15)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.005)
    ax.legend(loc="lower right", framealpha=0.95)


PARTITION_TITLE = {
    "iid":     "(a) IID",
    "dir1.0":  r"(b) Dir($\alpha=1.0$)",
    "dir0.5":  r"(c) Dir($\alpha=0.5$)",
    "dir0.1":  r"(d) Dir($\alpha=0.1$)",
    "dir0.05": r"(e) Dir($\alpha=0.05$)",
}
PARTITION_ORDER = ["iid", "dir1.0", "dir0.5", "dir0.1", "dir0.05"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bucket = gather()
    if not bucket:
        sys.exit("No e1 runs found.")

    modes = sorted({k[0] for k in bucket.keys()})
    for mode in modes:
        ptags = [p for p in PARTITION_ORDER
                 if any((m, p, _) in bucket for m, p, _ in bucket if m == mode)]
        ptags = [p for p in PARTITION_ORDER if any(k for k in bucket
                                                    if k[0] == mode and k[1] == p)]
        if not ptags:
            continue

        # one panel per partition
        for ptag in ptags:
            dfs_by_v = {v: dfs for (m, p, v), dfs in bucket.items()
                        if m == mode and p == ptag}
            if not dfs_by_v:
                continue
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            title = f"{mode.capitalize()} — " + (
                "IID" if ptag == "iid" else fr"Dirichlet $\alpha={ptag[3:]}$"
            )
            panel(ax, dfs_by_v, title)
            out = OUT / f"convergence_{mode}_{ptag}.pdf"
            fig.savefig(out); plt.close(fig)
            print(f"wrote {out}")

        # grid panel for the mode
        ncols = len(ptags)
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 3.5),
                                 sharey=True)
        if ncols == 1:
            axes = [axes]
        for ax, ptag in zip(axes, ptags):
            dfs_by_v = {v: dfs for (m, p, v), dfs in bucket.items()
                        if m == mode and p == ptag}
            panel(ax, dfs_by_v, PARTITION_TITLE.get(ptag, ptag))
        for ax in axes[1:]:
            ax.set_ylabel("")
        out = OUT / f"convergence_{mode}_grid.pdf"
        fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
