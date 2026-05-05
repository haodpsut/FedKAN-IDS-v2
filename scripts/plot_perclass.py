"""Per-class F1 grouped bar chart for multiclass runs (especially Dir(alpha=0.1)).

Reveals which classes (e.g. rare 'Theft' in NF-BoT-IoT-v2) each model handles.
Output: results/figures/perclass_f1_multiclass_<ptag>.pdf
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

VARIANT_ORDER = ["mlp_h32", "mlp_h80", "kan_h8", "kan_h16"]
VARIANT_LABEL = {
    "mlp_h32": "MLP-32",
    "mlp_h80": "MLP-PM (80)",
    "kan_h8":  "KAN-8 (Ours)",
    "kan_h16": "F-KAN-16",
}
VARIANT_COLOR = {
    "mlp_h32": "#1f77b4",
    "mlp_h80": "#9467bd",
    "kan_h8":  "#d62728",
    "kan_h16": "#2ca02c",
}

# Class names from prepare_data meta — sorted alphabetically there.
CLASS_NAMES = ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"]


def variant_of(metrics: dict) -> str | None:
    m = metrics.get("model_name")
    h = metrics.get("model_hidden")
    if not m or not h:
        return None
    return f"{m}_h{'x'.join(str(x) for x in h)}"


def gather_perclass(mode: str, ptag: str) -> dict:
    """{variant: {class_idx: [f1 across seeds]}}"""
    bucket: dict = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        parts = d.name.split("__")
        if len(parts) < 3 or parts[1] != ptag:
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text())
        if m.get("data_mode") != mode:
            continue
        v = variant_of(m)
        if v is None:
            continue
        f1s = m.get("final_metrics", {}).get("per_class_f1") or []
        if not f1s:
            continue
        slot = bucket.setdefault(v, {})
        for ci, f in enumerate(f1s):
            slot.setdefault(ci, []).append(float(f))
    return bucket


def plot(mode: str, ptag: str):
    data = gather_perclass(mode, ptag)
    if not data:
        return False

    # Identify class set actually present (from any variant).
    n_classes = max(max(slot.keys()) for slot in data.values()) + 1
    class_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"c{i}"
                    for i in range(n_classes)]

    variants_here = [v for v in VARIANT_ORDER if v in data]
    n = len(variants_here)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(n_classes)
    for i, v in enumerate(variants_here):
        means = []
        stds = []
        for ci in range(n_classes):
            vals = data[v].get(ci, [])
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        offsets = (i - (n - 1) / 2) * width
        ax.bar(x + offsets, means, width, yerr=stds,
               color=VARIANT_COLOR[v], label=VARIANT_LABEL[v],
               edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=0.7, capsize=2))

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel("Attack class")
    ax.set_ylabel("Per-class F1 (final round)")
    title_p = "IID" if ptag == "iid" else fr"Dirichlet $\alpha={ptag[3:]}$"
    ax.set_title(f"{mode.capitalize()} — {title_p}: per-class F1 (mean ± 1 std)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)
    out = OUT / f"perclass_f1_{mode}_{ptag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    any_done = False
    for mode in ("multiclass",):
        for ptag in ("iid", "dir1.0", "dir0.1"):
            if plot(mode, ptag):
                any_done = True
    if not any_done:
        sys.exit("No multiclass per-class data found.")


if __name__ == "__main__":
    main()
