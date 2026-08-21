"""Sinh lai BON hinh cua muc V tu run CPU, thay ban sinh tu results/runs (RTX 4090).

VI SAO. QA 20/08/2026: sau hinh cua bai thi bon hinh van doc `results/runs`, noi
310/310 run ghi `device: cuda`, trong khi §VI-E khang dinh moi thi nghiem chay tren
MOT may CPU. Sau script ve deu dat RUNS = ROOT/"results"/"runs" va khong doc truong
`device` bao gio, nen cong "mot thiet bi" trong make_tables_r1.py khong voi toi.

THUOC DO dong bo voi phan con lai: trung binh 5 vong cuoi cho moi con so tom tat.
Duong hoi tu ve nguyen ca 50 vong.

Nguon tung hinh, khai tuong minh:
  convergence_binary_grid   <- lowhet/ (IID, Dir1.0) + lrsweep_botiot/ lr0p01 (Dir0.1)
  cross_dataset_convergence <- lrsweep_{botiot,toniot,cseciic}/ lr0p01
  seed_distribution         <- nhu tren, mot cham moi seed
  perclass_f1_multiclass    <- mc_botiot/ (chang 0.2)
"""
from __future__ import annotations
import csv
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OLD = {11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026}
VAR = [("kan8", "kan_h8"), ("kan16", "kan_h16"), ("mlp32", "mlp_h32"), ("mlp80", "mlp_h80")]


def curves(pat):
    """Tra ve {seed: [acc theo vong]} cho mot mau thu muc."""
    o = {}
    for d in glob.glob(str(ROOT / pat)):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in OLD:
            continue
        try:
            o[int(m.group(1))] = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
        except Exception:
            pass
    return o


def band(ax, cur, style):
    if not cur:
        return
    T = min(len(v) for v in cur.values())
    xs = range(1, T + 1)
    mu = [100 * st.mean(v[i] for v in cur.values()) for i in range(T)]
    sd = [100 * (st.stdev([v[i] for v in cur.values()]) if len(cur) > 1 else 0) for i in range(T)]
    ax.plot(xs, mu, color=style["color"], ls=style["ls"], lw=1.7, label=style["label"])
    ax.fill_between(xs, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)],
                    color=style["color"], alpha=0.13, linewidth=0)


def fig_convergence_grid():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7), sharey=True)
    panels = [("IID", "results/lowhet/lowhet_%s__iid__seed*"),
              (r"Dir($\alpha{=}1.0$)", "results/lowhet/lowhet_%s__dir1.0__seed*"),
              (r"Dir($\alpha{=}0.1$)", "results/lrsweep_botiot/lrsweep_%s_lr0p01__dir0.1__seed*")]
    for ax, (title, pat), letter in zip(axes, panels, "abc"):
        n = 0
        for key, var in VAR:
            cur = curves(pat % key)
            n = max(n, len(cur))
            band(ax, cur, plotstyle.style_for(var))
        ax.set_title("(%s) %s   $n{=}%d$" % (letter, title, n), fontsize=12)
        ax.set_xlabel("communication round")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("global accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_binary_grid.pdf")
    plt.close(fig)
    print("  da ghi convergence_binary_grid.pdf")


def fig_cross_dataset():
    sets = [("results/lrsweep_botiot/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-BoT-IoT-v2"),
            ("results/lrsweep_toniot/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-ToN-IoT-v2"),
            ("results/lrsweep_cseciic/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-CSE-CIC-IDS2018-v2")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7), sharey=True)
    for ax, (pat, name) in zip(axes, sets):
        n = 0
        for key, var in VAR:
            cur = curves(pat % key)
            n = max(n, len(cur))
            band(ax, cur, plotstyle.style_for(var))
        ax.set_title("%s   $n{=}%d$" % (name, n), fontsize=11)
        ax.set_xlabel("communication round")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("global accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "cross_dataset_convergence.pdf")
    plt.close(fig)
    print("  da ghi cross_dataset_convergence.pdf")


def fig_seed_strip():
    sets = [("results/lrsweep_botiot/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-BoT-IoT-v2"),
            ("results/lrsweep_toniot/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-ToN-IoT-v2"),
            ("results/lrsweep_cseciic/lrsweep_%s_lr0p01__dir0.1__seed*", "NF-CSE-CIC-IDS2018-v2")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6), sharey=True)
    for ax, (pat, name) in zip(axes, sets):
        for x, (key, var) in enumerate(VAR):
            s = plotstyle.style_for(var)
            cur = curves(pat % key)
            ys = [100 * sum(v[-5:]) / 5 for v in cur.values()]
            ax.scatter([x + 0.0] * len(ys), ys, s=26, color=s["color"], alpha=0.75,
                       edgecolor="black", linewidth=0.4, zorder=3)
            if ys:
                ax.plot([x - 0.22, x + 0.22], [st.mean(ys)] * 2, color="black", lw=1.6, zorder=4)
            # seed 17 duoc danh dau vi §V-F ban ve no
            for sd, v in cur.items():
                if sd == 17:
                    ax.scatter([x], [100 * sum(v[-5:]) / 5], s=78, facecolor="none",
                               edgecolor="#D55E00", linewidth=1.5, zorder=5)
        ax.set_xticks(range(len(VAR)))
        ax.set_xticklabels([plotstyle.style_for(v)["label"].replace(" (", "\n(") for _, v in VAR],
                           fontsize=8)
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0].set_ylabel("accuracy (%), mean of\nfinal five rounds")
    axes[0].scatter([], [], s=78, facecolor="none", edgecolor="#D55E00", linewidth=1.5,
                    label="seed 17")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "seed_distribution.pdf")
    plt.close(fig)
    print("  da ghi seed_distribution.pdf")


def fig_perclass():
    lab, per = None, {}
    for key, var in (("kan8", "kan_h8"), ("mlp80", "mlp_h80")):
        vals = []
        for d in glob.glob(str(ROOT / ("results/mc_botiot/mc_%s_lr0p01__dir0.1__seed*" % key))):
            try:
                f = json.load(open(d + "/metrics.json"))["final_metrics"]["per_class_f1"]
                vals.append(f)
            except Exception:
                pass
        if vals:
            k = min(len(v) for v in vals)
            per[var] = [(st.mean(v[i] for v in vals), st.stdev([v[i] for v in vals]))
                        for i in range(k)]
            lab = ["Benign", "DDoS", "DoS", "Recon.", "Theft"][:k] or [str(i) for i in range(k)]
    if not per:
        print("  ⛔ khong co du lieu per-class")
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    w = 0.38
    for i, (var, rows) in enumerate(per.items()):
        s = plotstyle.style_for(var)
        x = [j + (i - 0.5) * w for j in range(len(rows))]
        ax.bar(x, [r[0] for r in rows], w, yerr=[r[1] for r in rows], capsize=3,
               color=s["color"], edgecolor="black", linewidth=0.5, label=s["label"])
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "perclass_f1_multiclass_dir0.1.pdf")
    plt.close(fig)
    print("  da ghi perclass_f1_multiclass_dir0.1.pdf")
    for var, rows in per.items():
        print("     %-9s %s" % (var, " ".join("%.2f" % r[0] for r in rows)))


def main():
    matplotlib.rcParams.update(plotstyle.RC)
    devs = set()
    for p in list(glob.glob(str(ROOT / "results/lrsweep_botiot/*/metrics.json")))[:40] + \
             list(glob.glob(str(ROOT / "results/lowhet/*/metrics.json")))[:40] + \
             list(glob.glob(str(ROOT / "results/mc_botiot/*/metrics.json")))[:40]:
        try:
            devs.add(json.load(open(p)).get("device", "?"))
        except Exception:
            pass
    print("  thiet bi cua moi nguon: %s" % sorted(devs))
    if devs != {"cpu"}:
        print("  ⛔ nguon khong thuan CPU, DUNG lai")
        return 1
    fig_convergence_grid()
    fig_cross_dataset()
    fig_seed_strip()
    fig_perclass()
    return 0


if __name__ == "__main__":
    sys.exit(main())
