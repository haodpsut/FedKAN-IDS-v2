"""Thay hinh 'skew_advantage_gradient' bang mot hinh KHONG khang dinh xu huong.

VI SAO PHAI THAY. Hinh cu (plot_cross_dataset.py, Fig B) co ba loi cung mot luc,
phat hien khi QA ngay 20/08/2026:

  1. No doc `results/runs`, tuc du lieu RTX 4090 cua BAN DA NOP, trong khi bai nay
     tuyen bo moi so deu tu MOT may CPU. Hinh in +6,0 / +5,1 / -0,4 con bai in
     +5,49 / +6,23 / -0,49.
  2. No dung thuoc do VONG CUOI, con bai dung trung binh 5 vong cuoi. Doi thuoc do
     lam BoT-IoT va ToN-IoT DOI CHO, tuc hinh ve dung cai thu tu don dieu ma bai da
     rut o than bai.
  3. Chu thich viet "mean-seed advantage is positive on every dataset" trong khi
     chinh hinh ve -0,4. Chu thich mau thuan voi hinh cua no.

Hinh moi ve DUNG so cua bai, va ve CA HAI thuoc do canh nhau de nguoi doc thay
thu tu doi theo thuoc do. Do la noi dung that su cua muc nay.
"""
from __future__ import annotations
import csv
import glob
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
CELLS = [("lrsweep_botiot", "NF-BoT-IoT-v2"),
         ("lrsweep_toniot", "NF-ToN-IoT-v2"),
         ("lrsweep_cseciic", "NF-CSE-CIC-IDS2018-v2")]


def series(d):
    return [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]


def cell(pre, arch):
    o = {}
    for d in glob.glob(str(ROOT / ("results/%s/lrsweep_%s_lr0p01__dir0.1__seed*" % (pre, arch)))):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in OLD:
            continue
        try:
            a = series(d)
            o[int(m.group(1))] = (sum(a[-5:]) / 5, a[-1])
        except Exception:
            pass
    return o


def main():
    matplotlib.rcParams.update(plotstyle.RC)
    labels, m5, last = [], [], []
    for pre, name in CELLS:
        k, m = cell(pre, "kan8"), cell(pre, "mlp80")
        sp = [s for s in k if s in m]
        if not sp:
            continue
        labels.append(name.replace("-IDS2018", "\n-IDS2018"))
        m5.append(100 * st.mean([k[s][0] - m[s][0] for s in sp]))
        last.append(100 * st.mean([k[s][1] - m[s][1] for s in sp]))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    x = range(len(labels))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], m5, w, color=plotstyle.DIFF_MEAN,
                edgecolor="black", linewidth=0.6, label="mean of final five rounds")
    b2 = ax.bar([i + w / 2 for i in x], last, w, color=plotstyle.DIFF_WORST,
                edgecolor="black", linewidth=0.6, label="final round only")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("FedKAN-8 advantage\nover MLP-PM-80 (pp)")
    ax.set_title("Same runs, two defensible estimators\n"
                 r"(binary Dir($\alpha{=}0.1$), shared $\eta{=}10^{-2}$, $n{=}10$)",
                 fontsize=11)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (0.25 if h >= 0 else -0.25),
                    "%+.2f" % h, ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=9)
    # danh dau dung cho DOI CHO, vi do la noi dung cua hinh
    ax.annotate("order of the first two\nreverses with the estimator",
                xy=(0.5, max(m5[:2]) + 1.4), xytext=(0.62, max(m5) + 4.2),
                fontsize=8.5, ha="center", color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8))
    ax.set_ylim(min(min(m5), min(last)) - 3.2, max(max(m5), max(last)) + 6.0)
    fig.tight_layout()
    out = OUT / "advantage_two_estimators.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("  da ghi %s" % out.relative_to(ROOT))
    for L, a, b in zip(labels, m5, last):
        print("    %-24s TB5 %+6.2f | vong cuoi %+6.2f" % (L.replace("\n", ""), a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
