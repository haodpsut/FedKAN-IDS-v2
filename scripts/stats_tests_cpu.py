"""Sinh lai ba bang thong ke tu run CPU, thay ban sinh tu results/runs (RTX 4090).

VI SAO. QA ngay 20/08/2026 phat hien Bang IV, VI, VII van sinh tu `results/runs`,
noi 310/310 run ghi `device: cuda`, trong khi §VI-E cua bai khang dinh "moi thi
nghiem trong ban sua nay chay tren MOT may, Apple M5, CPU". Cau do SAI voi ba bang.
Chung con mau thuan SO voi cac bang moi: Bang IV in +6,00 con Bang IX in +5,49 cho
CUNG mot o; Bang VII in +1,73 kem dau sao y nghia trong khi §V-F cua chinh bai noi
con so do "tinh tu mot vong duy nhat" va da bi rut.

Cong `verify_handtyped_tables.py` mu cho nay vi no chi kiem bang GO TAY; ba bang
tren vao bai qua \\input nen duoc mac dinh tin.

THUOC DO: trung binh 5 vong cuoi, dong bo voi phan con lai cua ban sua. Ban cu dung
VONG CUOI va khong khai o dau, do la mot trong bon quyet dinh phan tich lam con so
headline chay tu +6,12 xuong -0,54.

NGUON DU LIEU cho tung o, khai tuong minh de khong thua ke im lang:
  nhi phan   -> results/lrsweep_<bo>/  tai lr chung 0p01, 10 seed cu
  da lop     -> results/mc_<bo>/       chang 0.2, lr 0p01, 10 seed cu
               (rieng CSE-CIC da lop dung lrsweep_cseciic_mc50k, da co san)
"""
from __future__ import annotations
import csv
import glob
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
OLD = {11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026}

CELLS = [
    ("botiot", "NF-BoT-IoT-v2", "stats_tests_botiot", "tab:stats_botiot",
     "results/lrsweep_botiot/lrsweep_%s_lr0p01__dir0.1__seed*",
     "results/mc_botiot/mc_%s_lr0p01__dir0.1__seed*"),
    ("toniot", "NF-ToN-IoT-v2", "stats_tests_toniot", "tab:stats_toniot",
     "results/lrsweep_toniot/lrsweep_%s_lr0p01__dir0.1__seed*",
     "results/mc_toniot/mc_%s_lr0p01__dir0.1__seed*"),
    ("cseciic", "NF-CSE-CIC-IDS2018-v2", "stats_tests_cseciic", "tab:stats_cseciic",
     "results/lrsweep_cseciic/lrsweep_%s_lr0p01__dir0.1__seed*",
     "results/lrsweep_cseciic_mc50k/lrsweep_%s_lr0p01__dir0.1__seed*"),
]


def m5(d):
    a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
    return sum(a[-5:]) / 5


def load(pat, arch):
    o, dev = {}, set()
    for d in glob.glob(str(ROOT / (pat % arch))):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in OLD:
            continue
        try:
            o[int(m.group(1))] = m5(d)
            dev.add(json.load(open(d + "/metrics.json")).get("device", "?"))
        except Exception:
            pass
    return o, dev


def paired_p(d):
    n = len(d)
    s = st.stdev(d)
    if n < 2 or s == 0:
        return float("nan")
    t = abs(st.mean(d)) / (s / n ** 0.5)
    df, x = n - 1, (n - 1) / ((n - 1) + t * t)

    def betacf(a, b, x):
        c, dd = 1.0, 1.0 - (a + b) * x / (a + 1.0)
        dd = 1.0 / (dd if abs(dd) > 1e-300 else 1e-300)
        h = dd
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
            dd = 1.0 / (1.0 + aa * dd if abs(1.0 + aa * dd) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            h *= dd * c
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
            dd = 1.0 / (1.0 + aa * dd if abs(1.0 + aa * dd) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            de = dd * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return math.exp(lb) * betacf(a, b, x) / a
        return 1.0 - math.exp(lb) * betacf(b, a, 1 - x) / b

    return betai(df / 2.0, 0.5, x)


def boot(d, n=20000, seed=20260820):
    s, N, out = seed, len(d), []
    for _ in range(n):
        acc = 0.0
        for _ in range(N):
            s = (1103515245 * s + 12345) % (1 << 31)
            acc += d[(s >> 16) % N]      # bit CAO: bit thap cua LCG co chu ky rat ngan
        out.append(acc / N)
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def stars(p):
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def main():
    """MOT bang cho ca ba bo du lieu.

    Truoc 21/08 day la BA bang table* rieng, moi bang hai dong, tuc ba lan tieu de
    + ba lan chu thich cho sau dong so lieu. Bai dang 18 trang va can ve 14; gop lai
    tiet kiem gan hai trang ma khong bo mot con so nao.
    """
    devs = set()
    body = []
    for key, disp, fname, label, pat_bin, pat_mc in CELLS:
        first = True
        for mode, pat in (("Binary", pat_bin), ("Multiclass", pat_mc)):
            k, dk = load(pat, "kan8")
            m, dm = load(pat, "mlp80")
            devs |= dk | dm
            sp = sorted(set(k) & set(m))
            if not sp:
                continue
            kk = [100 * k[s] for s in sp]
            mm = [100 * m[s] for s in sp]
            d = [a - b for a, b in zip(kk, mm)]
            lo, hi = boot(d)
            p = paired_p(d)
            pooled = math.sqrt((st.variance(kk) + st.variance(mm)) / 2) or 1e-9
            body.append("%s & %s & %d & %.1f\\,$\\pm$\\,%.1f & %.1f\\,$\\pm$\\,%.1f "
                        "& %+.2f & [%+.1f,\\,%+.1f] & $%+.2f$ & $%.3f%s$ \\\\"
                        % (disp if first else "", mode, len(sp),
                           st.mean(kk), st.stdev(kk), st.mean(mm), st.stdev(mm),
                           st.mean(d), lo, hi, st.mean(d) / pooled, p, stars(p)))
            first = False
        body.append("\\addlinespace[2pt]")
    L = ["%% Auto-generated by scripts/stats_tests_cpu.py -- run CPU, TB 5 vong cuoi",
         "\\begin{table*}[t]", "\\centering",
         ("\\caption{Statistical comparison across the three datasets: FedKAN-8 against the "
          "parameter-matched FedAvg-MLP-PM-80, at the shared $\\eta=10^{-2}$ of the original "
          "protocol, under Dir($\\alpha{=}0.1$). Accuracy is the \\emph{mean over the final five "
          "communication rounds}, in \\%%; $\\Delta$ is the seed-paired mean difference "
          "(KAN $-$ MLP), CI its percentile-bootstrap 95\\%% interval, $d$ Cohen's effect size and "
          "$p$ the paired $t$-test. All runs were executed on the same machine (%s). "
          "Markers: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}" % (sorted(devs) or ["?"])[0]),
         "\\label{tab:stats_all}",
         "\\renewcommand{\\arraystretch}{1.15}", "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize",
         "\\begin{tabular}{@{}l l c c c c c c c@{}}", "\\toprule",
         "Dataset & Mode & $n$ & KAN-8 & MLP-PM-80 & $\\Delta$ (pp) & 95\\% CI (pp) & $d$ & $p$ \\\\",
         "\\midrule"] + body[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (OUT / "stats_tests_all.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    for f in ("stats_tests_botiot", "stats_tests_toniot", "stats_tests_cseciic"):
        q = OUT / (f + ".tex")
        if q.exists():
            q.unlink()
    print("  da ghi results/tables/stats_tests_all.tex (gop 3 bang table* thanh 1)")
    if len(devs) > 1:
        print("  ⛔ CANH BAO: run den tu NHIEU thiet bi: %s" % sorted(devs))
        return 1
    print("  ✅ moi run tu MOT thiet bi: %s" % (sorted(devs) or ["?"])[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
