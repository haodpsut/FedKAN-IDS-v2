"""Truy so cho BAY bang GO TAY trong bai, doi chieu voi du lieu tren dia.

VI SAO CAN. Bai co 16 bang; 9 bang sinh tu dong qua \\input{results/tables/...},
con 7 bang duoc GO TAY thang vao sections/05_experiments.tex, trong do co dung
bon bang chong do luan diem trung tam: tab:lrsweep, tab:protocols, tab:lrrobust,
tab:pooled. So trong chung do nguoi chep tu dau ra script, nghia la mot lan go
nham song sot qua MOI cong hien co: cong trinh bay chi doc overfull, cong truy so
chi doc results/tables/*.tex, va latexdiff thi khong biet gi ve so.

Day dung la lop loi "ban chep tay khong phai artifact". Script nay tinh lai tung
o tu results/ va IN CANH NHAU voi gia tri dang in trong bai, de nguoi doc so bang
mat thay vi tin.

Chay:  python3 scripts/verify_handtyped_tables.py [--tex ../v2-revision/sections/05_experiments.tex]
"""
from __future__ import annotations
import argparse
import csv
import glob
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = {11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026}
GRID = ["0p0003", "0p001", "0p003", "0p01", "0p03", "0p1"]
NARROW = ["0p003", "0p01", "0p03"]
CELLS = [("lrsweep_botiot", "NF-BoT-IoT-v2 (bin.)"),
         ("lrsweep_toniot", "NF-ToN-IoT-v2 (bin.)"),
         ("lrsweep_cseciic", "NF-CSE-CIC (bin.)"),
         ("lrsweep_cseciic_mc50k", "NF-CSE-CIC (multi.)")]


def mean5(d):
    a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
    return sum(a[-5:]) / 5


def cell(pre, arch, lr):
    o = {}
    for d in glob.glob(str(ROOT / ("results/%s/lrsweep_%s_lr%s__dir0.1__seed*" % (pre, arch, lr)))):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in OLD:
            continue
        try:
            o[int(m.group(1))] = mean5(d)
        except Exception:
            pass
    return o


def best_lr(pre, arch):
    c = [(st.mean(cell(pre, arch, lr).values()), lr) for lr in GRID if cell(pre, arch, lr)]
    return max(c)[1] if c else None


def paired_p(diffs):
    """t ghep cap, hai phia. Tu cai dat de khong keo them phu thuoc."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = st.mean(diffs)
    s = st.stdev(diffs)
    if s == 0:
        return 0.0
    t = abs(m) / (s / n ** 0.5)
    # xap xi hai phia qua phan phoi t, dung tich phan so
    import math
    df = n - 1
    x = df / (df + t * t)
    # I_x(df/2, 1/2) qua chuoi lien phan don gian: dung ham beta khong day du
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d, h = 1.0 / d, 1.0 / d
        for m_ in range(1, MAXIT + 1):
            m2 = 2 * m_
            aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
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


def row(label, typed, got, tol=0.02):
    ok = "  ok " if (isinstance(got, float) and abs(got - typed) <= tol) else "  ⛔ "
    return "%s%-34s bai in %8.2f | tinh lai %8.2f" % (ok, label, typed, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT.parent / "v2-revision/sections/05_experiments.tex"))
    a = ap.parse_args()
    bad = 0

    print("=" * 84)
    print("  TRUY SO CHO BANG GO TAY  (bai in <-> tinh lai tu results/)")
    print("=" * 84)

    print("\n  tab:lrsweep -- khoang cach o lr chung va o lr rieng")
    TYPED_SWEEP = {"NF-BoT-IoT-v2 (bin.)": (5.49, 0.76), "NF-ToN-IoT-v2 (bin.)": (6.23, 6.23),
                   "NF-CSE-CIC (bin.)": (-0.49, 1.12), "NF-CSE-CIC (multi.)": (1.38, 1.38)}
    TYPED_SD = {"NF-BoT-IoT-v2 (bin.)": (2.53, 1.05), "NF-ToN-IoT-v2 (bin.)": (0.71, None),
                "NF-CSE-CIC (bin.)": (0.33, None), "NF-CSE-CIC (multi.)": (0.87, None)}
    pooled = []
    for pre, name in CELLS:
        k0, m0 = cell(pre, "kan8", "0p01"), cell(pre, "mlp80", "0p01")
        sh = 100 * st.mean([k0[s] - m0[s] for s in k0 if s in m0])
        lk, lm = best_lr(pre, "kan8"), best_lr(pre, "mlp80")
        kb, mb = cell(pre, "kan8", lk), cell(pre, "mlp80", lm)
        tu = [kb[s] - mb[s] for s in kb if s in mb]
        pooled += [100 * x for x in tu]
        for tag, typed, got in [("shared", TYPED_SWEEP[name][0], sh),
                                ("tuned", TYPED_SWEEP[name][1], 100 * st.mean(tu))]:
            line = row("%s %s" % (name, tag), typed, got)
            print(line)
            bad += line.startswith("  ⛔")
        r0 = st.stdev(m0.values()) / st.stdev(k0.values())
        line = row("%s sd-ratio shared" % name, TYPED_SD[name][0], r0)
        print(line)
        bad += line.startswith("  ⛔")
        if TYPED_SD[name][1] is not None:
            rt = st.stdev(mb.values()) / st.stdev(kb.values())
            line = row("%s sd-ratio tuned" % name, TYPED_SD[name][1], rt)
            print(line)
            bad += line.startswith("  ⛔")

    print("\n  tab:lrrobust -- do trai accuracy tren dai lr (pp)")
    TYPED_ROB = {"NF-BoT-IoT-v2 (bin.)": (0.65, 3.53, 14.88, 9.49),
                 "NF-ToN-IoT-v2 (bin.)": (1.58, 6.38, 21.59, 17.79),
                 "NF-CSE-CIC (bin.)": (2.53, 6.72, 14.02, 16.52),
                 "NF-CSE-CIC (multi.)": (5.75, 5.94, 45.15, 32.86)}
    for pre, name in CELLS:
        vals = []
        for band in (NARROW, GRID):
            for arch in ("kan8", "mlp80"):
                ms = [st.mean(cell(pre, arch, lr).values()) for lr in band if cell(pre, arch, lr)]
                vals.append(100 * (max(ms) - min(ms)) if ms else float("nan"))
        # thu tu bai in: narrow KAN, narrow MLP, full KAN, full MLP
        got = [vals[0], vals[1], vals[2], vals[3]]
        for lab, typed, g in zip(["narrow KAN", "narrow MLP", "full KAN", "full MLP"],
                                 TYPED_ROB[name], got):
            line = row("%s %s" % (name, lab), typed, g, tol=0.05)
            print(line)
            bad += line.startswith("  ⛔")

    print("\n  tab:pooled -- gop moi cap seed, moi ben o lr tot nhat")
    print("     n cap        : bai in       40 | tinh lai %8d" % len(pooled))
    line = row("mean (seed pair)", 2.37, st.mean(pooled))
    print(line)
    bad += line.startswith("  ⛔")
    p = paired_p(pooled)
    print("     p (seed pair): bai in    0.032 | tinh lai %8.3f" % p)
    bad += abs(p - 0.032) > 0.01
    cellmeans = []
    i = 0
    for pre, name in CELLS:
        n = sum(1 for s in cell(pre, "kan8", best_lr(pre, "kan8"))
                if s in cell(pre, "mlp80", best_lr(pre, "mlp80")))
        cellmeans.append(st.mean(pooled[i:i + n]))
        i += n
    line = row("mean (cell)", 2.37, st.mean(cellmeans))
    print(line)
    bad += line.startswith("  ⛔")
    pc = paired_p(cellmeans)
    print("     p (cell)     : bai in    0.164 | tinh lai %8.3f" % pc)
    bad += abs(pc - 0.164) > 0.02

    print("\n  => %s" % ("FAIL: %d o lech" % bad if bad else "PASS: moi o go tay khop du lieu"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
