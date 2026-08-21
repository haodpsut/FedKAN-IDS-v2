"""Giao thuc long nhau tren CA BON O, khong con la ket qua mot o.

VI SAO. Phan bien mo phong 20/08: ket luan headline -0,54 pp chi do tren MOT o
(BoT-IoT), trong khi Bang IX cung bai bao +6,23 pp tren ToN-IoT voi ca hai ben da
duoc chinh, va o do giao thuc long nhau CHUA TUNG chay. "Khong co khac biet do
duoc" khi do la mot phat bieu ve BoT-IoT.

Chang 0.1 da chay 120 run bu vao: moi o con lai, hai kien truc, tai lr DA KHOA tu
10 seed cu, tren 20 seed 101-120 chua tung nhin.

GIAO THUC, viet ra de khoi phai doan:
  1. Chon lr cho tung kien truc bang trung binh 5 vong cuoi tren 10 seed CU.
  2. KHOA lai. Khong nhin seed moi trong buoc nay.
  3. Bao cao hieu ghep cap tren 20 seed MOI.
Buoc 2 la thu phan biet giao thuc nay voi ba giao thuc kia trong Bang X.
"""
from __future__ import annotations
import csv
import glob
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = {11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026}
NEW = set(range(101, 121))
GRID = ["0p0003", "0p001", "0p003", "0p01", "0p03", "0p1"]
CELLS = [("lrsweep_botiot", "NF-BoT-IoT-v2 (bin.)"),
         ("lrsweep_toniot", "NF-ToN-IoT-v2 (bin.)"),
         ("lrsweep_cseciic", "NF-CSE-CIC (bin.)"),
         ("lrsweep_cseciic_mc50k", "NF-CSE-CIC (multi.)")]


def m5(d):
    a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
    return sum(a[-5:]) / 5


def cell(pre, arch, lr, seeds):
    o = {}
    for d in glob.glob(str(ROOT / ("results/%s/lrsweep_%s_lr%s__dir0.1__seed*" % (pre, arch, lr)))):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in seeds:
            continue
        try:
            o[int(m.group(1))] = m5(d)
        except Exception:
            pass
    return o


def best_lr(pre, arch):
    c = [(st.mean(cell(pre, arch, lr, OLD).values()), lr) for lr in GRID if cell(pre, arch, lr, OLD)]
    return max(c)[1] if c else None


def paired_p(d):
    n = len(d)
    if n < 2:
        return float("nan")
    s = st.stdev(d)
    if s == 0:
        return 0.0
    t = abs(st.mean(d)) / (s / n ** 0.5)
    df = n - 1
    x = df / (df + t * t)

    def betacf(a, b, x):
        c, d_ = 1.0, 1.0 - (a + b) * x / (a + 1.0)
        d_ = 1.0 / (d_ if abs(d_) > 1e-300 else 1e-300)
        h = d_
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
            d_ = 1.0 / (1.0 + aa * d_ if abs(1.0 + aa * d_) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            h *= d_ * c
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
            d_ = 1.0 / (1.0 + aa * d_ if abs(1.0 + aa * d_) > 1e-300 else 1e-300)
            c = 1.0 + aa / c
            de = d_ * c
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


def boot_ci(d, n=20000, seed=12345):
    """Bootstrap ghep cap. Sinh so bang LCG co hat co dinh: ket qua lap lai duoc
    va khong phu thuoc phien ban numpy."""
    s = seed
    N = len(d)
    means = []
    for _ in range(n):
        acc = 0.0
        for _ in range(N):
            s = (1103515245 * s + 12345) % (1 << 31)
            # LAY BIT CAO. Bit thap cua LCG co chu ky rat ngan: "s % N" voi N=20
            # cho day lap gan tuan hoan chu khong phai mau ngau nhien, va CI tinh
            # tu do khong dung. Dich 16 bit roi moi lay du.
            acc += d[(s >> 16) % N]
        means.append(acc / N)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    print("=" * 92)
    print("  GIAO THUC LONG NHAU TREN CA BON O")
    print("  chon lr tren 10 seed CU -> khoa -> bao cao tren 20 seed 101-120 CHUA NHIN")
    print("=" * 92)
    print("  %-22s %-14s %8s %9s %8s %-18s %s"
          % ("o", "lr kan/mlp", "n", "hieu TB", "p", "CI 95%", "KAN thang"))
    print("  " + "-" * 90)
    rows = []
    for pre, name in CELLS:
        lk, lm = best_lr(pre, "kan8"), best_lr(pre, "mlp80")
        k, m = cell(pre, "kan8", lk, NEW), cell(pre, "mlp80", lm, NEW)
        sp = sorted(set(k) & set(m))
        if not sp:
            print("  %-22s (chua co du lieu seed moi)" % name)
            continue
        d = [100 * (k[s] - m[s]) for s in sp]
        lo, hi = boot_ci(d)
        win = sum(1 for x in d if x > 0)
        rows.append((name, st.mean(d), paired_p(d), len(d)))
        print("  %-22s %-14s %8d %+9.2f %8.3f  [%+6.2f,%+6.2f]  %2d/%d"
              % (name, "%s/%s" % (lk.replace("p", "."), lm.replace("p", ".")),
                 len(d), st.mean(d), paired_p(d), lo, hi, win, len(d)))

    print()
    if len(rows) == 4:
        vals = [r[1] for r in rows]
        sig = [r for r in rows if r[2] < 0.05]
        print("  Dai qua bon o : %+.2f den %+.2f pp" % (min(vals), max(vals)))
        print("  Dau            : %d o duong, %d o am" % (sum(v > 0 for v in vals), sum(v <= 0 for v in vals)))
        print("  Co y nghia p<.05: %d/4  %s" % (len(sig), [r[0] for r in sig]))
        print()
        print("  DOC THE NAO. Neu ca bon o deu khong co y nghia thi ket luan 'khong do duoc")
        print("  khac biet' la phat bieu ve CA BON o, khong con la ket qua mot o. Neu co o")
        print("  nao van duong ro rang thi bai PHAI noi ro dieu do thay vi khai quat hoa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
