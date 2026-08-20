"""Chon learning rate tren SEED GIU RIENG, roi bao cao tren seed con lai.

Nguoi doc ngoai (19/08) xep cach lam cu vao THIEN LECH CHON: lr tot nhat cua moi
kien truc duoc chon bang cach so trung binh tren CHINH 10 seed dung de bao cao.
Lam vay thoi phong ca hai ben, va thoi phong ben nao trai rong hon tren luoi thi
nhieu hon. Do dung la KAN o day.

Giao thuc sach: chia 10 seed thanh
    TUNE = 4 seed dau (theo thu tu so hoc, co dinh truoc)
    TEST = 6 seed con lai
Chon lr tren TUNE, KHOA lai, bao cao tren TEST. Khong nhin TEST lan nao truoc khi khoa.

DIEU KIEN LANH MANH: in ca ba con so
    (a) chon tren toan bo 10 seed, bao cao tren 10 seed  <- cach cu, THIEN LECH
    (b) chon tren TUNE, bao cao tren TEST                <- cach sach
    (c) chenh lech (a) tru (b)                           <- do lon cua thien lech
Neu (c) nho thi cach cu tuy khong sach nhung khong lam sai ket luan, va noi duoc
nhu vay. Neu (c) lon thi ket luan cu phai bo.
"""
from __future__ import annotations
import argparse
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026]
TUNE = SEEDS[:4]                       # 11, 17, 23, 29
TEST = SEEDS[4:]                       # 31, 37, 42, 43, 2024, 2026


def load(root):
    acc = defaultdict(dict)
    for d in glob.glob(str(ROOT / root / "*")):
        m = re.search(r"lrsweep_(\w+?)_lr([\dpm]+)__dir[\d.]+__seed(\d+)", d)
        if not m:
            continue
        lr = float(m.group(2).replace("p", ".").replace("m", "-"))
        try:
            a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
        except Exception:
            continue
        acc[(m.group(1), lr)][int(m.group(3))] = sum(a[-5:]) / 5
    return acc


def best_lr(acc, arch, seeds):
    cand = [(st.mean(acc[(arch, lr)][s] for s in seeds if s in acc[(arch, lr)]), lr)
            for lr in sorted({k[1] for k in acc}) if acc.get((arch, lr))]
    return max(cand)[1] if cand else None


def gap(acc, ka, kl, ma, ml, seeds):
    s = [x for x in seeds if x in acc[(ka, kl)] and x in acc[(ma, ml)]]
    d = [100 * (acc[(ka, kl)][x] - acc[(ma, ml)][x]) for x in s]
    if len(s) < 3:
        return None
    _, p = stats.ttest_rel([acc[(ka, kl)][x] for x in s], [acc[(ma, ml)][x] for x in s])
    return st.mean(d), p, len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="botiot,toniot,cseciic,cseciic_mc50k")
    args = ap.parse_args()

    print("=" * 92)
    print("  CHON lr TREN SEED GIU RIENG  (tune = %s | test = %s)" % (TUNE, TEST))
    print("=" * 92)
    print("  %-14s %-26s %11s %11s %9s" % ("o", "cach", "lr KAN/MLP", "hieu TB", "p"))
    print("  " + "-" * 82)
    bias = []
    for cell in args.cells.split(","):
        acc = load("results/lrsweep_" + cell)
        if not acc:
            continue
        # (a) cach cu: chon va bao cao deu tren 10 seed
        ka, ma = best_lr(acc, "kan8", SEEDS), best_lr(acc, "mlp80", SEEDS)
        ga = gap(acc, "kan8", ka, "mlp80", ma, SEEDS)
        # (b) cach sach: chon tren TUNE, bao cao tren TEST
        kb, mb = best_lr(acc, "kan8", TUNE), best_lr(acc, "mlp80", TUNE)
        gb = gap(acc, "kan8", kb, "mlp80", mb, TEST)
        if not ga or not gb:
            continue
        print("  %-14s %-26s %5g/%-5g %+10.3f %9.4f" % (cell, "(a) chon+bao cao tren 10", ka, ma, ga[0], ga[1]))
        print("  %-14s %-26s %5g/%-5g %+10.3f %9.4f" % ("", "(b) chon tren 4, bao cao 6", kb, mb, gb[0], gb[1]))
        print("  %-14s %-26s %11s %+10.3f" % ("", "(c) thien lech (a)-(b)", "", ga[0] - gb[0]))
        bias.append(ga[0] - gb[0])
        print()
    if bias:
        print("  Thien lech trung binh tren %d o: %+.3f pp (lon nhat %+.3f)"
              % (len(bias), st.mean(bias), max(bias, key=abs)))
        print()
        print("  DOC THE NAO: neu thien lech nho so voi hieu ung thi cach cu khong lam sai ket")
        print("  luan, va bai duoc noi vay KEM SO. Neu no cung bac voi hieu ung thi moi con so")
        print("  'da do lr' trong bai deu phai doc lai theo cot (b).")


if __name__ == "__main__":
    main()
