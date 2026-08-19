"""R2#3 va R2#8: seed 17 hong o dau, va vi sao KAN-16 thoat con KAN-8 thi khong.

Phan bien 2 doi ba thu, va bai hien khong co thu nao:
  (a) phan bo lop VA dac trung tung client cua phan hoach seed 17
  (b) vi sao F-KAN-16 thoat bay ma MLP thi khong
  (c) cac seed khac co bieu hien tuong tu khong

Script nay tra loi (a) va (c) bang do dac; (b) can them phan tich mo hinh nen chi
cung cap DAU VAO cho no.

DIEU KIEN LANH MANH quan trong nhat o day: khong duoc chi nhin seed 17. Neu chi
nhin mot seed thi bat ky bat thuong nao cung trong nhu nguyen nhan. Phai xep hang
CA 10 seed theo cung thang do roi hoi seed 17 dung thu may. Neu no khong phai
cuc doan nhat ma van la seed hong duy nhat thi gia thuyet "phan hoach lech nen
hong" SAI, va phai tim cho khac.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed          # noqa: E402
from src.data import build_federated_split           # noqa: E402


def entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    p = [c / n for c in counts if c > 0]
    return -sum(x * math.log(x, 2) for x in p)


def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if s == 0:
        return 0.0
    return (2 * sum((i + 1) * x for i, x in enumerate(xs)) - (n + 1) * s) / (n * s)


def describe(cfg, seed):
    set_seed(seed)
    split = build_federated_split(cfg["data"], seed=seed)
    # FederatedSplit.client_train la list[(X, y)] nen lay nhan truc tiep, khong
    # phai di vong qua DataLoader.
    ys = [y.tolist() for _, y in split.client_train]
    all_classes = sorted({v for lab in ys for v in lab})
    sizes, ent, frac1 = [], [], []
    for lab in ys:
        c = Counter(lab)
        counts = [c.get(k, 0) for k in all_classes]
        sizes.append(len(lab))
        ent.append(entropy(counts))
        frac1.append(c.get(1, 0) / max(1, len(lab)))
    return {
        "seed": seed,
        "n_clients": len(ys),
        "size_gini": gini(sizes),
        "size_min": min(sizes), "size_max": max(sizes),
        "ent_mean": float(np.mean(ent)), "ent_min": float(np.min(ent)),
        "n_single_class": sum(1 for e in ent if e < 1e-9),
        "frac1_min": float(np.min(frac1)), "frac1_max": float(np.max(frac1)),
        "frac1_spread": float(np.max(frac1) - np.min(frac1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/e1_botiot.yaml")
    ap.add_argument("--dataset", default=None, help="ghi de data.name, vd nf_toniot_v2")
    ap.add_argument("--mode", default="binary")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", default="11,17,23,29,31,37,42,43,2024,2026")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg["data"]["mode"] = args.mode
    cfg["data"]["partition"] = "dirichlet"
    cfg["data"]["alpha"] = args.alpha
    if args.dataset:
        cfg["data"]["name"] = args.dataset

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = [describe(cfg, s) for s in seeds]

    ds = cfg["data"]["name"]
    print("=" * 96)
    print("R2#3/#8: PHAP Y PHAN HOACH  (%s, %s, Dir alpha=%.2g)" % (ds, args.mode, args.alpha))
    print("=" * 96)
    print("  %-7s %9s %9s %9s %10s %11s %11s" % (
        "seed", "Gini co", "co nho", "co lon", "entropy TB", "entropy min", "chenh ty le lop1"))
    print("  " + "-" * 84)
    for r in sorted(rows, key=lambda z: -z["size_gini"]):
        mark = "  <== seed 17" if r["seed"] == 17 else ""
        print("  %-7d %9.4f %9d %9d %10.4f %11.4f %11.4f%s" % (
            r["seed"], r["size_gini"], r["size_min"], r["size_max"],
            r["ent_mean"], r["ent_min"], r["frac1_spread"], mark))
    print()
    for key, lab in [("size_gini", "Gini co client"), ("ent_mean", "entropy trung binh"),
                     ("frac1_spread", "chenh ty le lop 1")]:
        order = sorted(rows, key=lambda z: -z[key])
        rank = [r["seed"] for r in order].index(17) + 1
        print("  seed 17 xep hang %2d/%d theo %s" % (rank, len(rows), lab))
    print()
    print("  DOC BANG NAY THE NAO: neu seed 17 KHONG dung dau o thang do nao thi gia thuyet")
    print("  \"phan hoach lech nen KAN-8 hong\" chua duoc chung minh, va bai khong duoc noi")
    print("  la da giai thich duoc co che. Xep hang la phep thu, khong phai trang tri.")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print("\n  da ghi %s" % args.out)


if __name__ == "__main__":
    main()
