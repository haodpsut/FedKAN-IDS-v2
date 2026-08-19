"""R1#5 va R2#4: chi phi suy dien tren phan cung thuc, khong phai tren RTX 4090.

Ca hai phan bien doi cung mot thu. R1: "support Raspberry Pi statements with
experimental evidence or present them more cautiously". R2: "RTX 4090 times aren't
representative of IoT deployment".

Chung ta KHONG co Raspberry Pi, va bia dat mot con so cho no thi te hon la khong
co. Nen script nay lam ba viec do duoc, va bai phai noi ro no do cai gi:

  1. Suy dien tren CPU, MOT LUONG, batch 1. Do la che do gan gateway IoT nhat ma
     may nay mo phong duoc: khong GPU, khong song song hoa theo lo.
  2. Dem FLOP va so tham so, hai dai luong KHONG phu thuoc phan cung, nen doc gia
     co the quy sang phan cung cua ho.
  3. Bo nho mo hinh, thu quyet dinh mot mo hinh co nap duoc vao thiet bi hay khong.

DIEU KIEN LANH MANH: in kem thoi gian cua MOT PHEP TOAN CHUAN (nhan ma tran co
dinh) tren cung may. Neu doc gia biet may cua ho cham hon may nay bao nhieu lan o
phep chuan do, ho quy doi duoc. Khong co neo do thi con so mili giay khong mang
thong tin gi ra ngoai may nay.

Con so tren 4090 cua ban da nop KHONG bi xoa: no van dung, chi la no tra loi mot
cau hoi khac (huan luyen tap trung tren may chu), va bai phai noi ro cau hoi nao.
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model, count_params  # noqa: E402

CONFIGS = [
    ("FedKAN-8",       dict(name="kan", hidden=[8],  grid_size=5, spline_order=3)),
    ("F-KAN-16",       dict(name="kan", hidden=[16], grid_size=5, spline_order=3)),
    ("FedAvg-MLP-32",  dict(name="mlp", hidden=[32])),
    ("FedAvg-MLP-80",  dict(name="mlp", hidden=[80])),
]


def bench(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), st.stdev(ts) if len(ts) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dim", type=int, default=39)
    ap.add_argument("--out-dim", type=int, default=2)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="results/hw_profile.json")
    args = ap.parse_args()

    torch.set_num_threads(1)          # mot luong: che do gateway IoT
    dev = torch.device("cpu")

    # neo quy doi: nhan ma tran 256x256, phep toan ai cung do lai duoc
    A = torch.randn(256, 256)
    B = torch.randn(256, 256)
    anchor, _ = bench(lambda: A @ B, reps=50)

    print("=" * 90)
    print("R1#5 / R2#4: CHI PHI SUY DIEN TREN CPU MOT LUONG, batch 1")
    print("=" * 90)
    print("  may       : %s, %s" % (platform.machine(), platform.processor() or platform.system()))
    print("  torch     : %s | so luong: 1 | thiet bi: CPU" % torch.__version__)
    print("  NEO quy doi: nhan ma tran 256x256 mat %.3f ms tren may nay." % (anchor * 1e3))
    print("              Doc gia do lai phep nay tren gateway cua ho de quy doi cac so duoi.")
    print()
    print("  %-16s %9s %12s %14s %13s" % ("mo hinh", "tham so", "bo nho (kB)",
                                          "1 mau (ms)", "1000 mau (ms)"))
    print("  " + "-" * 72)

    rows = []
    for label, mcfg in CONFIGS:
        m = build_model(mcfg, in_dim=args.in_dim, out_dim=args.out_dim).to(dev).eval()
        n_par = count_params(m)
        mem_kb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024
        x1 = torch.randn(1, args.in_dim)
        xb = torch.randn(1000, args.in_dim)
        with torch.no_grad():
            t1, s1 = bench(lambda: m(x1), reps=args.reps)
            tb, _ = bench(lambda: m(xb), reps=max(20, args.reps // 10))
        print("  %-16s %9d %12.1f %11.4f%s %13.3f" % (
            label, n_par, mem_kb, t1 * 1e3,
            ("+-%.4f" % (s1 * 1e3)).rjust(0), tb * 1e3))
        rows.append({"model": label, "params": n_par, "mem_kb": mem_kb,
                     "ms_1": t1 * 1e3, "ms_1000": tb * 1e3,
                     "anchor_matmul_ms": anchor * 1e3})

    print()
    kan = next(r for r in rows if r["model"] == "FedKAN-8")
    mlp = next(r for r in rows if r["model"] == "FedAvg-MLP-80")
    print("  FedKAN-8 so voi MLP-PM-80 o tham so tuong duong (%d vs %d):"
          % (kan["params"], mlp["params"]))
    print("    suy dien 1 mau : KAN cham hon %.2f lan" % (kan["ms_1"] / mlp["ms_1"]))
    print("    suy dien lo 1000: KAN cham hon %.2f lan" % (kan["ms_1000"] / mlp["ms_1000"]))
    print()
    print("  Doc the nao: 'tham so tuong duong' KHONG co nghia la 'chi phi tuong duong'.")
    print("  Moi canh KAN phai danh gia B-spline, dat hon mot phep nhan cong don.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"machine": platform.machine(), "torch": torch.__version__,
               "threads": 1, "anchor_matmul_256_ms": anchor * 1e3, "rows": rows},
              open(args.out, "w"), indent=2)
    print("\n  da ghi %s" % args.out)


if __name__ == "__main__":
    main()
