"""R2#5: do nhay cua KAN theo G, bac spline k, va do rong lop an.

Phan bien 2: "Test variations in Grid size G in {3,5,10}, spline order k in {3,4,5},
hidden width beyond 8 and 16". Ban da nop chi chay G=5, k=3, do rong 8 va 16, va
tu chon do lam mac dinh ma khong cho thay lua chon do co quan trong khong.

BA TRUC DO RIENG, KHONG QUET DAY DU. Quet day du 3x3x4 = 36 cau hinh x 10 seed =
360 run cho MOT tap, va phan lon o do khong ai hoi toi. Thay vao do: giu hai truc
o mac dinh, doi mot truc. Cach nay tra loi dung cau hoi cua phan bien ("cai nay co
quan trong khong") voi 1/4 chi phi, va phai NOI RO trong bai la khong quet day du
chu khong de nguoi doc tuong da quet.

DIEU KIEN LANH MANH quan trong nhat: in kem SO THAM SO cho tung cau hinh. Tang G
tu 5 len 10 la tang tham so 1,5 lan; neu do chinh xac tang theo thi do co the chi
la hieu ung DUNG LUONG chu khong phai hieu ung cua luoi spline. Khong in cot tham
so thi hai cach giai thich do khong phan biet duoc, va bai se chon cach co loi
cho minh.
"""
from __future__ import annotations
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (nhan, model, hidden, grid, order) — mac dinh cua bai la kan/8/G5/k3
AXES = {
    "grid":  [("G3",  "kan", "8",  3, 3), ("G5",  "kan", "8",  5, 3), ("G10", "kan", "8", 10, 3)],
    "order": [("k3",  "kan", "8",  5, 3), ("k4",  "kan", "8",  5, 4), ("k5",  "kan", "8",  5, 5)],
    "width": [("h4",  "kan", "4",  5, 3), ("h8",  "kan", "8",  5, 3),
              ("h16", "kan", "16", 5, 3), ("h32", "kan", "32", 5, 3)],
}


def one(job, args):
    (tag, mname, hidden, grid, order), seed = job
    exp = f"sens_{tag}"
    out_root = ROOT / args.output_root
    run_dir = out_root / f"{exp}__dir{args.alpha}__seed{seed}"
    if args.skip_existing and (run_dir / "metrics.json").exists():
        return (exp, seed, "SKIP", 0.0)
    cmd = [args.python or sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(ROOT / "configs" / "experiments" / f"{args.config}.yaml"),
           "--seed", str(seed), "--exp-id", exp,
           "--model-name", mname, "--hidden", hidden,
           "--grid-size", str(grid), "--spline-order", str(order),
           "--mode", args.mode, "--partition", "dirichlet", "--alpha", str(args.alpha),
           "--lr", str(args.lr), "--output-root", str(out_root)]
    if args.downsample is not None:
        cmd += ["--downsample", str(args.downsample)]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.log", "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return (exp, seed, "OK" if p.returncode == 0 else f"LOI({p.returncode})", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--config", default="e1_botiot")
    ap.add_argument("--mode", default="binary")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--downsample", type=int, default=None)
    ap.add_argument("--seeds", default="11,17,23,29,31,37,42,43,2024,2026")
    ap.add_argument("--output-root", default="results/sensitivity")
    ap.add_argument("--python", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    # bo trung lap: G5/k3/h8 xuat hien o ca ba truc, chi chay mot lan
    seen, cfgs = set(), []
    for axis in ("grid", "order", "width"):
        for c in AXES[axis]:
            key = c[1:]
            if key in seen:
                continue
            seen.add(key)
            cfgs.append(c)
    jobs = list(itertools.product(cfgs, seeds))
    print("[sens] %d cau hinh x %d seed = %d run (da bo %d trung lap giua cac truc)"
          % (len(cfgs), len(seeds), len(jobs),
             sum(len(AXES[a]) for a in AXES) - len(cfgs)), flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for exp, seed, status, dt in ex.map(lambda j: one(j, args), jobs):
            done += 1
            print("[sens] %3d/%d  %-14s seed%-5d %-8s %5.1f phut"
                  % (done, len(jobs), exp, seed, status, dt / 60), flush=True)
    print("[sens] XONG sau %.1f phut" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
