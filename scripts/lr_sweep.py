"""R2#7: doc lap hoa learning rate khoi ket luan kien truc.

BAI NOP dung lr=0.01 cho MOI kien truc va khong dò riêng bao gio. Vay tuyen bo
"parameter-matched" moi chi khop THAM SO, chua khop SIEU THAM SO. Neu mot MLP
duoc dò lr rieng ma khep duoc khoang +6 pp thi ket luan kien truc cua bai chet.

Cach chay: moi kien truc duoc dò tren CUNG luoi lr, CUNG seed, CUNG o thi nghiem
(BoT-IoT, binary, Dir 0.1 - chinh la o sinh ra so headline). Sau do so kien truc
o lr TOT NHAT cua tung ben, khong phai o lr tot nhat cua mot ben.

THUOC DO: trung binh 5 vong cuoi, khong phai vong cuoi. Do tren cac run da co,
bien do dao dong 10 vong cuoi la 2,1-2,7 pp o ca bon kien truc, tuc mot vong don
la uoc luong nhieu. Vong cuoi van duoc in ra de doi chieu voi bai.

Log tho cua tung run duoc giu nguyen o results/lrsweep/<id>/stdout.log.
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

ARCHS = [
    # (nhan, model, hidden, grid_size)
    ("kan8",  "kan", "8",  5),
    ("kan16", "kan", "16", 5),
    ("mlp32", "mlp", "32", None),
    ("mlp80", "mlp", "80", None),
]
LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# 19/08: BAN DAU toi chi lay [42, 11, 23] va no cho ket luan SAI. Ba seed do khong
# co seed tham hoa nao, trong khi TOAN BO loi the cua KAN nam o duoi: MLP-PM-80 sup
# xuong 67,9 / 59,7 / 84,0 o seed 29 / 2026 / 2024, con KAN-8 te nhat la 81,1.
# Ba seed de cho khoang cach +0,72 pp; du 10 seed cho +5,63 pp. Tap kiem thien lech
# van cho mau xanh, nen sweep PHAI chay dung bo seed ma bai da bao cao.
SEEDS = [11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026]


def tag_lr(lr: float) -> str:
    return ("%g" % lr).replace(".", "p").replace("-", "m")


def one(job, args):
    (aname, mname, hidden, grid), lr, seed = job
    exp_id = f"lrsweep_{aname}_lr{tag_lr(lr)}"
    out_root = ROOT / args.output_root
    run_dir = out_root / f"{exp_id}__dir{args.alpha}__seed{seed}"
    if args.skip_existing and (run_dir / "metrics.json").exists():
        return (exp_id, seed, "SKIP", 0.0)

    cmd = [args.python or sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(ROOT / "configs" / "experiments" / f"{args.config}.yaml"),
           "--seed", str(seed), "--exp-id", exp_id,
           "--model-name", mname, "--hidden", hidden,
           "--mode", args.mode, "--partition", "dirichlet", "--alpha", str(args.alpha),
           "--lr", str(lr), "--output-root", str(out_root)]
    if args.downsample is not None:
        cmd += ["--downsample", str(args.downsample)]
    if grid is not None:
        cmd += ["--grid-size", str(grid)]

    env = dict(os.environ)
    # Moi tien trinh MOT luong BLAS: mac dinh numpy mo toi 80 luong tren may nay,
    # 6 tien trinh se thanh 480 luong tranh 80 nhan va cham hon chay tuan tu.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.log", "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    return (exp_id, seed, "OK" if p.returncode == 0 else f"LOI({p.returncode})", dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output-root", default="results/lrsweep")
    ap.add_argument("--config", default="e1_botiot")
    ap.add_argument("--python", default=None, help="python interpreter de goi run_experiment")
    ap.add_argument("--mode", default="binary")
    ap.add_argument("--alpha", type=float, default=0.1)
    # 19/08: o CSE-CIC MULTICLASS ban da nop dung 50000 chu khong phai 130000
    # cua cau hinh nen. Toi chay lan dau bang 130000 va do la mot THI NGHIEM
    # KHAC: no cho -1,24 pp trong khi bai bao +1,74 pp. Cau hinh phai khai ro
    # cho tung o, khong duoc thua ke im lang.
    ap.add_argument("--downsample", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    jobs = list(itertools.product(ARCHS, LRS, SEEDS))
    print(f"[sweep] {len(jobs)} run = {len(ARCHS)} kien truc x {len(LRS)} lr x {len(SEEDS)} seed",
          flush=True)
    print(f"[sweep] {args.workers} tien trinh song song, GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
          flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for exp_id, seed, status, dt in ex.map(lambda j: one(j, args), jobs):
            done += 1
            print(f"[sweep] {done:3d}/{len(jobs)}  {exp_id:28s} seed{seed:<5d} {status:8s} {dt/60:5.1f} phut",
                  flush=True)
    print(f"[sweep] XONG sau {(time.time()-t0)/60:.1f} phut", flush=True)


if __name__ == "__main__":
    main()
