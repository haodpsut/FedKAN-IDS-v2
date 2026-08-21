"""CHANG 0 cua ke hoach sua IoT-J: 340 run bo sung. Chay TRUOC khi viet dong nao.

Luat Hao chot 20/08/2026: revise thi chay het thuc nghiem TRUOC, doi chieu du
response reviewer, roi moi sua tung section.

VI SAO CHANG NAY DUNG DAU. Ca `results/lrext/`: 40 run da nam tren dia tu lau,
chung cho thay KAN sup con 42,44% o eta=0,3 trong khi MLP giu 90,50%, tuc GIET
tuyen bo duong duy nhat con song cua bai. Bang doi soat DEM chung, ban thao KHONG
bao cao o dau, va muc Threats con viet nhu the du lieu do chua ton tai. Viet truoc
thi van ban dong khung lay ket luan va du lieu chay sau bi bo roi.

BON ME, va vi sao moi me la bat buoc:

  0.1 giao thuc long nhau tren BA O CON LAI (120 run)
      Ket luan headline -0,54 pp hien chi co o MOT o (BoT-IoT). Phan bien se goi
      do la ket qua mot o. Chon lr tren 10 seed cu (DA CO), bao cao tren seed
      101-120 chua tung nhin.

  0.2 DA LOP BoT-IoT + ToN-IoT tren CPU (80 run)
      Khong co no thi Bang IV/VI/VII khong sinh lai duoc tu CPU, va cau "moi thi
      nghiem chay tren MOT may" van SAI. Hien ba bang do sinh tu results/runs,
      310/310 run ghi device=cuda.

  0.3 LUOI MO RONG eta in {0,3; 1,0} cho ba o con lai (120 run)
      lrext moi co BoT-IoT. Tuyen bo ben-voi-lr trai tren bon o nen phai do ca bon.

  0.4 SCAFFOLD duoi SGD (20 run)
      Bai TU NEU gia thuyet "SCAFFOLD lech pha vi ta dung Adam" ma khong kiem.
      Kiem no la doi mot dong cau hinh.

KHAI TUONG MINH TUNG O, KHONG THUA KE. Ca 19/08: o CSE-CIC da lop cua ban da nop
dung downsample=50000 con cau hinh nen la 130000; chay nham cho -1,24 pp thay vi
+1,38 pp, tuc DOI DAU ket luan. Moi me duoi day khai downsample cua rieng no.

Chay:  .venv/bin/python scripts/run_phase0.py --workers 6
       .venv/bin/python scripts/run_phase0.py --only 0.1 --dry-run
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
OLD_SEEDS = [11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026]
NEW_SEEDS = list(range(101, 121))

# lr tot nhat tung ben, DO TU 10 seed cu, khoa lai truoc khi cham seed moi.
# Nguon: scripts/emit_paper_macros.py -> best_lr(); da in trong Bang IX cua bai.
BEST = {
    "toniot":       {"kan8": 1e-2,  "mlp80": 1e-2},
    "cseciic":      {"kan8": 3e-3,  "mlp80": 1e-2},
    "cseciic_mc50k": {"kan8": 1e-2, "mlp80": 1e-2},
}
CELL = {   # ten o -> (config, mode, downsample, output-root)
    "botiot":        ("e1_botiot",  "binary",     130000, "results/lrsweep_botiot"),
    "toniot":        ("e1_toniot",  "binary",     130000, "results/lrsweep_toniot"),
    "cseciic":       ("e1_cseciic", "binary",     130000, "results/lrsweep_cseciic"),
    "cseciic_mc50k": ("e1_cseciic", "multiclass",  50000, "results/lrsweep_cseciic_mc50k"),
}
ARCH = {"kan8": ("kan", "8", 5), "kan16": ("kan", "16", 5),
        "mlp32": ("mlp", "32", None), "mlp80": ("mlp", "80", None)}


def tag(lr):
    return ("%g" % lr).replace(".", "p").replace("-", "m")


def job(exp_id, out_root, cfg, mode, down, arch, lr, seed, algo="fedavg", opt=None):
    mname, hidden, grid = ARCH[arch]
    run_dir = ROOT / out_root / f"{exp_id}__dir0.1__seed{seed}"
    cmd = [PY, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(ROOT / "configs" / "experiments" / f"{cfg}.yaml"),
           "--seed", str(seed), "--exp-id", exp_id,
           "--model-name", mname, "--hidden", hidden,
           "--mode", mode, "--partition", "dirichlet", "--alpha", "0.1",
           "--lr", str(lr), "--downsample", str(down),
           "--algo", algo, "--output-root", str(ROOT / out_root)]
    if grid is not None:
        cmd += ["--grid-size", str(grid)]
    if opt is not None:
        cmd += ["--optimizer", opt]
    return (run_dir, cmd)


def build():
    """Tra ve dict: ten me -> danh sach (run_dir, cmd)."""
    B = {"0.1": [], "0.2": [], "0.3": [], "0.4": []}

    # --- 0.1 giao thuc long nhau, ba o con lai, seed chua tung nhin ---------
    for cell in ("toniot", "cseciic", "cseciic_mc50k"):
        cfg, mode, down, root = CELL[cell]
        for arch, lr in BEST[cell].items():
            for s in NEW_SEEDS:
                B["0.1"].append(job(f"lrsweep_{arch}_lr{tag(lr)}", root, cfg, mode, down,
                                    arch, lr, s))

    # --- 0.2 da lop BoT-IoT + ToN-IoT tren CPU, lr chung cua giao thuc goc ---
    for cell, cfg in (("botiot", "e1_botiot"), ("toniot", "e1_toniot")):
        root = f"results/mc_{cell}"
        for arch in ARCH:
            for s in OLD_SEEDS:
                B["0.2"].append(job(f"mc_{arch}_lr0p01", root, cfg, "multiclass", 50000,
                                    arch, 1e-2, s))

    # --- 0.3 luoi mo rong, ba o con lai -------------------------------------
    for cell in ("toniot", "cseciic", "cseciic_mc50k"):
        cfg, mode, down, _ = CELL[cell]
        root = f"results/lrext_{cell}"
        for arch in ("kan8", "mlp80"):
            for lr in (0.3, 1.0):
                for s in OLD_SEEDS:
                    B["0.3"].append(job(f"lrext_{arch}_lr{tag(lr)}", root, cfg, mode, down,
                                        arch, lr, s))

    # --- 0.4 SCAFFOLD duoi SGD, o headline ----------------------------------
    cfg, mode, down, _ = CELL["botiot"]
    for arch in ("kan8", "mlp80"):
        for s in OLD_SEEDS:
            B["0.4"].append(job(f"scaffold_sgd_{arch}", "results/scaffold_sgd", cfg, mode,
                                down, arch, 1e-2, s, algo="scaffold", opt="sgd"))
    return B


def run_one(item, skip_existing=True):
    run_dir, cmd = item
    if skip_existing and (run_dir / "metrics.json").exists():
        return "SKIP"
    env = dict(os.environ)
    # MOT luong BLAS moi tien trinh: numpy mac dinh mo toi so nhan cua may, sau do
    # 6 tien trinh se tranh nhau va cham hon chay tuan tu.
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.log", "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return "OK" if p.returncode == 0 else f"LOI({p.returncode})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="vd: 0.1 0.4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    a = ap.parse_args()

    B = build()
    names = a.only or sorted(B)
    total = sum(len(B[n]) for n in names)
    print("=" * 78)
    print("  CHANG 0 — %d run tren %d me: %s" % (total, len(names), ", ".join(names)))
    print("=" * 78)
    for n in names:
        done = sum(1 for d, _ in B[n] if (d / "metrics.json").exists())
        print("  me %s: %3d run  (da co %d, con %d)" % (n, len(B[n]), done, len(B[n]) - done))
    if a.dry_run:
        print("\n  --dry-run: in mot lenh mau cua tung me\n")
        for n in names:
            print("  [%s] %s" % (n, " ".join(B[n][0][1][1:])[:190]))
        return 0

    t0 = time.time()
    for n in names:
        items = B[n]
        print("\n  ── me %s: %d run ──" % (n, len(items)), flush=True)
        ok = skip = err = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(lambda it: run_one(it, not a.no_skip), items), 1):
                if r == "OK":
                    ok += 1
                elif r == "SKIP":
                    skip += 1
                else:
                    err += 1
                    print("     ⛔ %s -> %s" % (items[i - 1][0].name, r), flush=True)
                if i % 20 == 0 or i == len(items):
                    print("     %3d/%3d  ok=%d skip=%d loi=%d  (%.1f phut)"
                          % (i, len(items), ok, skip, err, (time.time() - t0) / 60), flush=True)
    print("\n  XONG chang 0 sau %.1f phut" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
