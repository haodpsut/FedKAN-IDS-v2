"""R1#4: chi phi truyen that, khong chi dem tham so.

Phan bien 1: "Beyond parameter transmission, add transmitted bytes, communication
rounds, convergence efficiency, or training time". Ca bon deu da nam san trong
per_round.csv (cot comm_uplink_bytes, comm_downlink_bytes, wallclock_s), khong
can chay lai gi.

HAI COT DANG CHU Y, va chung do hai thu khac nhau:

  byte_den_dich   = tong byte uplink cho toi khi dat nguong do chinh xac
  vong_den_dich   = so vong cho toi khi dat nguong do

"Re hon" theo byte moi vong khong co nghia la re hon de DAT MOT MUC. Bai hien
chi bao cao tong byte sau 50 vong, tuc gia dinh ca hai ben deu phai chay du 50
vong. Neu mot kien truc dat nguong o vong 20 thi so do sai lech han.

NGUONG lay theo phan tram cua muc tot nhat ma BAT KY kien truc nao dat duoc
trong nhom, de khong thien vi ben nao. Ô nao khong bao gio dat nguong thi in
"khong dat" chu khong im lang bo qua: bo qua se lam trung binh dep len.
"""
from __future__ import annotations
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(d: Path):
    rows = list(csv.DictReader(open(d / "per_round.csv")))
    return [{"r": int(x["round"]), "acc": float(x["accuracy"]),
             "up": float(x["comm_uplink_bytes"]), "dn": float(x["comm_downlink_bytes"]),
             "t": float(x["wallclock_s"])} for x in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--filter", default="e1_botiot_binary")
    ap.add_argument("--target-frac", type=float, default=0.98,
                    help="nguong = target_frac * do chinh xac tot nhat trong nhom")
    args = ap.parse_args()

    groups = defaultdict(list)
    for d in sorted(Path(args.runs).glob("*")):
        if not (d / "per_round.csv").exists() or args.filter not in d.name:
            continue
        if "dir0.1" not in d.name:
            continue
        groups[d.name.split("__")[0]].append(load(d))

    if not groups:
        print("  khong co run nao khop; khong bao cao gi.")
        return

    best = max(max(x["acc"] for x in run) for runs in groups.values() for run in runs)
    thr = args.target_frac * best
    print("=" * 92)
    print("R1#4: CHI PHI TRUYEN VA HIEU QUA HOI TU  (BoT-IoT binary, Dir 0.1)")
    print("=" * 92)
    print("  do chinh xac tot nhat bat ky kien truc nao dat: %.4f%%" % (100 * best))
    print("  nguong dung chung (%.0f%% cua muc do)          : %.4f%%" % (100 * args.target_frac, 100 * thr))
    print()
    print("  %-30s %11s %11s %10s %11s %8s" % (
        "kien truc", "tong up(MB)", "vong->nguong", "up->nguong", "giay/vong", "n dat"))
    print("  " + "-" * 88)
    for k in sorted(groups):
        runs = groups[k]
        tot = [r[-1]["up"] * 0 + sum(x["up"] for x in r) for r in runs]
        spr = [st.mean(x["t"] for x in r) for r in runs]
        hit_r, hit_b, n_hit = [], [], 0
        for r in runs:
            for x in r:
                if x["acc"] >= thr:
                    hit_r.append(x["r"])
                    hit_b.append(sum(y["up"] for y in r if y["r"] <= x["r"]))
                    n_hit += 1
                    break
        print("  %-30s %10.2f %11s %10s %10.2f %5d/%d" % (
            k, st.mean(tot) / 1e6,
            "%.1f" % st.mean(hit_r) if hit_r else "khong dat",
            "%.2f MB" % (st.mean(hit_b) / 1e6) if hit_b else "-",
            st.mean(spr), n_hit, len(runs)))
    print()
    print("  Cot 'n dat' la dieu kien lanh manh: neu mot kien truc chi dat nguong o vai seed")
    print("  thi trung binh cua no chi tinh tren nhung seed do, va con so khong so sanh duoc.")


if __name__ == "__main__":
    main()
