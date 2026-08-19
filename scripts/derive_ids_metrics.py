"""R1#3: bo sung MCC, FPR, Precision, Recall ma KHONG phai chay lai thi nghiem.

Phan bien 1 doi "Precision, Recall, F1-score, MCC, ROC-AUC, or False Positive Rate".
Ba cai dau da luu san trong metrics.json (per_class_precision/recall/f1). Hai cai
tiep theo suy duoc chinh xac cho bai toan NHI PHAN tu nhung gi da luu:

    recall_0 = TN/(TN+FP)  =>  FPR = 1 - recall_0     (truc tiep)

    acc = (n0*recall_0 + n1*recall_1)/N,  n0+n1=N
    => n0/N = (acc - recall_1)/(recall_0 - recall_1)   (khi recall_0 != recall_1)
    => TN = n0*recall_0, FP = n0*(1-recall_0), TP = n1*recall_1, FN = n1*(1-recall_1)
    => MCC tu ma tran nham lan day du

ROC-AUC thi KHONG suy duoc: no can diem so lien tuc, ma cac run chi luu nhan du
doan. Muon co phai chay lai va luu diem so. Noi thang chuyen do trong bai chu
dung im lang bo qua mot yeu cau cua phan bien.

KIEM CHUNG NOI TAI: ma tran tai lap phai sinh ra dung `per_class_precision` da
luu. Neu lech qua dung sai thi tai lap SAI va con so bi loai, khong duoc bao cao.
Cot "so o da kiem" duoc in ra vi mot cong khong in so o da kiem la mot cong khong
chung minh duoc gi.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

TOL = 5e-3   # dung sai cho kiem chung precision tai lap


def confusion_from_metrics(m: dict):
    """Tra ve (TN, FP, FN, TP, ok) cho run nhi phan; ok=False neu khong tai lap duoc."""
    fm = m["final_metrics"]
    rec = fm.get("per_class_recall")
    pre = fm.get("per_class_precision")
    acc = fm.get("accuracy")
    if not rec or not pre or len(rec) != 2:
        return None
    r0, r1 = rec
    if abs(r0 - r1) < 1e-9:
        return None                       # he suy bien, khong tach duoc n0/N
    f0 = (acc - r1) / (r0 - r1)           # ty le lop 0 trong tap kiem
    if not (0.0 <= f0 <= 1.0):
        return None
    f1_ = 1.0 - f0
    TN, FP = f0 * r0, f0 * (1 - r0)
    TP, FN = f1_ * r1, f1_ * (1 - r1)
    # kiem chung: precision tai lap co khop precision da luu khong
    p1 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    p0 = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    ok = abs(p1 - pre[1]) < TOL and abs(p0 - pre[0]) < TOL
    return TN, FP, FN, TP, ok


def mcc(TN, FP, FN, TP):
    num = TP * TN - FP * FN
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--pattern", default="binary")
    args = ap.parse_args()

    groups = defaultdict(list)
    n_seen = n_ok = n_bad = n_skip = 0

    for d in sorted(Path(args.runs).glob("*")):
        f = d / "metrics.json"
        if not f.exists() or args.pattern not in d.name:
            continue
        m = json.load(open(f))
        n_seen += 1
        c = confusion_from_metrics(m)
        if c is None:
            n_skip += 1
            continue
        TN, FP, FN, TP, ok = c
        if not ok:
            n_bad += 1
            continue
        n_ok += 1
        key = d.name.split("__")[0]
        groups[key].append({
            "fpr": FP / (FP + TN) if (FP + TN) > 0 else 0.0,
            "mcc": mcc(TN, FP, FN, TP),
            "precision": TP / (TP + FP) if (TP + FP) > 0 else 0.0,
            "recall": TP / (TP + FN) if (TP + FN) > 0 else 0.0,
            "acc": m["final_metrics"]["accuracy"],
        })

    print("=" * 84)
    print("R1#3: METRIC IDS SUY TU RUN DA CO  (khong chay lai thi nghiem)")
    print("=" * 84)
    print("  o quet          : %d run khop mau '%s'" % (n_seen, args.pattern))
    print("  o TAI LAP DUOC  : %d" % n_ok)
    print("  o loai vi lech  : %d  (precision tai lap khac ban luu > %.0e)" % (n_bad, TOL))
    print("  o loai vi suy bien: %d  (recall hai lop bang nhau, khong tach duoc n0/N)" % n_skip)
    print()
    if not n_ok:
        print("  KHONG co o nao tai lap duoc: khong bao cao con so nao.")
        return
    print("  %-34s %9s %9s %10s %9s %4s" % ("nhom run", "acc", "FPR", "MCC", "recall", "n"))
    print("  " + "-" * 80)
    for k in sorted(groups):
        v = groups[k]
        print("  %-34s %8.4f%% %8.4f%% %9.4f %8.4f%% %4d" % (
            k, 100 * st.mean(x["acc"] for x in v), 100 * st.mean(x["fpr"] for x in v),
            st.mean(x["mcc"] for x in v), 100 * st.mean(x["recall"] for x in v), len(v)))
    print()
    print("  ROC-AUC: KHONG suy duoc tu du lieu da luu (can diem so lien tuc, run chi luu nhan).")
    print("  Phai chay lai co luu diem so, hoac khai ro trong thu tra loi la khong cung cap.")


if __name__ == "__main__":
    main()
