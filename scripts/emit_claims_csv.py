"""Sinh CSV GON tu 8087 tep ket qua, de pipeline truy nguoc tung so trong bai.

Pipeline doi mot bang phang: moi dong la mot (o thi nghiem, kien truc, lr, seed)
kem cac thuoc do. Khong co bang nay thi verify_numbers khong chay duoc, va moi so
trong bai la so GO TAY khong ai kiem.

Ghi CA HAI thuoc do (vong cuoi VA trung binh 5 vong cuoi) vi bai bao cao ca hai va
so sanh chung; chi ghi mot cai thi cong khong kiem duoc cau "doi thuoc do thi so doi
bao nhieu".
"""
import csv, glob, json, math, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "claims_source.csv"

CELLS = {
    "lrsweep_botiot":      ("nf_botiot_v2",  "binary",     130000),
    "lrsweep_toniot":      ("nf_toniot_v2",  "binary",     130000),
    "lrsweep_cseciic":     ("nf_cseciic_v2", "binary",     130000),
    "lrsweep_cseciic_mc50k":("nf_cseciic_v2","multiclass",  50000),
}

def fpr_mcc(fm):
    rec, acc = fm.get("per_class_recall"), fm.get("accuracy")
    if not rec or len(rec) != 2 or abs(rec[0]-rec[1]) < 1e-9: return None, None
    r0, r1 = rec; f0 = (acc-r1)/(r0-r1)
    if not 0 <= f0 <= 1: return None, None
    f1_ = 1-f0
    TN, FP, TP, FN = f0*r0, f0*(1-r0), f1_*r1, f1_*(1-r1)
    den = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (FP/(FP+TN) if FP+TN > 0 else None,
            (TP*TN-FP*FN)/den if den > 0 else None)

rows = []
def add(cell, dataset, mode, ds, arch, lr, seed, algo, grid, d):
    p = Path(d)
    try:
        pr = list(csv.DictReader(open(p/"per_round.csv")))
        a = [float(r["accuracy"]) for r in pr]
        m = json.load(open(p/"metrics.json"))
    except Exception:
        return
    fm = m["final_metrics"]
    fpr, mcc = fpr_mcc(fm)
    rows.append(dict(
        cell=cell, dataset=dataset, mode=mode, downsample=ds, arch=arch,
        lr=lr, seed=seed, algo=algo, grid_size=grid, n_params=m.get("n_params"),
        acc_last=a[-1], acc_mean5=sum(a[-5:])/5, acc_mean10=sum(a[-10:])/10,
        acc_best=max(a), f1_macro=fm.get("f1_macro"),
        fpr=fpr, mcc=mcc,
        uplink_total_MB=sum(float(r["comm_uplink_bytes"]) for r in pr)/1e6,
        rounds=len(a),
        osc10=max(a[-10:])-min(a[-10:]),
    ))

for cell,(dsname,mode,down) in CELLS.items():
    for d in glob.glob(str(ROOT/"results"/cell/"*")):
        mm = re.search(r"lrsweep_(\w+?)_lr([\dpm]+)__dir[\d.]+__seed(\d+)", d)
        if not mm: continue
        add(cell, dsname, mode, down, mm.group(1),
            float(mm.group(2).replace("p",".").replace("m","-")),
            int(mm.group(3)), "fedavg", 5, d)

for d in glob.glob(str(ROOT/"results"/"algos"/"*")):
    mm = re.search(r"algo_(\w+?)_(kan8|mlp80)__dir[\d.]+__seed(\d+)", d)
    if not mm: continue
    add("algos", "nf_botiot_v2", "binary", 130000, mm.group(2), 0.01,
        int(mm.group(3)), mm.group(1), 5, d)

SENS = {"G3":(8,3),"G5":(8,5),"G10":(8,10),"k4":(8,5),"k5":(8,5),
        "h4":(4,5),"h16":(16,5),"h32":(32,5)}
for d in glob.glob(str(ROOT/"results"/"sensitivity_botiot"/"*")):
    mm = re.search(r"sens_(\w+?)__dir[\d.]+__seed(\d+)", d)
    if not mm: continue
    tag = mm.group(1)
    add("sensitivity", "nf_botiot_v2", "binary", 130000, "kan_"+tag, 0.01,
        int(mm.group(2)), "fedavg", SENS.get(tag,(8,5))[1], d)

for d in glob.glob(str(ROOT/"results"/"g10"/"*")):
    mm = re.search(r"g10_(\w+?)__dir[\d.]+__seed(\d+)", d)
    if not mm: continue
    add("g10", "nf_"+mm.group(1)+"_v2", "binary", 130000, "kan8_G10", 0.01,
        int(mm.group(2)), "fedavg", 10, d)

OUT.parent.mkdir(parents=True, exist_ok=True)
cols = list(rows[0].keys())
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("  da ghi %s: %d dong x %d cot" % (OUT.relative_to(ROOT), len(rows), len(cols)))
from collections import Counter
for k in ("cell","algo"):
    print("   ", k, dict(Counter(r[k] for r in rows)))
