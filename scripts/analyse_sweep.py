"""Doc mot sweep lr va tra loi R2#7: khoang cach con lai bao nhieu khi MOI kien
truc duoc dò lr rieng.

Thuoc do la TRUNG BINH 5 VONG CUOI. Ly do: bien do dao dong 10 vong cuoi do duoc
la 2,1-2,7 pp o ca bon kien truc, nen mot vong don la uoc luong nhieu. Cot vong
cuoi van in ra de doi chieu voi ban da nop.
"""
import csv, glob, re, sys, statistics as st
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/lrsweep_botiot"
ARCH = ["kan8", "kan16", "mlp32", "mlp80"]

def m5(d):
    a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
    return sum(a[-5:]) / 5, a[-1]

acc, fin = defaultdict(dict), defaultdict(dict)
for d in glob.glob(ROOT + "/*"):
    m = re.search(r"lrsweep_(\w+?)_lr([\dpm]+)__dir[\d.]+__seed(\d+)", d)
    if not m: continue
    lr = float(m.group(2).replace("p", ".").replace("m", "-"))
    try: v, f = m5(d)
    except Exception: continue
    acc[(m.group(1), lr)][int(m.group(3))] = v
    fin[(m.group(1), lr)][int(m.group(3))] = f

LRS = sorted({k[1] for k in acc})
if not LRS:
    print("  khong doc duoc run nao o", ROOT); raise SystemExit

print("=" * 74)
print("  %s   (trung binh 5 vong cuoi)" % ROOT)
print("=" * 74)
print("  %-9s" % "lr", end="")
for a in ARCH: print("%13s" % a, end="")
print("   n")
print("  " + "-" * 68)
for lr in LRS:
    print("  %-9g" % lr, end="")
    n = 0
    for a in ARCH:
        v = acc.get((a, lr), {}); n = max(n, len(v))
        print("%12s " % ("%.3f%%" % (100*st.mean(v.values())) if v else "-"), end="")
    print("   %d" % n)

best = {}
print()
for a in ARCH:
    c = [(st.mean(acc[(a, lr)].values()), lr) for lr in LRS if acc.get((a, lr))]
    if not c: continue
    best[a] = max(c)
    at01 = st.mean(acc[(a, 0.01)].values()) if acc.get((a, 0.01)) else float("nan")
    print("  %-7s lr tot nhat %-7g -> %.3f%%  | tai lr=0.01 -> %.3f%%  | tuning duoc %+.2f pp"
          % (a, best[a][1], 100*best[a][0], 100*at01, 100*(best[a][0]-at01)))

def boot(d, n=20000, seed=0):
    r = np.random.default_rng(seed); d = np.array(d)
    return np.percentile([r.choice(d, len(d), replace=True).mean() for _ in range(n)], [2.5, 97.5])

print()
print("  %-42s %9s %8s %20s" % ("so sanh KAN-8 voi MLP-PM-80", "hieu TB", "p ghep", "KTC bootstrap 95%"))
print("  " + "-" * 82)
rows = [("ca hai o lr=0.01 (nhu ban da nop)", ("kan8", 0.01), ("mlp80", 0.01))]
if "kan8" in best and "mlp80" in best:
    rows.append(("moi ben o lr tot nhat cua minh", ("kan8", best["kan8"][1]), ("mlp80", best["mlp80"][1])))
for lab, k, m in rows:
    s = sorted(set(acc[k]) & set(acc[m]))
    if len(s) < 3: continue
    d = [100*(acc[k][x] - acc[m][x]) for x in s]
    _, p = stats.ttest_rel([acc[k][x] for x in s], [acc[m][x] for x in s])
    lo, hi = boot(d)
    print("  %-42s %+8.3f %8.4f   [%+7.3f, %+7.3f]  n=%d" % (lab, st.mean(d), p, lo, hi, len(s)))

print()
print("  do lech chuan giua seed (tuyen bo cu: KAN giam phuong sai 2,6 lan)")
for lab, key in [("KAN-8 @0.01", ("kan8", 0.01)), ("MLP-80 @0.01", ("mlp80", 0.01))] + \
                ([("KAN-8 @tot nhat", ("kan8", best["kan8"][1])),
                  ("MLP-80 @tot nhat", ("mlp80", best["mlp80"][1]))] if "mlp80" in best else []):
    if not acc.get(key): continue
    v = [100*x for x in acc[key].values()]
    print("    %-20s sigma %6.3f pp | te nhat %.3f%%" % (lab, st.stdev(v), min(v)))
if acc.get(("kan8",0.01)) and acc.get(("mlp80",0.01)) and "mlp80" in best:
    r0 = st.stdev([100*x for x in acc[("mlp80",0.01)].values()]) / st.stdev([100*x for x in acc[("kan8",0.01)].values()])
    r1 = st.stdev([100*x for x in acc[("mlp80",best["mlp80"][1])].values()]) / st.stdev([100*x for x in acc[("kan8",best["kan8"][1])].values()])
    print("    ty le MLP/KAN o lr=0.01: %.2fx  ->  khi moi ben dò lr: %.2fx" % (r0, r1))
