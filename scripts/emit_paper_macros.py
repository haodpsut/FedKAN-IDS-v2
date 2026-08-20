"""MOT CHO O cho moi con so headline: sinh results/tables/macros.tex tu du lieu.

VI SAO. Do tren chinh bai nay ngay 20/08/2026: 216 tren 442 con so (49%) duoc GO
TAY vao van xuoi, va con so quan trong nhat bi chep nhieu nhat -- `+0.76` xuat hien
12 lan, `+6.23` 11 lan, `+5.49` 9 lan. Khi mot phat hien doi (sang 20/08: +0.76 hoa
ra sai co che, dung ra la +0.11 khi chi chinh MLP), sua cho dung nghia la sua 12 ban
sao roi rac bang tay. Khong ai giu noi 12 ban sao dong bo bang su chu y.

Day la loi KIEN TRUC chu khong phai loi can than, va cong khong chua duoc: cong chi
bat duoc phan no biet tim, nen ket qua la "sua nua voi" -- dung nghia den.

CACH DUNG. Bai viet \gapTunedBot chu khong viet $+0.76$. Doi giao thuc thi chay lai
script nay, moi cho tu dung. Lop loi "so cu con sot" BIEN MAT thay vi duoc bat.

PHEP THU NULL CUA CHINH LAN REFACTOR NAY. Thay so bang macro ma lam doi mot con so
dang in thi te hon la de nguyen. Quy trinh bat buoc: chup pdftotext TRUOC, thay,
dung lai, diff. Chenh lech phai la 0 dong.

Chay:  python3 scripts/emit_paper_macros.py
"""
from __future__ import annotations
import csv
import glob
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "macros.tex"
OLD = {11, 17, 23, 29, 31, 37, 42, 43, 2024, 2026}
NEW = set(range(101, 121))
GRID = ["0p0003", "0p001", "0p003", "0p01", "0p03", "0p1"]
NARROW = ["0p003", "0p01", "0p03"]
CELLS = [("lrsweep_botiot", "Bot"), ("lrsweep_toniot", "Ton"),
         ("lrsweep_cseciic", "Cse"), ("lrsweep_cseciic_mc50k", "CseMc")]


def mean5(d):
    a = [float(r["accuracy"]) for r in csv.DictReader(open(d + "/per_round.csv"))]
    return sum(a[-5:]) / 5


def cell(pre, arch, lr, seeds=OLD):
    o = {}
    for d in glob.glob(str(ROOT / ("results/%s/lrsweep_%s_lr%s__dir0.1__seed*" % (pre, arch, lr)))):
        m = re.search(r"seed(\d+)$", d)
        if not m or int(m.group(1)) not in seeds:
            continue
        try:
            o[int(m.group(1))] = mean5(d)
        except Exception:
            pass
    return o


def best_lr(pre, arch, seeds=OLD):
    c = [(st.mean(cell(pre, arch, lr, seeds).values()), lr)
         for lr in GRID if cell(pre, arch, lr, seeds)]
    return max(c)[1] if c else None


def paired_p(d):
    n = len(d)
    if n < 2:
        return float("nan")
    s = st.stdev(d)
    if s == 0:
        return 0.0
    t = abs(st.mean(d)) / (s / n ** 0.5)
    df, x = n - 1, (n - 1) / ((n - 1) + t * t)

    def betacf(a, b, x):
        c, d_ = 1.0, 1.0 - (a + b) * x / (a + 1.0)
        d_ = 1.0 / (d_ if abs(d_) > 1e-300 else 1e-300)
        h = d_
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
            d_ = 1.0 / max(abs(1.0 + aa * d_), 1e-300) * (1 if 1.0 + aa * d_ > 0 else -1)
            c = 1.0 + aa / c
            h *= d_ * c
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
            d_ = 1.0 / max(abs(1.0 + aa * d_), 1e-300) * (1 if 1.0 + aa * d_ > 0 else -1)
            c = 1.0 + aa / c
            de = d_ * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return math.exp(lb) * betacf(a, b, x) / a
        return 1.0 - math.exp(lb) * betacf(b, a, 1 - x) / b

    return betai(df / 2.0, 0.5, x)


def main():
    M = {}          # ten macro -> chuoi in ra
    pooled = []

    for pre, tag in CELLS:
        k0, m0 = cell(pre, "kan8", "0p01"), cell(pre, "mlp80", "0p01")
        lk, lm = best_lr(pre, "kan8"), best_lr(pre, "mlp80")
        kb, mb = cell(pre, "kan8", lk), cell(pre, "mlp80", lm)
        sp = [s for s in k0 if s in m0]
        M["gapShared" + tag] = "%+.2f" % (100 * st.mean([k0[s] - m0[s] for s in sp]))
        M["gapMlpOnly" + tag] = "%+.2f" % (100 * st.mean([k0[s] - mb[s] for s in k0 if s in mb]))
        tu = [kb[s] - mb[s] for s in kb if s in mb]
        M["gapTuned" + tag] = "%+.2f" % (100 * st.mean(tu))
        pooled += [100 * x for x in tu]
        M["sdRatioShared" + tag] = "%.2f" % (st.stdev(m0.values()) / st.stdev(k0.values()))
        M["sdRatioTuned" + tag] = "%.2f" % (st.stdev(mb.values()) / st.stdev(kb.values()))
        # do trai tren dai lr
        for band, bn in ((NARROW, "Narrow"), (GRID, "Full")):
            for arch, an in (("kan8", "Kan"), ("mlp80", "Mlp")):
                ms = [st.mean(cell(pre, arch, lr).values()) for lr in band if cell(pre, arch, lr)]
                M["spread%s%s%s" % (bn, an, tag)] = "%.2f" % (100 * (max(ms) - min(ms)))
        # thuoc do vong cuoi, de doi chieu
        def last(d_):
            a = [float(r["accuracy"]) for r in csv.DictReader(open(d_ + "/per_round.csv"))]
            return a[-1]
        try:
            kl = {s: last(g) for lr in ["0p01"] for g in
                  glob.glob(str(ROOT / ("results/%s/lrsweep_kan8_lr%s__dir0.1__seed*" % (pre, lr))))
                  for s in [int(re.search(r"seed(\d+)$", g).group(1))] if s in OLD}
            ml = {s: last(g) for lr in ["0p01"] for g in
                  glob.glob(str(ROOT / ("results/%s/lrsweep_mlp80_lr%s__dir0.1__seed*" % (pre, lr))))
                  for s in [int(re.search(r"seed(\d+)$", g).group(1))] if s in OLD}
            M["gapSharedLast" + tag] = "%+.2f" % (100 * st.mean([kl[s] - ml[s] for s in kl if s in ml]))
        except Exception:
            pass

    # BoT-IoT: moi ben duoc gi khi doi sang lr rieng
    k0 = cell("lrsweep_botiot", "kan8", "0p01")
    m0 = cell("lrsweep_botiot", "mlp80", "0p01")
    kb = cell("lrsweep_botiot", "kan8", best_lr("lrsweep_botiot", "kan8"))
    mb = cell("lrsweep_botiot", "mlp80", best_lr("lrsweep_botiot", "mlp80"))
    M["gainMlpBot"] = "%.2f" % (100 * (st.mean(mb.values()) - st.mean(m0.values())))
    M["gainKanBot"] = "%.2f" % (100 * (st.mean(kb.values()) - st.mean(k0.values())))
    M["bestLrKanBot"] = "3\\times10^{-3}"
    M["bestLrMlpBot"] = "10^{-1}"

    # giao thuc long nhau tren 20 seed chua nhin
    kn = cell("lrsweep_botiot", "kan8", best_lr("lrsweep_botiot", "kan8"), NEW)
    mn = cell("lrsweep_botiot", "mlp80", best_lr("lrsweep_botiot", "mlp80"), NEW)
    nd = [100 * (kn[s] - mn[s]) for s in kn if s in mn]
    if nd:
        M["gapNested"] = "%+.2f" % st.mean(nd)
        M["nNested"] = str(len(nd))
        M["pNested"] = "%.2f" % paired_p(nd)
        M["winNested"] = str(sum(1 for x in nd if x > 0))

    # gop
    M["gapPooled"] = "%+.2f" % st.mean(pooled)
    M["nPooled"] = str(len(pooled))
    M["pPooledPair"] = "%.3f" % paired_p(pooled)
    cm, i = [], 0
    for pre, tag in CELLS:
        n = len([s for s in cell(pre, "kan8", best_lr(pre, "kan8"))
                 if s in cell(pre, "mlp80", best_lr(pre, "mlp80"))])
        cm.append(st.mean(pooled[i:i + n]))
        i += n
    M["pPooledCell"] = "%.3f" % paired_p(cm)
    M["nCells"] = str(len(cm))

    # dai qua bon o, TUNG GIAO THUC RIENG (lop loi 11b: khong duoc ghep hai giao thuc)
    sh = [float(M["gapShared" + t]) for _, t in CELLS]
    tn = [float(M["gapTuned" + t]) for _, t in CELLS]
    M["rangeSharedLo"], M["rangeSharedHi"] = "%+.2f" % min(sh), "%+.2f" % max(sh)
    M["rangeTunedLo"], M["rangeTunedHi"] = "%+.2f" % min(tn), "%+.2f" % max(tn)

    # DEM RUN. Phai dung DUNG dinh nghia cua tab:runaccount, khong duoc glob bua.
    # Lan dau viet script nay toi glob "lrsweep_*" va ra 1.440 thay vi 1.200, vi no
    # om ca thu muc `lrsweep_cseciic_mc` (240 run cau hinh SAI, da cach ly bang
    # DUNG-DOC-THU-MUC-NAY.md). Tu dong hoa ma khong khop dinh nghia cua bai thi de
    # ra dung lop loi vua di sua. Danh sach duoi day la danh sach TRANG, co y.
    SWEEP_DIRS = ["lrsweep_botiot", "lrsweep_toniot", "lrsweep_cseciic", "lrsweep_cseciic_mc50k"]
    n_sweep = sum(len(list((ROOT / "results" / d).glob("*/metrics.json"))) for d in SWEEP_DIRS)
    # "{,}" chu khong phai ",": trong che do toan cua LaTeX, dau phay tran la dau
    # ngan danh sach nen "1,200" in ra thanh "1, 200". Phep thu null cua lan refactor
    # nay bat duoc dung loi do va khong bat duoc gi khac.
    M["nRunsSweep"] = "{:,}".format(n_sweep).replace(",", "{,}")

    # chi phi phan cung: neo quy doi, de nguoi doc chuyen sang may cua ho
    try:
        hw = json.load(open(ROOT / "results/hw_profile.json"))
        M["anchorMatmul"] = "%.3f" % hw["anchor_matmul_256_ms"]
    except Exception:
        pass

    # HE SO BEN lr: SO DAN XUAT, bat buoc phai tinh chu khong duoc go.
    # Bai viet "a factor of 2.7 to 5.4"; hai con so do la ti so cua chinh cac do
    # trai o tren. Go tay chung nghia la khi do trai doi, he so KHONG doi theo, va
    # cau van van doc troi chay -- dung loai loi khong ai phat hien duoc.
    fac = [float(M["spreadNarrowMlp" + tg]) / float(M["spreadNarrowKan" + tg])
           for tg in ("Bot", "Ton", "Cse")]
    M["robustFactorLo"], M["robustFactorHi"] = "%.1f" % min(fac), "%.1f" % max(fac)

    # do lech chuan tho cua o BoT-IoT (bai in ca hai ben)
    M["sdKanBot"] = "%.2f" % (100 * st.stdev(cell("lrsweep_botiot", "kan8", "0p01").values()))
    M["sdMlpBot"] = "%.2f" % (100 * st.stdev(cell("lrsweep_botiot", "mlp80", "0p01").values()))
    M["accKanBotShared"] = "%.2f" % (100 * st.mean(cell("lrsweep_botiot", "kan8", "0p01").values()))
    M["accMlpBotShared"] = "%.2f" % (100 * st.mean(cell("lrsweep_botiot", "mlp80", "0p01").values()))

    # Bien the KHONG DAU. Van xuoi co cho viet "the $5.49$~pp gap" (khong dau) va cho
    # viet "from $+5.49$" (co dau). Neu chi co mot bien the thi lan thay macro se doi
    # chu IN RA, ma phep thu null cua lan refactor nay doi hoi 0 dong chenh lech.
    for k in [k for k in list(M) if k.startswith(("gap", "range"))]:
        M[k + "U"] = M[k].lstrip("+-")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/emit_paper_macros.py -- KHONG SUA TAY.\n")
        f.write("% Moi con so headline chi co MOT cho o. Xem dau file script de biet vi sao.\n")
        for k in sorted(M):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
    print("  da ghi %s  (%d macro)" % (OUT.relative_to(ROOT), len(M)))
    for k in sorted(M):
        print("    \\%-22s %s" % (k, M[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
