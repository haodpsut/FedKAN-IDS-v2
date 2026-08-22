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

    # ---- so cua Observation 1 va Observation 2 ------------------------------
    # Truoc 21/08 chung la LITERAL v1 go tay: bai in 97,53 / 99,00 trong khi Bang I
    # (sinh tu 10 seed CPU) in 97,67 / 98,51. Doc gia so hai cho thay lech ngay.
    def _low(arch, reg):
        v = [mean5(d) for d in glob.glob(str(ROOT / ("results/lowhet/lowhet_%s__%s__seed*" % (arch, reg))))]
        return v
    for arch, nm in (("kan8", "KanIid"), ("kan16", "KanSixteenIid"),
                     ("mlp32", "MlpThirtyTwoIid"), ("mlp80", "MlpIid")):
        v = _low(arch, "iid")
        if v:
            M["acc" + nm] = "%.2f" % (100 * st.mean(v))
    iid = {a: st.mean(_low(a, "iid")) for a in ("kan8", "kan16", "mlp32", "mlp80") if _low(a, "iid")}
    d10 = {a: st.mean(_low(a, "dir1.0")) for a in ("kan8", "kan16", "mlp32", "mlp80") if _low(a, "dir1.0")}
    if iid:
        M["spreadIid"] = "%.2f" % (100 * (max(iid.values()) - min(iid.values())))
        M["leadMlpIid"] = "%.2f" % (100 * (iid["mlp80"] - iid["kan8"]))
    if d10:
        M["spreadDirOne"] = "%.2f" % (100 * (max(d10.values()) - min(d10.values())))

    # worst-seed o headline, hai giao thuc canh nhau (Observation 2 / B2)
    def _worst(arch, lr):
        v = [mean5(d) for d in glob.glob(str(ROOT / ("results/lrsweep_botiot/lrsweep_%s_lr%s__dir0.1__seed*" % (arch, lr))))
             if int(re.search(r"seed(\d+)$", d).group(1)) in OLD]
        return min(v) if v else None
    wk, wm = _worst("kan8", "0p01"), _worst("mlp80", "0p01")
    tk = _worst("kan8", best_lr("lrsweep_botiot", "kan8"))
    tm = _worst("mlp80", best_lr("lrsweep_botiot", "mlp80"))
    if None not in (wk, wm, tk, tm):
        M["worstKanShared"] = "%.2f" % (100 * wk)
        M["worstMlpShared"] = "%.2f" % (100 * wm)
        M["worstKanTuned"] = "%.2f" % (100 * tk)
        M["worstMlpTuned"] = "%.2f" % (100 * tm)
        M["worstGapShared"] = "%+.2f" % (100 * (wk - wm))
        M["worstGapTuned"] = "%+.2f" % (100 * (tk - tm))

    # ---- CHANG 0 (20/08): so moi tu 340 run bo sung -------------------------
    # 0.1 GIAO THUC LONG NHAU TREN CA BON O. Truoc chang 0, ket luan headline
    # -0,54 pp chi do tren MOT o, va bai phat bieu no nhu khang dinh chung. Nay
    # do du bon o thi thay no CHI dung o BoT-IoT.
    for pre, tg in CELLS:
        lk, lm = best_lr(pre, "kan8"), best_lr(pre, "mlp80")
        kn, mn = cell(pre, "kan8", lk, NEW), cell(pre, "mlp80", lm, NEW)
        sp = sorted(set(kn) & set(mn))
        if not sp:
            continue
        d = [100 * (kn[s] - mn[s]) for s in sp]
        M["nested" + tg] = "%+.2f" % st.mean(d)
        M["nestedP" + tg] = "%.3f" % paired_p(d)
        M["nestedWin" + tg] = str(sum(1 for x in d if x > 0))
        M["nestedN" + tg] = str(len(d))

    # 0.3 LUOI DAY DU, ke ca eta=0,3 va 1,0. Ti le MLP/KAN > 1 nghia la KAN ben
    # hon. Tren dai HEP cua ban cu ti le la 5,39/4,04/2,66/1,03; tren luoi DAY DU
    # no thanh 0,33/0,74/0,86/0,88, tuc KAN kem ben hon o CA BON o. Con so
    # "2,7 den 5,4 lan" chi ton tai trong dai hep ma chinh ta chon.
    EXTDIR = {"Bot": "lrext", "Ton": "lrext_toniot",
              "Cse": "lrext_cseciic", "CseMc": "lrext_cseciic_mc50k"}
    for pre, tg in CELLS:
        for arch, an in (("kan8", "Kan"), ("mlp80", "Mlp")):
            vals = [st.mean(cell(pre, arch, lr).values()) for lr in GRID if cell(pre, arch, lr)]
            for lab in ("03", "0p3"):
                g = glob.glob(str(ROOT / ("results/%s/lrext_%s_lr%s__dir0.1__seed*" % (EXTDIR[tg], arch, lab))))
                if g:
                    vals.append(st.mean(mean5(d) for d in g))
                    break
            for lab in ("10", "1", "1p0"):
                g = glob.glob(str(ROOT / ("results/%s/lrext_%s_lr%s__dir0.1__seed*" % (EXTDIR[tg], arch, lab))))
                if g:
                    vals.append(st.mean(mean5(d) for d in g))
                    break
            M["spreadWide%s%s" % (an, tg)] = "%.2f" % (100 * (max(vals) - min(vals)))
        M["ratioWide" + tg] = "%.2f" % (float(M["spreadWideMlp" + tg]) / float(M["spreadWideKan" + tg]))

    # DO NHAY theo G, bac spline k, do rong h. 80 run da chay tu truoc, nhung chi
    # G=10 vao bai; thu tra loi R2.5 lai HUA co k va h kem so cu the. Dua het vao.
    for nm in ("G3", "G5", "G10", "k4", "k5", "h4", "h16", "h32"):
        g = glob.glob(str(ROOT / ("results/sensitivity_botiot/sens_%s__*" % nm)))
        if g:
            # TEN MACRO CHI DUOC CO CHU CAI. "\\sensGrid10" khong phai mot lenh hop le:
            # LaTeX doc \\sensGrid roi de "10" lai, va \\newcommand voi ten do lam vo
            # ca preamble. Ca 21/08: trang dau cua ban dung chi con 9 tu, va cong
            # trinh bay van bao 0 overfull 0 ref treo vi no khong doc dong "! Undefined
            # control sequence" trong log. Doi so sang chu.
            DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                   "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
            key = nm.replace("G", "Grid").replace("k", "Order").replace("h", "Width")
            key = "".join(DIG.get(c, c) for c in key)
            M["sens" + key] = "%.2f" % (100 * st.mean(mean5(d) for d in g))

    # gia tri tai eta=0,3 de van xuoi trich duoc, thay vi go tay
    for tg, ed in (("Bot", "lrext"), ("Ton", "lrext_toniot"),
                   ("Cse", "lrext_cseciic"), ("CseMc", "lrext_cseciic_mc50k")):
        for arch, an in (("kan8", "Kan"), ("mlp80", "Mlp")):
            for lab in ("03", "0p3"):
                g = glob.glob(str(ROOT / ("results/%s/lrext_%s_lr%s__dir0.1__seed*" % (ed, arch, lab))))
                if g:
                    M["ext%s%s" % (an, tg)] = "%.2f" % (100 * st.mean(mean5(d) for d in g))
                    break

    # 0.4 SCAFFOLD duoi SGD. Bai TU NEU gia thuyet "hong vi dung Adam" ma khong
    # kiem. Do roi: duoi SGD no TE HON, va sd = 0 nghia la moi seed dung o dung
    # mot diem. Gia thuyet cua bai SAI.
    for arch, an in (("kan8", "Kan"), ("mlp80", "Mlp")):
        g = glob.glob(str(ROOT / ("results/scaffold_sgd/scaffold_sgd_%s__*" % arch)))
        if g:
            v = [mean5(d) for d in g]
            M["scaffoldSgd" + an] = "%.2f" % (100 * st.mean(v))
            M["scaffoldSgdSd" + an] = "%.2f" % (100 * st.stdev(v))

    # 0.2 DA LOP tren CPU cho BoT-IoT va ToN-IoT. Truoc chang 0, hai o nay chi co
    # so RTX 4090, nen cau "moi thi nghiem chay tren MOT may" la SAI.
    for pre, tg in (("mc_botiot", "Bot"), ("mc_toniot", "Ton")):
        r = {}
        for d in glob.glob(str(ROOT / ("results/%s/*" % pre))):
            m = re.search(r"mc_(kan8|mlp80)_", d)
            if m:
                try:
                    r.setdefault(m.group(1), []).append(mean5(d))
                except Exception:
                    pass
        if "kan8" in r and "mlp80" in r:
            M["mcGap" + tg] = "%+.2f" % (100 * (st.mean(r["kan8"]) - st.mean(r["mlp80"])))

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
