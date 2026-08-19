"""MOT cho duy nhat dinh nghia kieu ve cho MOI hinh cua bai.

Ban nop truoc co ba script ve, moi script tu dat rcParams va tu chon mau, nen
cung mot kien truc doi mau giua cac hinh va nguoi doc phai tra chu giai lai o
tung hinh. R1#7 goi do la van de trinh bay; goc re la kieu ve co BA cho cai dat.
Dat mot cho, ba script import ve.

Bang mau Okabe-Ito: an toan voi ba dang mu mau pho bien, va van phan biet duoc
khi in den trang vi do sang khac nhau. Kem theo KIEU NET rieng cho tung phuong
phap, de ban in den trang khong phai doan.
"""
import matplotlib as mpl

# Co chu tinh cho hinh SAU khi thu ve mot cot IEEE (~3,5 inch). Ve o 4,5 inch roi
# thu con 3,5 inch thi moi so nho di 1,3 lan, do la ly do ban cu bi che la chu nho.
RC = {
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14, "axes.titlesize": 14, "legend.fontsize": 11,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2, "lines.markersize": 5,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.92, "legend.edgecolor": "0.6",
    "savefig.bbox": "tight", "savefig.dpi": 300,
}

VARIANT_STYLE = {
    "kan_h8":     dict(color="#0072B2", marker="o", ls="-",  label="FedKAN (Ours, 8h)"),
    "kan_h16":    dict(color="#009E73", marker="s", ls="--", label="F-KAN (16h)"),
    "mlp_h32":    dict(color="#D55E00", marker="^", ls="-.", label="FedAvg-MLP (32h)"),
    "mlp_h80":    dict(color="#CC79A7", marker="D", ls=":",  label="FedAvg-MLP-PM (80h)"),
    "kan_h16x16": dict(color="#E69F00", marker="v", ls="-",  label="F-KAN (16x16)"),
}
FALLBACK = ["#56B4E9", "#F0E442", "#999999", "#000000"]

# Mau cho cot "hieu so", tach khoi mau phuong phap de khong ai doc nham mot cot
# hieu so thanh mot kien truc.
DIFF_MEAN = "#0072B2"
DIFF_WORST = "#E69F00"


def apply():
    mpl.rcParams.update(RC)


def style_for(variant, idx=0):
    if variant in VARIANT_STYLE:
        return VARIANT_STYLE[variant]
    return dict(color=FALLBACK[idx % len(FALLBACK)], marker="x", ls="-", label=variant)
