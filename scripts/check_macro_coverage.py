"""Chong tai phat: con so nao DA CO macro thi khong duoc go tay lai vao bai.

Refactor 20/08/2026 dua 71 con so ve mot cho o. Nhung refactor mot lan khong giu
duoc: buoi viet sau, nguoi viet (hoac AI) go thang "$+0.76$" vi no nhanh hon go
"\\gapTunedBot", va sau vai lan sua giao thuc thi bai lai lech. Cong nay chan dung
duong quay lai do.

CACH LAM. Doc macros.tex, lay tap gia tri. Quet cac tep .tex cua bai, tim so viet
truc tiep. Neu mot so trung gia tri voi mot macro thi bao: cho do phai dung macro.

BA DIEU CO Y KHONG LAM:
  - Khong doi HET moi con so phai la macro. So chi xuat hien mot lan va khong doi
    theo giao thuc (vi du "50 rounds", "10 clients") thi go tay la hop ly.
  - Khong doc trong cac tep results/tables/*.tex, chung la dau ra sinh tu dong.
  - Khong tinh cac so trong \\label, \\ref, \\cite.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", required=True)
    ap.add_argument("--tex", nargs="+", required=True)
    a = ap.parse_args()

    src = Path(a.macros).read_text(encoding="utf-8")
    mac = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", src):
        name, val = m.group(1), m.group(2)
        v = val.replace("{,}", ",").strip()
        if re.fullmatch(r"[+-]?[\d,]+(?:\.\d+)?", v):
            mac.setdefault(v, []).append(name)
    print("  doc %d macro, %d gia tri phan biet" % (len(re.findall(r"newcommand", src)), len(mac)))

    hits = []
    for f in a.tex:
        p = Path(f)
        if not p.exists() or "results/tables" in str(p):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            # MIEN TRU CO LY DO. Cong nay khop theo GIA TRI nen no bao dong gia khi
            # hai dai luong tinh co bang nhau: ca IoT-J 20/08, "$6.72$" vua la do trai
            # lr (pp) vua la luong truyen (MB), va ap may moc thi hong 5 cho. Dong nao
            # co "% not-a-macro: <ly do>" thi bo qua. BAT BUOC ghi ly do -- cong do
            # vinh vien se bi lo di, con mien tru khong ly do thi khong kiem lai duoc.
            if re.search(r"%\s*not-a-macro:\s*\S", line):
                continue
            clean = re.sub(r"\\(?:label|ref|cite|eqref|input)\{[^}]*\}", "", line)
            for m in re.finditer(r"\$([+-]?\d+(?:\{,\})?\d*\.\d+)\$", clean):
                v = m.group(1).replace("{,}", ",")
                for cand in (v, v.lstrip("+")):
                    if cand in mac:
                        hits.append((str(p), i, m.group(1), mac[cand][0]))
                        break

    if hits:
        print("\n  ⛔ %d con so go tay trong khi DA CO macro cho no:" % len(hits))
        for f, i, lit, name in hits[:40]:
            print("     %s:%d   $%s$  ->  \\%s" % (f.split("/")[-1], i, lit, name))
        if len(hits) > 40:
            print("     ... va %d cho nua" % (len(hits) - 40))
        print("\n     Vi sao chan: con so co hai cho o thi lan doi giao thuc sau se lech mot")
        print("     trong hai. Do dung la co che da gay 5 loi tu mau thuan ngay 20/08.")
    else:
        print("\n  ✅ khong con so nao bi go tay trong khi da co macro")

    print("\n  => %s" % ("FAIL: %d cho" % len(hits) if hits else "PASS"))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
