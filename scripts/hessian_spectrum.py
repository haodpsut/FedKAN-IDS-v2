"""R2#6: bang chung thuc nghiem cho gia thiet loi CUC BO cua Dinh ly 1.

Phan bien 2 doi "empirical evidence (e.g. Hessian spectrum analysis) supporting
local convexity assumptions". Dinh ly gia thiet L-tron va mu-loi manh; khong ai
tin dieu do dung TOAN CUC voi cross-entropy, va chung ta da thua nhan o R2#1.
Cau hoi con lai la: o LAN CAN nghiem tim duoc, pho Hessian trong the nao?

Do cai gi, va vi sao:
  lambda_max  : hang so tron L cuc bo. Co trong cong thuc cua dinh ly.
  lambda_min  : neu > 0 thi loi manh cuc bo, va mu = lambda_min.
  ty le am    : phan tram tri rieng am. >0 nghia la diem tim duoc la YEN NGUA,
                khong phai cuc tieu, va gia thiet loi manh sai NGAY CA cuc bo.
  L/mu        : so dieu kien; cận cua dinh ly ty le voi L/mu^2 nen so nay quyet
                dinh cận co mang thong tin hay khong.

DIEU KIEN LANH MANH: do tren CA HAI kien truc. Neu chi do KAN thi khong biet con
so la dac trung cua KAN hay la dac trung cua bai toan. Neu hai ben giong nhau thi
pho Hessian KHONG giai thich duoc chenh lech nao giua chung, va phai noi ra.

Ky thuat: Hessian day cua 3,3k tham so la 3,3k x 3,3k = 10,8 trieu phan tu, tinh
duoc nhung cham. Dung Hutchinson + Lanczos qua torch.autograd de lay pho gan dung
voi so lan nhan Hessian-vector vua phai.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed, get_device  # noqa: E402
from src.data import build_federated_split               # noqa: E402
from src.models import build_model, count_params         # noqa: E402
from src.fl import federated_train                       # noqa: E402


def hvp(loss_fn, params, v):
    """Tich Hessian-vector bang vi phan tu dong hai lan."""
    loss = loss_fn()
    g = torch.autograd.grad(loss, params, create_graph=True)
    flat_g = torch.cat([x.reshape(-1) for x in g])
    gv = (flat_g * v).sum()
    h = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([x.reshape(-1) for x in h]).detach()


def lanczos(loss_fn, params, n_dim, m=40, seed=0):
    """Lanczos khong tai truc giao hoa: tra ve m tri rieng Ritz."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_dim, generator=g)
    q = q / q.norm()
    Q, alphas, betas = [q], [], []
    beta = 0.0
    q_prev = torch.zeros(n_dim)
    for i in range(m):
        w = hvp(loss_fn, params, Q[-1])
        alpha = torch.dot(w, Q[-1]).item()
        w = w - alpha * Q[-1] - beta * q_prev
        # tai truc giao hoa day du: m nho nen chap nhan duoc, va no chan hien tuong
        # tri rieng ma cua Lanczos, thu se lam ty le am trong lon hon that
        for u in Q:
            w = w - torch.dot(w, u) * u
        beta = w.norm().item()
        alphas.append(alpha)
        if beta < 1e-8 or i == m - 1:
            break
        betas.append(beta)
        q_prev = Q[-1]
        Q.append(w / beta)
    T = np.diag(alphas)
    for i, b in enumerate(betas):
        T[i, i + 1] = T[i + 1, i] = b
    return np.linalg.eigvalsh(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/e1_botiot.yaml")
    ap.add_argument("--mode", default="binary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--lanczos", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="results/hessian.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    cfg["data"]["mode"] = args.mode
    cfg["fl"]["rounds"] = args.rounds
    dev = get_device("cpu")

    ARCHS = [("FedKAN-8", dict(name="kan", hidden=[8], grid_size=5, spline_order=3)),
             ("FedAvg-MLP-80", dict(name="mlp", hidden=[80]))]

    print("=" * 88)
    print("R2#6: PHO HESSIAN TAI NGHIEM TIM DUOC  (%s, %s, %d vong)"
          % (cfg["data"]["name"], args.mode, args.rounds))
    print("=" * 88)
    print("  %-16s %8s %12s %12s %10s %12s" % (
        "kien truc", "tham so", "lambda_max", "lambda_min", "%tri am", "L/|mu|"))
    print("  " + "-" * 76)

    rows = []
    for label, mcfg in ARCHS:
        set_seed(args.seed)
        split = build_federated_split(cfg["data"], seed=args.seed)
        tl, te = split.loaders(cfg["fl"]["batch_size"])
        cfg["model"] = mcfg

        def factory():
            torch.manual_seed(1234)
            return build_model(mcfg, in_dim=split.n_features, out_dim=split.n_classes)

        res = federated_train(model_factory=factory, train_loaders=tl, test_loader=te,
                              cfg_fl=cfg["fl"], device=dev)
        model = factory().to(dev)
        model.load_state_dict(res["final_state"], strict=True)
        model.eval()

        X, y = split.test
        idx = torch.randperm(len(X))[:args.batch]
        xb, yb = X[idx].to(dev), y[idx].to(dev)
        ce = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        n_dim = sum(p.numel() for p in params)

        ev = lanczos(lambda: ce(model(xb), yb), params, n_dim, m=args.lanczos, seed=args.seed)
        lmax, lmin = float(ev.max()), float(ev.min())
        neg = 100.0 * float((ev < -1e-8).sum()) / len(ev)
        cond = abs(lmax / lmin) if abs(lmin) > 1e-12 else float("inf")
        print("  %-16s %8d %12.4e %12.4e %9.1f%% %12.3g" % (
            label, count_params(model), lmax, lmin, neg, cond))
        rows.append({"model": label, "params": count_params(model), "lambda_max": lmax,
                     "lambda_min": lmin, "pct_negative": neg, "cond": cond,
                     "eigs": [float(x) for x in ev]})

    print()
    any_neg = any(r["pct_negative"] > 0 for r in rows)
    if any_neg:
        print("  DOC THE NAO: co tri rieng AM ⇒ diem hoi tu la YEN NGUA, khong phai cuc tieu.")
        print("  Gia thiet loi manh cua Dinh ly 1 sai NGAY CA o lan can, va bai phai noi vay.")
    else:
        print("  DOC THE NAO: khong co tri rieng am trong %d huong Ritz ⇒ CHUA BAC BO duoc" % args.lanczos)
        print("  loi cuc bo. Day KHONG phai chung minh loi manh: Lanczos chi tham do mot khong")
        print("  gian con, va vang mat bang chung khong phai bang chung vang mat.")
    if len(rows) == 2:
        print("  Hai kien truc: L/|mu| = %.3g so voi %.3g. %s"
              % (rows[0]["cond"], rows[1]["cond"],
                 "Khac biet dang ke." if max(rows[0]["cond"], rows[1]["cond"]) >
                 3 * min(rows[0]["cond"], rows[1]["cond"]) else
                 "Cung bac ⇒ pho Hessian KHONG giai thich chenh lech nao giua chung."))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": args.config, "mode": args.mode, "seed": args.seed,
               "lanczos_m": args.lanczos, "rows": rows}, open(args.out, "w"), indent=2)
    print("\n  da ghi %s" % args.out)


if __name__ == "__main__":
    main()
