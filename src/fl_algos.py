"""R1#2 va R2#2: cac thuat toan lien ket manh hon FedAvg, va mot baseline NEN.

Phan bien 1 doi FedProx / SCAFFOLD / MOON / FedNova / FedDyn. Phan bien 2 doi it
nhat mot baseline nen (FedPAQ hoac top-k). Ban nop chi co FedAvg va mot bien the
FedProx cho MLP, nen so sanh "KAN hon MLP" thuc chat la "KAN duoi FedAvg hon MLP
duoi FedAvg", chua noi duoc gi ve KAN duoi mot thuat toan tot hon.

TACH FILE RIENG CO CHU DICH: duong FedAvg trong src/fl.py da sinh ra 310 run cu va
da duoc doi chieu, khong dung vao. File nay goi lai aggregate/evaluate/state_bytes
cua fl.py chu khong viet lai, de hai duong khong troi ve hai phia.

KE TOAN BYTE, cho khong ai lam tu dong duoc:
  fedavg, fedprox, moon : 1 x |Theta| moi client moi vong
  scaffold              : 2 x |Theta|  (gui ca Delta y VA Delta c)  <-- dat gap doi
  fedpaq (top-k)        : k x |Theta| x (4 byte gia tri + 4 byte chi so) / 4 byte
Neu bo qua cot nay thi SCAFFOLD trong nhu "thang mien phi", ma no khong mien phi.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .fl import aggregate, evaluate, state_bytes, _make_optimizer

ALGOS = ("fedavg", "fedprox", "scaffold", "moon", "fedpaq")


# --------------------------------------------------------------------- tien ich

def _sub(a: dict, b: dict) -> dict:
    return {k: (a[k].float() - b[k].float()) for k in a
            if a[k].dtype in (torch.float32, torch.float64, torch.float16)}


def _add_(a: dict, b: dict, scale: float = 1.0) -> None:
    for k in b:
        if k in a and a[k].dtype in (torch.float32, torch.float64, torch.float16):
            a[k] = (a[k].float() + scale * b[k].float()).to(a[k].dtype)


def _topk_mask(delta: dict, frac: float) -> tuple[dict, int]:
    """Giu frac phan tram phan tu lon nhat theo tri tuyet doi. Tra ve (delta thua, so phan tu giu)."""
    flat = torch.cat([v.flatten().abs() for v in delta.values()])
    n_keep = max(1, int(frac * flat.numel()))
    thr = torch.topk(flat, n_keep, largest=True).values.min()
    out, kept = {}, 0
    for k, v in delta.items():
        m = v.abs() >= thr
        out[k] = v * m
        kept += int(m.sum())
    return out, kept


def _penultimate(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bieu dien cho MOON: dau ra cua lop an cuoi.

    Ca KANClassifier lan MLP o day deu la nn.ModuleList/Sequential mot lop an, nen
    lay dau ra truoc lop cuoi. Neu cau truc khac thi tra ve logits, va MOON suy
    bien ve FedAvg cong mot so hang gan nhu hang so: truong hop do PHAI bao ra
    chu khong duoc im lang, vi mot MOON suy bien trong y het MOON vo dung.
    """
    layers = getattr(model, "layers", None) or getattr(model, "net", None)
    if layers is None:
        raise RuntimeError(
            "MOON: khong tim thay .layers/.net de lay bieu dien. Khong duoc lang le "
            "tra ve logits, vi mot MOON suy bien trong y het mot MOON vo dung.")
    mods = [m for m in layers if isinstance(m, nn.Module)]
    h = x
    # Lap dung nhu forward: moi lop an deu qua tanh, lop cuoi thi bo qua.
    for m in mods[:-1]:
        h = torch.tanh(m(h))
    return h


# ------------------------------------------------------------------ client step

def client_update_algo(model, loader, cfg_fl, device, *, algo, global_state,
                       c_global=None, c_local=None, prev_local_state=None,
                       global_model=None, model_ctor=None):
    model.train()
    optim = _make_optimizer(model, cfg_fl)
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg_fl.get("local_epochs", 1)
    mu = float(cfg_fl.get("prox_mu", 0.01))
    moon_mu = float(cfg_fl.get("moon_mu", 1.0))
    moon_tau = float(cfg_fl.get("moon_tau", 0.5))
    lr = float(cfg_fl.get("lr", 0.01))

    gref = {k: v.to(device).float() for k, v in global_state.items()
            if v.dtype in (torch.float32, torch.float64, torch.float16)}
    n_steps = 0

    # Mau AM cua MOON la mo hinh cuc bo VONG TRUOC. Vong dau chua co no, va o
    # vong do MOON that su chi con so hang duong: phai bao ra chu khong duoc coi
    # nhu da chay MOON du.
    prev_model = None
    if algo == "moon" and prev_local_state is not None and model_ctor is not None:
        prev_model = model_ctor().to(device)
        prev_model.load_state_dict(prev_local_state, strict=True)
        prev_model.eval()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)

            if algo == "fedprox":
                reg = 0.0
                for k, p in model.named_parameters():
                    if k in gref:
                        reg = reg + ((p - gref[k]) ** 2).sum()
                loss = loss + 0.5 * mu * reg

            elif algo == "moon" and global_model is not None:
                z = _penultimate(model, x)
                with torch.no_grad():
                    z_glob = _penultimate(global_model, x)
                    z_prev = _penultimate(prev_model, x) if prev_model is not None else None
                cos = nn.functional.cosine_similarity
                pos = cos(z, z_glob, dim=-1) / moon_tau
                if z_prev is not None:
                    neg = cos(z, z_prev, dim=-1) / moon_tau
                    con = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg)))
                else:
                    con = -torch.log(torch.sigmoid(pos))
                loss = loss + moon_mu * con.mean()

            loss.backward()

            if algo == "scaffold" and c_global is not None:
                with torch.no_grad():
                    for k, p in model.named_parameters():
                        if p.grad is not None and k in c_global:
                            p.grad.add_(c_global[k].to(device) - c_local[k].to(device))

            optim.step()
            n_steps += 1

    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    c_local_new = None
    if algo == "scaffold":
        # Option II cua Karimireddy va cs.: c_i^+ = c_i - c + (x - y_i)/(K*eta)
        c_local_new = {}
        for k in c_local:
            drift = (global_state[k].float() - new_state[k].float()) / max(1, n_steps) / lr
            c_local_new[k] = c_local[k].float() - c_global[k].float() + drift
    return new_state, c_local_new, n_steps


# ------------------------------------------------------------------- vong lien ket

def federated_train_algo(*, model_factory, train_loaders, test_loader, cfg_fl,
                         device, algo="fedavg", on_round_end=None):
    assert algo in ALGOS, f"algo la {algo}, phai thuoc {ALGOS}"
    rounds = cfg_fl["rounds"]
    fraction = cfg_fl.get("fraction", 1.0)
    topk_frac = float(cfg_fl.get("topk_frac", 0.1))
    K = len(train_loaders)
    rng = np.random.RandomState(0)

    global_state = {k: v.detach().cpu().clone()
                    for k, v in model_factory().state_dict().items()}
    theta_bytes = state_bytes(global_state)

    float_keys = [k for k, v in global_state.items()
                  if v.dtype in (torch.float32, torch.float64, torch.float16)]
    c_global = {k: torch.zeros_like(global_state[k].float()) for k in float_keys}
    c_locals = [{k: torch.zeros_like(global_state[k].float()) for k in float_keys}
                for _ in range(K)]
    prev_locals = [None] * K

    history = {"rounds": [], "comm_uplink_bytes": [], "comm_downlink_bytes": [],
               "wallclock_s": [], "metrics": []}

    for t in range(rounds):
        t0 = time.perf_counter()
        m = max(int(fraction * K), 1)
        selected = rng.choice(K, m, replace=False).tolist()

        gm = None
        if algo == "moon":
            gm = model_factory().to(device)
            gm.load_state_dict(global_state, strict=True)
            gm.eval()

        states, sizes, dc, kept_total = [], [], [], 0
        for k in selected:
            cm = model_factory().to(device)
            cm.load_state_dict(global_state, strict=True)
            new_state, c_new, _ = client_update_algo(
                cm, train_loaders[k], cfg_fl, device, algo=algo,
                global_state=global_state, c_global=c_global, c_local=c_locals[k],
                prev_local_state=prev_locals[k], global_model=gm,
                model_ctor=model_factory)

            if algo == "fedpaq":
                delta = _sub(new_state, global_state)
                sparse, kept = _topk_mask(delta, topk_frac)
                kept_total += kept
                sent = {kk: global_state[kk].float() + sparse[kk] for kk in sparse}
                for kk in new_state:
                    if kk not in sent:
                        sent[kk] = new_state[kk]
                new_state = {kk: v.to(global_state[kk].dtype) for kk, v in sent.items()}

            if algo == "scaffold":
                dc.append({kk: (c_new[kk] - c_locals[k][kk]) for kk in c_new})
                c_locals[k] = c_new
            if algo == "moon":
                prev_locals[k] = new_state

            states.append(new_state)
            sizes.append(len(train_loaders[k].dataset))

        global_state = aggregate(states, sizes)
        if algo == "scaffold" and dc:
            for kk in c_global:
                c_global[kk] = c_global[kk] + sum(d[kk] for d in dc) / K

        eval_model = model_factory().to(device)
        eval_model.load_state_dict(global_state, strict=True)
        m_round = evaluate(eval_model, test_loader, device)

        # ke toan byte theo dung thuat toan, khong dung chung mot con so
        if algo == "scaffold":
            up = 2 * m * theta_bytes
        elif algo == "fedpaq":
            up = kept_total * 8            # 4 byte gia tri + 4 byte chi so
        else:
            up = m * theta_bytes
        down = m * theta_bytes

        wallclock = time.perf_counter() - t0
        history["rounds"].append(t + 1)
        history["comm_uplink_bytes"].append(up)
        history["comm_downlink_bytes"].append(down)
        history["wallclock_s"].append(wallclock)
        history["metrics"].append(m_round)
        if on_round_end:
            on_round_end(t + 1, {"wallclock_s": wallclock, "comm_uplink_bytes": up,
                                 "comm_downlink_bytes": down, **m_round})

    return {"history": history, "final_state": global_state,
            "bytes_per_round_uplink": theta_bytes, "algo": algo}
