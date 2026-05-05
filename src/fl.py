"""Federated Averaging server + client + driver loop."""
from __future__ import annotations
import copy
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .models import state_bytes


def _make_optimizer(model: nn.Module, cfg_fl: dict) -> torch.optim.Optimizer:
    name = cfg_fl.get("optimizer", "adam").lower()
    lr = cfg_fl["lr"]
    wd = cfg_fl.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def client_update(
    model: nn.Module,
    loader: DataLoader,
    cfg_fl: dict,
    device: torch.device,
) -> dict:
    """Run E local epochs of SGD, return the updated state_dict (CPU tensors)."""
    model.train()
    optim = _make_optimizer(model, cfg_fl)
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg_fl.get("local_epochs", 1)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def aggregate(states: list[dict], weights: list[float]) -> dict:
    """Weighted element-wise average of state_dicts."""
    total = sum(weights)
    norm = [w / total for w in weights]
    out = {}
    for k in states[0]:
        # Skip non-floating buffers (e.g. integer counts) by passing them through.
        if states[0][k].dtype in (torch.float32, torch.float64, torch.float16):
            stack = torch.stack([s[k].float() * w for s, w in zip(states, norm)], 0)
            out[k] = stack.sum(0).to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_pred.append(logits.argmax(1).cpu().numpy())
        all_true.append(y.numpy())
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def federated_train(
    *,
    model_factory: Callable[[], nn.Module],
    train_loaders: list[DataLoader],
    test_loader: DataLoader,
    cfg_fl: dict,
    device: torch.device,
    on_round_end: Callable[[int, dict], None] | None = None,
) -> dict:
    """Synchronous FedAvg loop.

    model_factory builds a fresh client model whose state we overwrite from
    the global state at the start of each round — this keeps the client free
    of optimizer history between rounds.
    """
    rounds = cfg_fl["rounds"]
    fraction = cfg_fl.get("fraction", 1.0)
    K = len(train_loaders)
    rng = np.random.RandomState(0)

    # Initialise global state from a fresh model.
    global_state = {k: v.detach().cpu().clone() for k, v in model_factory().state_dict().items()}
    bytes_per_round_uplink = state_bytes(global_state)

    history = {"rounds": [], "comm_uplink_bytes": [], "comm_downlink_bytes": [],
               "wallclock_s": [], "metrics": []}

    for t in range(rounds):
        t0 = time.perf_counter()
        m = max(int(fraction * K), 1)
        selected = rng.choice(K, m, replace=False).tolist()

        states, sizes = [], []
        for k in selected:
            cm = model_factory().to(device)
            cm.load_state_dict(global_state, strict=True)
            new_state = client_update(cm, train_loaders[k], cfg_fl, device)
            states.append(new_state)
            sizes.append(len(train_loaders[k].dataset))

        global_state = aggregate(states, sizes)

        eval_model = model_factory().to(device)
        eval_model.load_state_dict(global_state, strict=True)
        m_round = evaluate(eval_model, test_loader, device)

        wallclock = time.perf_counter() - t0
        comm_uplink = m * bytes_per_round_uplink
        comm_downlink = m * bytes_per_round_uplink

        history["rounds"].append(t + 1)
        history["comm_uplink_bytes"].append(comm_uplink)
        history["comm_downlink_bytes"].append(comm_downlink)
        history["wallclock_s"].append(wallclock)
        history["metrics"].append(m_round)

        if on_round_end:
            on_round_end(t + 1, {
                "wallclock_s": wallclock,
                "comm_uplink_bytes": comm_uplink,
                "comm_downlink_bytes": comm_downlink,
                **m_round,
            })

    return {
        "history": history,
        "final_state": global_state,
        "bytes_per_round_uplink": bytes_per_round_uplink,
    }
