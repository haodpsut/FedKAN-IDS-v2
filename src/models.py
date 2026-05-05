"""Model factory: KAN, MLP. CNN-1D and CNN-GRU added in Sprint 2."""
from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int, hidden: Iterable[int], out_dim: int
) -> nn.Module:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def build_kan(
    in_dim: int,
    hidden: Iterable[int],
    out_dim: int,
    grid_size: int = 5,
    spline_order: int = 3,
) -> nn.Module:
    """Build a KAN via the efficient_kan library.

    efficient_kan.KAN expects a width vector [in, h1, h2, ..., out].
    """
    from efficient_kan import KAN  # imported lazily to keep tests light

    widths = [in_dim, *list(hidden), out_dim]
    return KAN(widths, grid_size=grid_size, spline_order=spline_order)


def build_model(cfg_model: dict, in_dim: int, out_dim: int) -> nn.Module:
    name = cfg_model["name"].lower()
    hidden = cfg_model.get("hidden", [32])
    if name == "mlp":
        return build_mlp(in_dim, hidden, out_dim)
    if name == "kan":
        return build_kan(
            in_dim,
            hidden,
            out_dim,
            grid_size=cfg_model.get("grid_size", 5),
            spline_order=cfg_model.get("spline_order", 3),
        )
    raise ValueError(f"Unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_bytes(state_dict: dict, dtype_bytes: int = 4) -> int:
    """Approximate uplink size assuming float32 (4 B per param)."""
    return sum(t.numel() for t in state_dict.values()) * dtype_bytes
