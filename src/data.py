"""Federated split factories: synthetic + NetFlow-v2 (BoT-IoT / ToN-IoT / CSE-CIC).

A `cfg_data` block looks like (real dataset):

    name: nf_botiot_v2
    mode: binary                  # or 'multiclass'
    test_ratio: 0.2
    alpha: 0.1
    n_clients: 10
    train_downsample_per_class: 50000   # optional
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


_NETFLOW_NAMES = ("nf_botiot_v2", "nf_toniot_v2", "nf_cseciic_v2")


@dataclass
class FederatedSplit:
    client_train: list[tuple[torch.Tensor, torch.Tensor]]
    test: tuple[torch.Tensor, torch.Tensor]
    n_features: int
    n_classes: int
    class_names: list[str]

    def loaders(self, batch_size: int) -> tuple[list[DataLoader], DataLoader]:
        train_loaders = [
            DataLoader(TensorDataset(x, y), batch_size=batch_size,
                       shuffle=True, drop_last=False)
            for x, y in self.client_train
        ]
        test_loader = DataLoader(
            TensorDataset(*self.test), batch_size=batch_size, shuffle=False
        )
        return train_loaders, test_loader


# -- Synthetic -----------------------------------------------------------------

def make_synthetic(
    n_samples: int, n_features: int, n_classes: int, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, n_features) * 1.5
    per_class = n_samples // n_classes
    Xs, ys = [], []
    for c in range(n_classes):
        x = rng.randn(per_class, n_features) * 0.7 + centers[c]
        x = np.tanh(x) + 0.2 * rng.randn(per_class, n_features)
        Xs.append(x.astype(np.float32))
        ys.append(np.full(per_class, c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# -- Stratified train/test split ----------------------------------------------

def stratified_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    tr = np.concatenate(train_idx); te = np.concatenate(test_idx)
    rng.shuffle(tr); rng.shuffle(te)
    return X[tr], y[tr], X[te], y[te]


# -- Dirichlet partition ------------------------------------------------------

def dirichlet_partition(
    y: np.ndarray, n_clients: int, alpha: float, seed: int = 42,
    min_per_client: int = 10,
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    while True:
        client_idx: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet([alpha] * n_clients)
            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            chunks = np.split(idx_c, splits)
            for k in range(n_clients):
                client_idx[k].extend(chunks[k].tolist())
        sizes = [len(idx) for idx in client_idx]
        if min(sizes) >= min_per_client:
            break  # otherwise resample partition
    return [np.array(sorted(idx)) for idx in client_idx]


# -- Driver -------------------------------------------------------------------

def build_federated_split(cfg_data: dict, seed: int) -> FederatedSplit:
    name = cfg_data["name"]
    mode = cfg_data.get("mode", "binary")

    if name == "synthetic":
        X, y = make_synthetic(
            n_samples=cfg_data["n_samples"],
            n_features=cfg_data["n_features"],
            n_classes=cfg_data["n_classes"],
            seed=seed,
        )
        class_names = [f"class_{i}" for i in range(cfg_data["n_classes"])]
        n_classes = cfg_data["n_classes"]

    elif name in _NETFLOW_NAMES:
        from .datasets import netflow_v2
        ds = netflow_v2.load_cached(name)
        X = ds.X
        if mode == "multiclass":
            y = ds.y_multiclass
            class_names = ds.class_names
            n_classes = ds.n_classes_multiclass
        elif mode == "binary":
            y = ds.y_binary
            class_names = ["benign", "attack"]
            n_classes = 2
        else:
            raise ValueError(f"mode must be 'binary' or 'multiclass'; got {mode}")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Optional class-wise downsample of TRAIN+TEST pool (test is split *after*).
    cap = cfg_data.get("downsample_per_class")
    if cap:
        from .datasets.netflow_v2 import maybe_downsample
        X, y = maybe_downsample(X, y, cap, seed=seed)

    X_tr, y_tr, X_te, y_te = stratified_split(
        X, y, test_ratio=cfg_data.get("test_ratio", 0.2), seed=seed
    )

    parts = dirichlet_partition(
        y_tr,
        n_clients=cfg_data["n_clients"],
        alpha=cfg_data["alpha"],
        seed=seed,
    )
    client_train = [
        (torch.from_numpy(X_tr[idx]), torch.from_numpy(y_tr[idx])) for idx in parts
    ]
    return FederatedSplit(
        client_train=client_train,
        test=(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        n_features=int(X.shape[1]),
        n_classes=int(n_classes),
        class_names=class_names,
    )
