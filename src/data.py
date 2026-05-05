"""Data generation, partitioning, and DataLoader factories.

For Sprint 1 only the synthetic dataset is implemented; real NetFlow loaders
(NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2) will be added in
Sprint 2 once the FL pipeline is verified end-to-end.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FederatedSplit:
    """Holds per-client train tensors plus a single pooled test set."""
    client_train: list[tuple[torch.Tensor, torch.Tensor]]
    test: tuple[torch.Tensor, torch.Tensor]
    n_features: int
    n_classes: int

    def loaders(self, batch_size: int) -> tuple[list[DataLoader], DataLoader]:
        train_loaders = [
            DataLoader(
                TensorDataset(x, y),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )
            for x, y in self.client_train
        ]
        test_loader = DataLoader(
            TensorDataset(*self.test), batch_size=batch_size, shuffle=False
        )
        return train_loaders, test_loader


def make_synthetic(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a tabular classification dataset with mild non-linearity.

    Each class is centered at a random Gaussian; we add a tanh feature
    transformation so KAN's spline activations have something non-trivial
    to fit beyond a pure linear separator.
    """
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


def dirichlet_partition(
    y: np.ndarray, n_clients: int, alpha: float, seed: int = 42
) -> list[np.ndarray]:
    """Partition indices among clients using class-wise Dirichlet sampling.

    Standard recipe: for each class c, sample p_c ~ Dir(alpha) over clients
    and split that class's indices proportionally.
    """
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * n_clients)
        # Cumulative split points
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        chunks = np.split(idx_c, splits)
        for k in range(n_clients):
            client_idx[k].extend(chunks[k].tolist())
    return [np.array(sorted(idx)) for idx in client_idx]


def build_federated_split(
    cfg_data: dict, seed: int
) -> FederatedSplit:
    name = cfg_data["name"]
    if name != "synthetic":
        raise NotImplementedError(
            f"Dataset '{name}' will be added in Sprint 2; only 'synthetic' is wired up."
        )

    X, y = make_synthetic(
        n_samples=cfg_data["n_samples"],
        n_features=cfg_data["n_features"],
        n_classes=cfg_data["n_classes"],
        seed=seed,
    )

    n = len(X)
    n_test = int(n * cfg_data["test_ratio"])
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    parts = dirichlet_partition(
        y_train,
        n_clients=cfg_data["n_clients"],
        alpha=cfg_data["alpha"],
        seed=seed,
    )
    client_train = [
        (torch.from_numpy(X_train[idx]), torch.from_numpy(y_train[idx]))
        for idx in parts
    ]
    return FederatedSplit(
        client_train=client_train,
        test=(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        n_features=cfg_data["n_features"],
        n_classes=cfg_data["n_classes"],
    )
