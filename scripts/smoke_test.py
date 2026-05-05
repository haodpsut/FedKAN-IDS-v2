"""Quick offline sanity check (no network, no real data).

Trains FedKAN for 3 rounds on a tiny synthetic split and asserts that the
final accuracy clears a low bar. Intended for CI and Colab first-cell.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import set_seed, get_device  # noqa: E402
from src.data import build_federated_split  # noqa: E402
from src.models import build_model, count_params  # noqa: E402
from src.fl import federated_train  # noqa: E402


def run(model_name: str) -> float:
    set_seed(42)
    device = get_device("auto")
    cfg_data = dict(name="synthetic", n_samples=1000, n_features=8,
                    n_classes=2, test_ratio=0.2, alpha=0.5, n_clients=2)
    cfg_fl = dict(rounds=3, fraction=1.0, local_epochs=1, batch_size=128,
                  lr=0.01, optimizer="adam", weight_decay=0.0)
    cfg_model = (dict(name="kan", hidden=[8], grid_size=5, spline_order=3)
                 if model_name == "kan" else dict(name="mlp", hidden=[16]))

    split = build_federated_split(cfg_data, seed=42)
    factory = lambda: build_model(cfg_model, in_dim=split.n_features,
                                  out_dim=split.n_classes)
    n = count_params(factory())
    train_loaders, test_loader = split.loaders(cfg_fl["batch_size"])
    res = federated_train(model_factory=factory, train_loaders=train_loaders,
                          test_loader=test_loader, cfg_fl=cfg_fl, device=device)
    final_acc = res["history"]["metrics"][-1]["accuracy"]
    print(f"  [{model_name}] params={n:,}  final_acc={final_acc:.4f}  "
          f"bytes/round={res['bytes_per_round_uplink']:,}")
    return final_acc


def main():
    print("Smoke test — verifies FL pipeline runs end to end.")
    acc_kan = run("kan")
    acc_mlp = run("mlp")
    assert acc_kan > 0.6, f"KAN smoke acc too low: {acc_kan}"
    assert acc_mlp > 0.6, f"MLP smoke acc too low: {acc_mlp}"
    print("OK — both KAN and MLP reach > 0.60 in 3 rounds on synthetic data.")


if __name__ == "__main__":
    main()
