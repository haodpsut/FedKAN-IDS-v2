# FedKAN-IDS-v2

Federated Kolmogorov-Arnold Networks for NetFlow-based IoT Intrusion Detection — systematic empirical study targeting **IEEE Internet of Things Journal**.

## Status

**Sprint 1 — pipeline scaffold (current).** End-to-end FL training loop runs on synthetic data; KAN and MLP supported. Real datasets (NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2) and additional baselines (F-KAN, 1D-CNN, DeepFed, FedPAQ, top-k SGD) land in Sprint 2.

## Layout

```
configs/                 # YAML experiment configs
  base.yaml
  experiments/
    smoke.yaml           # 5-round sanity check
    smoke_mlp.yaml
src/                     # library code
  data.py                # synthetic dataset, Dirichlet partitioning
  models.py              # KAN / MLP factory, parameter counters
  fl.py                  # FedAvg server, client, driver
  metrics.py             # accuracy, F1, per-class
  utils.py               # config / seed / IO helpers
scripts/
  run_experiment.py      # CLI: --config <yaml> --seed <int>
  smoke_test.py          # offline 3-round assert test
notebooks/
  10_run_batch.ipynb     # Colab Pro+ entry point
results/                 # raw runs (committed)
  runs/{exp_id}__seed{N}/
    config_snapshot.yaml
    per_round.csv
    metrics.json
```

## Local quickstart

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_experiment.py --config configs/experiments/smoke.yaml --seed 42
```

## Colab quickstart

Open `notebooks/10_run_batch.ipynb` in Colab → run cells top to bottom. Cell 3 prompts for a GitHub PAT (kept in memory only).

## Workflow

1. Code is developed locally and pushed to `main`.
2. Experiments are launched on Google Colab Pro+ (A100) via the notebook above.
3. Each `(config, seed)` writes one directory under `results/runs/`; the notebook commits these back.
4. Aggregation and figure generation run locally from `results/aggregated/` (added in a later sprint).

## Citation

To appear.
