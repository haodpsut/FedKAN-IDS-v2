# FedKAN-IDS-v2

Federated Kolmogorov-Arnold Networks for NetFlow-based IoT Intrusion Detection — systematic empirical study targeting **IEEE Internet of Things Journal**.

## Status

**Sprint 2 (M1) — real datasets.** Three NetFlow-v2 datasets are wired up via Kaggle download + parquet caching. Binary and multi-class modes both supported. Additional baselines (F-KAN, 1D-CNN, DeepFed, FedPAQ, top-k SGD) and profiling land in M2.

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

### One-time setup

1. **GitHub PAT** with `repo` scope (https://github.com/settings/tokens) — paste in cell 3 every session.
2. **Kaggle API token** — go to https://www.kaggle.com/settings → "Create New Token" → save the resulting `kaggle.json` to your Drive at `MyDrive/secrets/kaggle.json`. Cell A1 in the notebook copies it into place.

### Data preparation

Cell A2 runs `scripts/prepare_data.py` which:
1. Calls the Kaggle CLI to download the dataset CSV under `data/raw/<name>/`.
2. Drops identifier columns, encodes labels, Min-Max scales features.
3. Writes a parquet cache to `data/cache/<name>.parquet` plus a YAML metadata file.

Cache lives on the Colab disk (not Drive), so it gets rebuilt when the runtime is fresh — that takes 1–3 minutes per dataset.

## Workflow

1. Code is developed locally and pushed to `main`.
2. Experiments are launched on Google Colab Pro+ (A100) via the notebook above.
3. Each `(config, seed)` writes one directory under `results/runs/`; the notebook commits these back.
4. Aggregation and figure generation run locally from `results/aggregated/` (added in a later sprint).

## Citation

To appear.
