# FedKAN-IDS-v2

Federated Kolmogorov-Arnold Networks for NetFlow-based IoT Intrusion Detection — systematic empirical study targeting IEEE Internet of Things Journal.

## Status
Repository under active development. Skeleton + baselines being staged.

## Layout (planned)
```
configs/      YAML experiment configs
src/          data loaders, models, FL training, profiling, viz
scripts/      CLI entry points (run_experiment, aggregate, make_figures)
notebooks/    Colab entry-point notebooks
results/      raw runs (committed) + aggregated CSVs + final PDF figures
paper/        LaTeX sources
```

## Workflow
1. Code is developed locally; committed to `main`.
2. Experiments are run on Google Colab Pro+ (A100); results are committed back from the Colab session.
3. Figures and statistical tables are produced from `results/` via `scripts/make_figures.py`.

## Citation
TBD upon publication.
