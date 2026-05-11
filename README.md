# FedKAN-IDS-v2

**Federated Kolmogorov-Arnold Networks for NetFlow-based IoT Intrusion Detection** — a systematic, parameter-matched empirical study across three standardised NetFlow-v2 datasets, anchored in a convergence analysis that introduces a spline-bias term independent of the data-heterogeneity factor.

Targeting **IEEE Internet of Things Journal**.

## What this repo contains

- **Library code** for federated learning with KAN and MLP clients (`src/`).
- **CLI runners** for single experiments and full grids (`scripts/`).
- **Per-run metrics** for all 289 reproducible runs (`results/runs/`) used in the paper.
- **Auto-generated tables and figures** that the paper `\input`s directly from `results/tables/` and `\includegraphics`es from `results/figures/`.
- **Full LaTeX manuscript** (`paper/`) — `main.pdf` builds from the same `results/` tree.

## Headline result

Across **three NetFlow-v2 datasets** evaluated with **n = 10 seeds**, the parameter-matched FedKAN-8 (~3.3k params) versus FedAvg-MLP-PM-80 (~3.4k params) under extreme client heterogeneity (Dirichlet α = 0.1):

| Dataset | Binary mean Δ | Worst-seed Δ | Multi-class mean Δ |
|---|---|---|---|
| NF-BoT-IoT-v2 | **+6.00 pp** (CI [+0.50, +14.57] excludes 0) | **+24.24 pp** | tied |
| NF-ToN-IoT-v2 | +5.08 pp (CI [-0.29, +9.93]) | -11.07 pp (seed-17 outlier) | tied |
| NF-CSE-CIC-IDS2018-v2 | tied (-0.38 pp) | -21.23 pp (seed-17 outlier) | **+1.73 pp** (CI [+0.73, +2.74] excludes 0, p=0.012) |

Two complementary advantage patterns emerge: KAN's binary-detection advantage tracks class-distribution skew, while its multi-class advantage scales oppositely with the number of attack classes. A reproducible failure mode (seed 17's Dirichlet partition) is identified, with two mitigations confirmed (larger hidden width, richer spline grid).

## Layout

```
configs/
  base.yaml                              # default hyperparameters
  datasets/{nf_botiot_v2, nf_toniot_v2, nf_cseciic_v2}.yaml
  experiments/
    e1_botiot.yaml                       # base grid config (overridden per dataset)
    smoke.yaml, smoke_mlp.yaml           # 5-round sanity checks
    e1_mini_botiot_*.yaml                # legacy 3-seed configs
src/
  data.py            # synthetic + NetFlow dataset dispatch, Dirichlet + IID partitioning
  datasets/
    netflow_v2.py    # Sarhan-v2 family loader: download, preprocess, parquet cache
  models.py          # KAN (via efficient_kan), MLP factories; param counters
  fl.py              # FedAvg server, client, driver loop with byte accounting
  metrics.py         # accuracy, F1 (macro / weighted / per-class)
  utils.py           # config / seed / IO helpers
scripts/
  prepare_data.py                  # one-time Kaggle download + preprocess
  inspect_data.py                  # diagnostic: label distribution + feature stats
  run_experiment.py                # CLI: --config <yaml> --seed <int> + many overrides
  run_grid.sh                      # bash equivalent of the notebook for workstation use
  smoke_test.py                    # offline 3-round assertion
  aggregate.py                     # summary CSV from per-run metrics
  stats_tests.py                   # Welch's t-test, paired t-test, bootstrap CI, Cohen d
  make_latex_tables.py             # headline + worst-seed LaTeX tables
  build_cross_dataset_table.py     # cross-dataset replication LaTeX table
  plot_convergence.py              # per-dataset convergence figures
  plot_perclass.py                 # multi-class per-class F1 figures
  plot_cross_dataset.py            # 3-dataset cross-dataset figures (3 of them)
  rebuild_paper_artifacts.sh       # one command: regenerate every table/figure/PDF
notebooks/
  10_run_batch.ipynb               # Colab Pro+ entry point (NF-BoT-IoT-v2 grid)
  20_run_m3c_toniot.ipynb          # Colab entry point (NF-ToN-IoT-v2 grid)
docs/
  SERVER_SETUP.md                  # conda environment recipe for the RTX 4090 setup
results/
  runs/{exp_id}__{partition}__seed{N}/
    config_snapshot.yaml           # resolved config at run time
    per_round.csv                  # per-round acc, F1, comm cost, wall-clock
    metrics.json                   # final metrics + run-level metadata
  aggregated/
    summary.csv                    # one row per run
    summary_by_cell.csv            # mean/std aggregated per (variant, mode, partition)
    stats_tests_{botiot,toniot,cseciic}.txt
  tables/                          # auto-generated LaTeX tables \input'd by the paper
  figures/                         # auto-generated PDFs \includegraphics'd by the paper
paper/
  main.tex                         # IEEEtran[journal] entry point
  references.bib                   # 16 references including all 7 KAN-FL papers
  sections/{01..06}*.tex           # one file per section
  main.pdf                         # latest build (12 pages, 528 KB, 0 warnings)
  README.md                        # paper-build instructions
```

## Hardware

All experiments reported in the paper were executed on a workstation-class **NVIDIA RTX 4090 24GB** managed via the `conda` environment described in [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md). The Colab notebooks (`notebooks/10_run_batch.ipynb`, `notebooks/20_run_m3c_toniot.ipynb`) are also functional and are provided as a fallback for users without a dedicated GPU, but the wall-clock numbers in the paper refer to the RTX 4090 setup.

Per-round wall-clock on the RTX 4090 (excluding the first-round CUDA warm-up):
- FedKAN: 1.8–2.5 s per round
- MLP-PM: 1.5–2.1 s per round

A single 50-round, 10-client run completes in ~2–3 minutes.

## Quickstart

### 1. Environment (RTX 4090 workstation, conda-only)

Follow [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md). Summary:
```bash
conda create -n fedkan python=3.11 -y
conda activate fedkan
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install numpy pandas pyarrow scipy scikit-learn matplotlib seaborn pyyaml tqdm thop kaggle
pip install "git+https://github.com/Blealtan/efficient-kan.git@master"
```

Then verify:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/smoke_test.py    # synthetic 3-round assertion test
```

### 2. Prepare datasets (one-time, ~10 minutes total)

Put `kaggle.json` at `~/.kaggle/kaggle.json` (Kaggle → Settings → Create Legacy API Key). Then:
```bash
python scripts/prepare_data.py --dataset nf_botiot_v2
python scripts/inspect_data.py  --dataset nf_botiot_v2   # sanity-check label + feature distributions
python scripts/prepare_data.py --dataset nf_toniot_v2
python scripts/inspect_data.py  --dataset nf_toniot_v2
python scripts/prepare_data.py --dataset nf_cseciic_v2
python scripts/inspect_data.py  --dataset nf_cseciic_v2
```

The preprocessing pipeline drops the four ID columns, replaces ±inf with NaN, clips to the [0.0001, 0.9999] per-column quantile band, applies `log1p` compression to handle heavy-tail byte/packet counts, and Min-Max scales to [0, 1]. Output: a parquet cache in `data/cache/`.

### 3. Run experiments

The bash runner is the recommended path. Examples:
```bash
DATASET=botiot bash scripts/run_grid.sh                    # full 72-run grid on NF-BoT-IoT-v2
DATASET=toniot MINIMAL=1 bash scripts/run_grid.sh          # only Dir(α=0.1), 24 runs on NF-ToN-IoT-v2
DATASET=toniot FILL_M3A=1 bash scripts/run_grid.sh         # +7 seeds for n=10 statistics
DATASET=cseciic FILL_M3A=1 bash scripts/run_grid.sh        # 56 additional NF-CSE-CIC-IDS2018-v2 runs
```

The runner auto-commits and pushes every 4 runs (with retry on push failure), so a disconnect costs at most ~10 minutes of work. `--skip-existing` is enabled, so re-running the same command resumes from where it left off.

For a single experiment without the bash wrapper:
```bash
python scripts/run_experiment.py \
    --config configs/experiments/e1_botiot.yaml \
    --seed 42 --model-name kan --hidden 8 --grid-size 5 \
    --mode binary --downsample 130000 \
    --partition dirichlet --alpha 0.1
```

### 4. Rebuild paper

After new runs land:
```bash
bash scripts/rebuild_paper_artifacts.sh
```

This regenerates `results/aggregated/`, `results/tables/`, `results/figures/`, and compiles `paper/main.pdf` (`pdflatex + bibtex + pdflatex × 2`). Local TeX Live or Overleaf both work.

## Citation

```bibtex
@misc{do2026fedkanids,
  title={{FedKAN-IDS}: Heterogeneity-Robust Federated {Kolmogorov-Arnold} Networks for {NetFlow}-Based {IoT} Intrusion Detection},
  author={Do, Phuc Hao and Nguyen, Van Long and Le, Tran Duc and Dinh, Truong Duy},
  year={2026},
  note={Manuscript under review at IEEE Internet of Things Journal.}
}
```

## Authors

- Phuc Hao Do
- Van Long Nguyen
- Tran Duc Le
- Truong Duy Dinh (corresponding author, <duydt@ptit.edu.vn>)

## License

Code released under the MIT License. The preprocessed NetFlow-v2 datasets remain under their original CC-BY-NC-SA-4.0 license as distributed via Kaggle by Dhoogla.
