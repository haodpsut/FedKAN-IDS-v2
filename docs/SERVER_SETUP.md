# RTX 4090 Server Setup (conda-only, no sudo)

This guide walks the user-level setup for running FedKAN-IDS experiments on a workstation with an RTX 4090 where only conda is available (no apt, no sudo).

## Step 1 — Create the conda environment

```bash
# 1. Create + activate
conda create -n fedkan python=3.11 -y
conda activate fedkan

# 2. Install PyTorch with CUDA. Pick ONE of these:
# 2a. If the system already has the NVIDIA driver and CUDA 12.x runtime:
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# 2b. If you'd rather have conda manage CUDA toolkit too:
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Other deps (everything except efficient_kan)
pip install numpy>=1.24 pandas>=2.0 pyarrow>=14.0 scipy>=1.11 \
    scikit-learn>=1.3 matplotlib>=3.7 seaborn>=0.12 \
    pyyaml>=6.0 tqdm>=4.65 thop kaggle

# 4. efficient_kan from GitHub (no PyPI release)
pip install "git+https://github.com/Blealtan/efficient-kan.git@master"

# 5. Verify CUDA + KAN imports
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
python -c "from efficient_kan import KAN; m = KAN([8,4,2], grid_size=5, spline_order=3); print('KAN OK, params:', sum(p.numel() for p in m.parameters()))"
```

Expected output of step 5:
```
torch 2.x.x cuda: True
device: NVIDIA GeForce RTX 4090
KAN OK, params: 800
```

## Step 2 — Clone the repo + configure git

```bash
git clone https://github.com/haodpsut/FedKAN-IDS-v2.git
cd FedKAN-IDS-v2

git config user.email "haodp@dau.edu.vn"
git config user.name  "Phuc Hao Do"

# Set the push URL with PAT — replace YOUR_PAT_HERE
git remote set-url origin "https://haodpsut:YOUR_PAT_HERE@github.com/haodpsut/FedKAN-IDS-v2.git"
```

To get a PAT: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate, scope `repo`. Same one as for Colab.

## Step 3 — Get Kaggle credentials onto the server

If the workstation doesn't have a browser, the simplest path is to upload `kaggle.json` over SSH from your laptop:

```bash
# On laptop
scp ~/.kaggle/kaggle.json user@server:~/.kaggle/kaggle.json
```

Then on the server:
```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

If you don't have ssh, you can also get `kaggle.json` by `cat`-ing the file content from Drive into a new server file — the file is plain JSON.

## Step 4 — Smoke test the pipeline

```bash
# Run the offline synthetic-data smoke test (no Kaggle needed)
python scripts/smoke_test.py
```

Expected: each model reaches >0.6 in 3 rounds and the script prints `OK`.

## Step 5 — Prepare datasets

```bash
# BoT-IoT first (smallest, ~70MB raw)
python scripts/prepare_data.py --dataset nf_botiot_v2

# Inspect what came out (sanity check)
python scripts/inspect_data.py --dataset nf_botiot_v2

# Then ToN-IoT
python scripts/prepare_data.py --dataset nf_toniot_v2
python scripts/inspect_data.py --dataset nf_toniot_v2

# CSE-CIC-IDS2018 (larger, ~250MB)
python scripts/prepare_data.py --dataset nf_cseciic_v2
python scripts/inspect_data.py --dataset nf_cseciic_v2
```

> **Critical**: pay attention to the `inspect_data.py` output for ToN-IoT. The Colab runs landed at acc=0.5 — likely a label-column mismatch in the parquet. If the binary distribution shows only one class, see [Step 7: Fix ToN-IoT labels](#step-7--fix-ton-iot-labels-if-degenerate) before running experiments.

## Step 6 — Run experiments via the bash grid runner

The notebook cells 4d/4c are reproduced in `scripts/run_grid.sh`. Recommended sequence:

```bash
# 6a. M3c minimal on ToN-IoT (24 runs ~ 1.5h on RTX 4090)
DATASET=toniot MINIMAL=1 bash scripts/run_grid.sh

# 6b. Once that confirms the pattern, run the full ToN-IoT grid (72 runs ~ 4h)
DATASET=toniot bash scripts/run_grid.sh

# 6c. Full BoT-IoT M2 grid is already done; if you want to regenerate just
#     for verification, run-experiment's --skip-existing makes this cheap:
DATASET=botiot bash scripts/run_grid.sh

# 6d. Add 7 more seeds on the killer cells for n=10 statistics
DATASET=toniot FILL_M3A=1 bash scripts/run_grid.sh

# 6e. CSE-CIC-IDS2018-v2 cross-dataset replication
DATASET=cseciic MINIMAL=1 bash scripts/run_grid.sh
```

Each run takes ~1.5--2.5 min on RTX 4090 (vs ~5--7 min on Colab T4). The script auto-commits and pushes every `PUSH_EVERY_N=4` runs (configurable via env var) and retries failed pushes 3x with backoff.

If a long run is interrupted, just rerun the same command — `--skip-existing` handles resumption transparently.

## Step 7 — Fix ToN-IoT labels (if degenerate)

If `inspect_data.py --dataset nf_toniot_v2` shows the binary distribution as a single class (or some other degenerate condition), open an issue or ask Claude to patch `src/datasets/netflow_v2.py`. Common fixes:
- `binary_label_column` is named `Label` in the YAML but the actual column is `Class` or `Attack_Type` in this Kaggle mirror.
- Labels are strings (`"Benign"/"Attack"`) rather than 0/1 — needs explicit mapping.

## Step 8 — Pull on the dev machine to analyse

After the server pushes, on the dev machine:

```bash
git pull --rebase
python scripts/aggregate.py
python scripts/stats_tests.py
python scripts/plot_convergence.py
python scripts/plot_perclass.py
python scripts/make_latex_tables.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

PDF appears at `paper/main.pdf`.
