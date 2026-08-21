# FedKAN-IDS v2 — artifact for *What Federated KANs Cost and What They Actually Buy*

Artifact for the IEEE IoT-J revision of manuscript **IoT-66559-2026**. It contains the simulator,
every configuration, every per-run metric, the scripts that generate each table and figure of the
paper, and the LaTeX source of both the submitted version and this revision.

> **Read this first.** The revision reverses several claims of the version submitted in May 2026.
> If you arrived here from the May submission, the numbers below are not the ones in that paper,
> and the difference is the point of the revision. Both source trees are in this repository so the
> two can be diffed directly.

## What the paper now reports

Under a nested protocol, where each architecture's learning rate is chosen on ten seeds and the
comparison is reported on twenty seeds the selection never saw:

| Cell | FedKAN-8 minus MLP-PM-80 | p (paired) |
|---|---|---|
| NF-BoT-IoT-v2 binary | **-0.54 pp** | 0.581 |
| NF-ToN-IoT-v2 binary | **+4.89 pp** | 0.066 |
| NF-CSE-CIC binary | +2.52 pp | 0.271 |
| NF-CSE-CIC multi-class | **+1.55 pp** | 0.014 |

The advantage that the May submission reported (+5.49 pp on NF-BoT-IoT-v2) is absent on that cell
once the baseline is tuned, and smaller but present on the other three. Three claims of the May
version are withdrawn outright:

- **Variance reduction.** 2.53x becomes 1.05x once both architectures are tuned, and is below one
  on three cells of four.
- **Worst-seed robustness.** +21.45 pp becomes -0.09 pp under per-architecture tuning.
- **Learning-rate robustness.** Over a grid extended to eta = 1, the spread ratio is 0.33, 0.74, 0.86
  and 0.88: below one everywhere, so the parameter-matched MLP is the more robust of the two.

What remains is a cost measurement. At matched parameters on a single CPU thread, FedKAN costs
19.9x more per-sample inference, 37.9x more per batch of 1000, and transmits 15.4 kB per client
per round against 13.4 kB. The 17.2% uplink excess is the B-spline knot vector, carried in
`state_dict` as a buffer: 564 non-learnable scalars against 3,280 learnable ones.

## Hardware

**Every number in the revision was produced on one machine, an Apple M5, CPU only.** The May
submission ran on an NVIDIA RTX 4090; those 310 runs are retained under `results/runs/` for
reference and are **not** used for any number in the current paper. Table VIII of the manuscript
reconciles the run counts. Scripts that build headline tables print the `device` field of every
run they read and warn if more than one appears.

## Runs on disk

| group | runs |
|---|---|
| runs the reported numbers rest on | **1,860** |
| mis-configured cell, retained and labelled | 240 |
| original submission, different hardware, superseded | 310 |
| **total** | **2410** |

The mis-configured directory is kept deliberately, with a `DUNG-DOC-THU-MUC-NAY.md` note: it used
`downsample=130000` where the submitted cell used `50000`, and the discrepancy flips the sign of
that cell's result. It is a datum about silent config inheritance, not clutter.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py          # fetches the three NetFlow-v2 datasets
bash scripts/run_all_local.sh                     # learning-rate sweep, four cells
.venv/bin/python scripts/run_phase0.py            # held-out seeds, extended grid, SCAFFOLD/SGD
bash scripts/rebuild_paper_artifacts.sh           # every table and figure, then the PDF
```

No GPU is required and none is used. `rebuild_paper_artifacts.sh` regenerates each table and
figure from the committed run directories; `scripts/emit_paper_macros.py` writes the LaTeX macro
file that holds every headline number, so the manuscript never contains a hand-typed one.

## Layout

```
  src/                  simulator: KAN and MLP clients, FedAvg server, four extra aggregators
  scripts/              experiment runners, analysis, table and figure generators, gates
  configs/experiments/  one YAML per dataset; per-cell overrides are explicit, never inherited
  results/              per-run metrics, generated tables and figures
  paper-r1/             LaTeX source of this revision
  paper-as-submitted/   LaTeX source of the May 2026 submission, for diffing
```

## Verification scripts

The repository carries the checks used while preparing the revision, because several of them
caught real errors:

- `verify_handtyped_tables.py` recomputes every cell of the tables typed by hand into the
  manuscript and prints them beside the printed values.
- `check_macro_coverage.py` flags any literal in the LaTeX that duplicates a generated macro.
- `make_tables_r1.py` and `plot_all_cpu.py` refuse to proceed if their inputs mix devices.

## Citation

```bibtex
@article{do2026fedkanids,
  title={What Federated {KANs} Cost and What They Actually Buy: A Controlled Study for
         {NetFlow} Intrusion Detection at Gateway Scale},
  author={Do, Phuc Hao and Nguyen, Van Long and Le, Tran Duc and Dinh, Truong Duy},
  journal={under revision, IEEE Internet of Things Journal},
  year={2026}
}
```
