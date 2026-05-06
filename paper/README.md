# Paper sources

LaTeX skeleton for the FedKAN-IDS journal submission, targeting **IEEE Internet of Things Journal**.

## Layout
```
paper/
├── main.tex                  # entry point — IEEEtran journal
├── references.bib            # bibliography
└── sections/
    ├── 01_intro.tex          # introduction (skeleton + contributions)
    ├── 02_preliminaries.tex  # Definitions 1, 2 (KAN layer + Federated IDS)
    ├── 03_method.tex         # methodology (skeleton, pending)
    ├── 04_theory.tex         # Propositions 1, 2 + Lemma 1 + Theorem 1 + Corollary
    ├── 05_experiments.tex    # experimental section (hooks for tables/figures)
    └── 06_conclusion.tex     # discussion + conclusion
```

## Build
Requires `IEEEtran.cls` (in path or in this directory). On Overleaf the class is preinstalled.
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Figures and tables
The LaTeX sources reference `../results/figures/*.pdf` and `\input` `../results/tables/*.tex` directly,
so the paper updates automatically whenever `scripts/aggregate.py`, `scripts/plot_*.py`, or
`scripts/make_latex_tables.py` are re-run from a fresh experiment batch.

## Status

| Section | Status |
|---|---|
| I.   Introduction | **complete** — full motivation, KAN-FL related work, contributions |
| II.  Preliminaries | **complete** — Defs 1, 2 |
| III. Method | **complete** — TikZ flow figure, Algorithms 1 + 2, cost discussion |
| IV.  Theory | **complete** — Props 1, 2; Lemma 1; Theorem 1; Corollary |
| V.   Experiments | skeleton with hooks for headline / worst-seed / per-class tables; awaiting M3c data |
| VI.  Conclusion | skeleton + threats-to-validity bullets |
