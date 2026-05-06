"""Pairwise statistical tests between FedKAN (Ours) and the strongest baseline.

For each (mode, partition) cell where both variants have >= MIN_SEEDS seeds:
  - Pair runs by seed (same seed = same Dirichlet partition + init)
  - Welch's t-test (independent) + paired t-test (when paired data available)
  - Bootstrap 95% CI for the mean accuracy difference
  - Cohen's d effect size

Output:
  results/aggregated/stats_tests.txt    (human-readable)
  results/tables/stats_tests.tex        (paper-ready table)
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT_TXT = ROOT / "results" / "aggregated" / "stats_tests.txt"
OUT_TEX = ROOT / "results" / "tables" / "stats_tests.tex"

A = "kan_h8"            # FedKAN-IDS (ours)
B = "mlp_h80"           # parameter-matched MLP baseline (the toughest comparison)
MIN_SEEDS = 5
N_BOOT = 5000
RNG = np.random.default_rng(0)


def variant_of(m: dict) -> str | None:
    name = m.get("model_name")
    h = m.get("model_hidden")
    if not name or not h:
        return None
    return f"{name}_h{'x'.join(str(x) for x in h)}"


def ptag_of(m: dict) -> str:
    p = m.get("data_partition", "dirichlet")
    if p == "iid":
        return "iid"
    return f"dir{m.get('data_alpha')}"


def collect() -> dict:
    """{(mode, ptag, variant): {seed: {accuracy, f1_macro, per_class_f1}}}"""
    out: dict = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text())
        v = variant_of(m)
        if v is None:
            continue
        key = (m.get("data_mode", "binary"), ptag_of(m), v)
        seed = m.get("seed")
        fm = m.get("final_metrics", {})
        out.setdefault(key, {})[seed] = {
            "accuracy": fm.get("accuracy"),
            "f1_macro": fm.get("f1_macro"),
            "per_class_f1": fm.get("per_class_f1"),
        }
    return out


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sx2 = x.var(ddof=1) if nx > 1 else 0.0
    sy2 = y.var(ddof=1) if ny > 1 else 0.0
    pooled = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / max(nx + ny - 2, 1))
    if pooled < 1e-12:
        return float("inf") if x.mean() != y.mean() else 0.0
    return (x.mean() - y.mean()) / pooled


def bootstrap_ci_diff(diffs: np.ndarray, alpha: float = 0.05,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(diffs)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    return (float(np.quantile(boot_means, alpha / 2)),
            float(np.quantile(boot_means, 1 - alpha / 2)))


def cell_report(mode: str, ptag: str, runs: dict) -> dict | None:
    a_data = runs.get((mode, ptag, A), {})
    b_data = runs.get((mode, ptag, B), {})
    common_seeds = sorted(set(a_data.keys()) & set(b_data.keys()))
    a_only = sorted(set(a_data.keys()) - set(b_data.keys()))
    b_only = sorted(set(b_data.keys()) - set(a_data.keys()))

    if min(len(a_data), len(b_data)) < MIN_SEEDS:
        return None

    a_acc = np.array([a_data[s]["accuracy"] for s in sorted(a_data)])
    b_acc = np.array([b_data[s]["accuracy"] for s in sorted(b_data)])

    # Independent Welch's t-test (uses all available seeds per side).
    t_w, p_w = sstats.ttest_ind(a_acc, b_acc, equal_var=False)

    rep = {
        "mode": mode, "ptag": ptag,
        "n_a": len(a_acc), "n_b": len(b_acc),
        "mean_a": float(a_acc.mean()), "std_a": float(a_acc.std(ddof=1)),
        "mean_b": float(b_acc.mean()), "std_b": float(b_acc.std(ddof=1)),
        "min_a": float(a_acc.min()), "min_b": float(b_acc.min()),
        "welch_t": float(t_w), "welch_p": float(p_w),
        "cohen_d": cohens_d(a_acc, b_acc),
        "a_only_seeds": a_only, "b_only_seeds": b_only,
        "paired": False,
    }

    if len(common_seeds) >= MIN_SEEDS:
        a_p = np.array([a_data[s]["accuracy"] for s in common_seeds])
        b_p = np.array([b_data[s]["accuracy"] for s in common_seeds])
        diffs = a_p - b_p
        t_pair, p_pair = sstats.ttest_rel(a_p, b_p)
        ci = bootstrap_ci_diff(diffs)
        rep.update(
            paired=True,
            n_paired=len(common_seeds),
            paired_seeds=common_seeds,
            paired_t=float(t_pair), paired_p=float(p_pair),
            mean_diff=float(diffs.mean()),
            ci_lo=ci[0], ci_hi=ci[1],
        )
    return rep


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<10^{-4}"
    if p < 1e-3:
        return "<10^{-3}"
    if p < 1e-2:
        return "<10^{-2}"
    return f"{p:.3f}"


def main():
    runs = collect()
    cells = sorted({(m, p) for (m, p, _v) in runs})
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    txt_lines = [f"Pairwise stats: {A} (Ours) vs {B} (toughest baseline)",
                 f"  MIN_SEEDS={MIN_SEEDS}, N_BOOT={N_BOOT}", ""]
    tex_rows: list[str] = []

    skipped = []
    for mode, ptag in cells:
        rep = cell_report(mode, ptag, runs)
        if rep is None:
            skipped.append((mode, ptag))
            continue
        head = f"=== {mode}/{ptag}  (n_A={rep['n_a']}, n_B={rep['n_b']})"
        head += f"  paired={rep['paired']}"
        txt_lines.append(head)
        txt_lines.append(
            f"  KAN-8 (Ours): mean={rep['mean_a']*100:.2f}%  "
            f"std={rep['std_a']*100:.2f}  worst={rep['min_a']*100:.2f}%"
        )
        txt_lines.append(
            f"  MLP-PM (B):  mean={rep['mean_b']*100:.2f}%  "
            f"std={rep['std_b']*100:.2f}  worst={rep['min_b']*100:.2f}%"
        )
        txt_lines.append(
            f"  Welch t = {rep['welch_t']:+.3f}, p = {fmt_p(rep['welch_p'])}, "
            f"Cohen d = {rep['cohen_d']:+.3f}"
        )
        if rep["paired"]:
            txt_lines.append(
                f"  Paired t = {rep['paired_t']:+.3f}, p = {fmt_p(rep['paired_p'])} "
                f"(n_paired={rep['n_paired']})"
            )
            txt_lines.append(
                f"  Mean diff (A-B) = {rep['mean_diff']*100:+.2f}pp, "
                f"95% CI = [{rep['ci_lo']*100:+.2f}, {rep['ci_hi']*100:+.2f}]"
            )
        txt_lines.append("")

        # LaTeX row.
        sig = ""
        p = rep.get("paired_p", rep["welch_p"])
        if p < 1e-3: sig = r"$^{***}$"
        elif p < 1e-2: sig = r"$^{**}$"
        elif p < 0.05: sig = r"$^{*}$"
        ci_str = "--"
        if rep["paired"]:
            ci_str = (f"[{rep['ci_lo']*100:+.2f}, "
                      f"{rep['ci_hi']*100:+.2f}]")
        tex_rows.append(
            " & ".join([
                mode.capitalize(),
                ptag if ptag == "iid" else fr"Dir($\alpha{{=}}{ptag[3:]}$)",
                f"{rep['n_paired']}" if rep["paired"] else f"{rep['n_a']}/{rep['n_b']}",
                f"{rep['mean_a']*100:.2f}\\,$\\pm$\\,{rep['std_a']*100:.2f}",
                f"{rep['mean_b']*100:.2f}\\,$\\pm$\\,{rep['std_b']*100:.2f}",
                f"{rep['mean_diff']*100:+.2f}" if rep["paired"] else
                    f"{(rep['mean_a']-rep['mean_b'])*100:+.2f}",
                ci_str,
                fr"${rep['cohen_d']:+.2f}$",
                f"${fmt_p(p)}${sig}",
            ]) + r" \\"
        )

    if skipped:
        txt_lines.append("Skipped (insufficient seeds):")
        for m, p in skipped:
            txt_lines.append(f"  {m}/{p}")

    OUT_TXT.write_text("\n".join(txt_lines), encoding="utf-8")
    print("\n".join(txt_lines))
    print(f"\nWrote {OUT_TXT}")

    # LaTeX table.
    tex = [
        r"% Auto-generated by scripts/stats_tests.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical comparison: \textbf{FedKAN-IDS (Ours)} vs FedAvg-MLP-PM (parameter-matched). Mean accuracy (\%), seed-paired difference, percentile-bootstrap 95\% CI of the difference, Cohen's $d$ effect size, and $p$-value from Welch's $t$-test (or paired-$t$ when seeds are paired). $^{*}p{<}0.05$, $^{**}p{<}10^{-2}$, $^{***}p{<}10^{-3}$.}",
        r"\label{tab:stats_botiot}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l l c c c c c c c}",
        r"\toprule",
        r"Mode & Partition & $n$ & KAN-8 & MLP-PM-80 & $\Delta$ (pp) & 95\% CI & $d$ & $p$ \\",
        r"\midrule",
    ] + tex_rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
