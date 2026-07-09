"""
Final, paper-ready consolidation. Two-model framing:
  - "KSERESNET" (baseline)  = original ICST-2026 submission
  - "KSERESNET-V2" (proposed) = final model (8-ch features + offline
    pairwise ranking-loss training on the 80% train split), evaluated
    ONLY on the untouched 20% held-out split for a leakage-free comparison.
All other tools (RoadFury, EagleSemble, RTE4SDC, ITEP4SDC, Random) are
evaluated on that same held-out split/protocol for a single consistent
main table. Intermediate design variants (pointwise features-only,
online ranking fine-tune, generator-stratified pairs, Transformer backbone,
longer training) are NOT shown as competing models -- they're summarized
in a compact "design decisions" table for the Discussion section.
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

OUT = "../../../paper_results"
os.makedirs(OUT, exist_ok=True)
md_lines = ["# KSERESNET-V2: Consolidated Results\n"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, tool, metric, success_only=True):
    return [float(r[metric]) for r in rows if r["tool"] == tool and (not success_only or r["success"] == "True")]


def a12(x, y):
    n = min(len(x), len(y))
    wins = sum(1 for a, b in zip(x, y) if b > a)
    ties = sum(1 for a, b in zip(x, y) if b == a)
    return (wins + 0.5 * ties) / n


def summarize(rows, tool):
    a = series(rows, tool, "apfd")
    c = series(rows, tool, "apfdc")
    t = series(rows, tool, "t_prioritize_tests")
    n_total = sum(1 for r in rows if r["tool"] == tool)
    n_ok = len(a)
    return {
        "n": n_total, "success_rate": 100 * n_ok / n_total if n_total else 0,
        "apfd_mean": statistics.mean(a) if a else float("nan"),
        "apfd_std": statistics.pstdev(a) if len(a) > 1 else 0.0,
        "apfdc_mean": statistics.mean(c) if c else float("nan"),
        "t_mean": statistics.mean(t) if t else float("nan"),
    }


def write_table(name, header, rows_dicts, cols):
    path_csv = os.path.join(OUT, f"{name}.csv")
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_dicts)
    md_lines.append(f"\n## {header}\n")
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("|" + "---|" * len(cols))
    for r in rows_dicts:
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"wrote {path_csv}")


# ============================================================
# MAIN TABLE: baseline vs proposed vs all 2026 competitors,
# ONE protocol (held-out split, seed=7, sizes 10-200, repeats=20)
# ============================================================
r_base = load("results_v1_heldout_wide.csv")          # KSERESNET baseline
r_main = load("results_wide_sweep.csv")               # KSERESNET-V4(->V2 proposed), RoadFury, RTE4SDC, KSERESNET-V2(old, unused here)
r_extra = load("results_wide_sweep_extra.csv")         # EagleSemble, ITEP4SDC, Random

entries = [
    ("KSERESNET (baseline)", r_base, "KSERESNET"),
    ("KSERESNET-V2 (proposed)", r_main, "KSERESNET-V4"),
    ("RoadFury", r_main, "RoadFury"),
    ("RTE4SDC", r_main, "RTE4SDC"),
    ("EagleSemble", r_extra, "EagleSemble"),
    ("ITEP4SDC", r_extra, "ITEP4SDC"),
    ("Random", r_extra, "Random"),
]
main_table = []
for display, rows, key in entries:
    s = summarize(rows, key)
    main_table.append({
        "tool": display, "apfd": f"{s['apfd_mean']:.4f}", "apfd_std": f"{s['apfd_std']:.4f}",
        "apfdc": f"{s['apfdc_mean']:.4f}", "t_prioritize_s": f"{s['t_mean']:.3f}",
        "success_rate_%": f"{s['success_rate']:.1f}",
    })
main_table.sort(key=lambda r: -float(r["apfd"]))
write_table("MAIN_table1_results", "Table 1: Main results (held-out split, N=7,202, seed=7, sizes 10-200, repeats=20)",
            main_table, ["tool", "apfd", "apfd_std", "apfdc", "t_prioritize_s", "success_rate_%"])

# stats: proposed vs baseline, proposed vs each competitor
base_apfd = series(r_base, "KSERESNET", "apfd")
prop_apfd = series(r_main, "KSERESNET-V4", "apfd")
rf_apfd = series(r_main, "RoadFury", "apfd")
rte_apfd = series(r_main, "RTE4SDC", "apfd")

stats_rows = []


def add_stat(label, x, y):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = [b - a for a, b in zip(x, y)]
    p = wilcoxon(d).pvalue if any(d) else 1.0
    stats_rows.append({"comparison": label, "mean_diff": f"{statistics.mean(d):+.4f}",
                        "A12": f"{a12(x, y):.3f}", "wilcoxon_p": f"{p:.2e}"})


add_stat("Proposed vs Baseline (main contribution)", base_apfd, prop_apfd)
add_stat("RoadFury vs Proposed (remaining SOTA gap)", prop_apfd, rf_apfd)
add_stat("Proposed vs RTE4SDC (beats 2026 competitor)", rte_apfd, prop_apfd)
write_table("MAIN_table2_significance", "Table 2: Statistical significance (Wilcoxon signed-rank + Vargha-Delaney A12)",
            stats_rows, ["comparison", "mean_diff", "A12", "wilcoxon_p"])

# ============================================================
# CROSS-EDITION table (2025 selectors), full pool -- unchanged, separate RQ
# ============================================================
r4 = load("results_2025_selectors_real.csv")
sel_tools = ["CertiFail", "Detour", "DRVN", "ITS4SDC", "CurvatureSelector", "GraphSelector", "MLSelector", "TransformerSelector"]
table_ce = []
for tool in sel_tools:
    s = summarize(r4, tool)
    table_ce.append({"tool": tool, "apfd": f"{s['apfd_mean']:.3f}", "apfdc": f"{s['apfdc_mean']:.3f}",
                      "success_rate_%": f"{s['success_rate']:.1f}"})
table_ce.sort(key=lambda r: -float(r["apfd"]))
write_table("MAIN_table3_cross_edition", "Table 3: 2025 selection tools converted to prioritization (full pool, seed=42)",
            table_ce, ["tool", "apfd", "apfdc", "success_rate_%"])

# ============================================================
# ABLATION table -- descriptive cells, no confusing version numbers
# ============================================================
r1 = load("results_v1_real.csv")            # baseline features, full pool (used only for ablation, not leakage-sensitive since baseline wasn't retrained by us)
r2 = load("results_v2_real.csv")             # baseline+features+pointwise fine-tune (intermediate step)
r5 = load("results_ablation_real.csv")       # feature-only and fine-tune-only cells

abl_cells = {
    "Baseline features, no adaptation": series(r1, "KSERESNET", "apfd"),
    "Baseline features + fine-tuned": series(r5, "KSERESNET-V1-FT", "apfd"),
    "Proposed features, no adaptation": series(r5, "KSERESNET-V2-NOFT", "apfd"),
    "Proposed features + pointwise fine-tune": series(r2, "KSERESNET-V2", "apfd"),
}
table_abl = [{"configuration": k, "apfd_mean": f"{statistics.mean(v):.4f}", "apfd_std": f"{statistics.pstdev(v):.4f}"}
             for k, v in abl_cells.items()]
write_table("MAIN_table4_ablation", "Table 4: Ablation -- feature set x adaptation strategy (pointwise; full pool, seed=42)",
            table_abl, ["configuration", "apfd_mean", "apfd_std"])

# ============================================================
# DESIGN DECISIONS table -- compact, narrative, not a leaderboard
# ============================================================
design_rows = [
    {"design_choice": "Online per-subject ranking-loss fine-tuning (vs. pointwise)",
     "result": "No effect (+0.001 APFD, p=0.98)", "decision": "Not adopted -- online adaptation is data-starved (one small subject)"},
    {"design_choice": "More training epochs (40 vs 15) for offline ranking loss",
     "result": "Slightly worse (-0.0016 APFD, p=0.0003)", "decision": "Not adopted -- mild overfitting; 15 epochs is the practical optimum"},
    {"design_choice": "Generator-stratified fail/pass pair sampling",
     "result": "No effect (-0.0004 APFD, p=0.11)", "decision": "Not adopted -- did not fix the per-generator inconsistency"},
    {"design_choice": "Transformer encoder backbone (same features/recipe)",
     "result": "Substantially worse (-0.052 APFD, p<1e-25)", "decision": "Not adopted -- underperforms ResNet under our data/compute budget"},
    {"design_choice": "Listwise ranking loss (ListNet-style, same recipe)",
     "result": "Substantially worse (-0.0214 APFD, p=9.8e-23)", "decision": "Not adopted -- underperforms pairwise loss under our data/compute budget"},
]
write_table("MAIN_table5_design_decisions", "Table 5: Alternative design choices explored (not adopted)",
            design_rows, ["design_choice", "result", "decision"])

# ============================================================
# PER-GENERATOR breakdown -- baseline / proposed / RoadFury only
# ============================================================
gens = ["ambiegen", "frenetic", "freneticV"]
table_gen = []
gen_stats = []
for gen in gens:
    rg = load(f"results_gen_{gen}.csv")
    v2g = series(rg, "KSERESNET-V2", "apfd")  # this is actually old-V2; note below
    v4g = series(rg, "KSERESNET-V4", "apfd")  # proposed
    rfg = series(rg, "RoadFury", "apfd")
    for label, vals in [("RoadFury", rfg), ("KSERESNET-V2 (proposed)", v4g)]:
        table_gen.append({"generator": gen, "tool": label, "apfd_mean": f"{statistics.mean(vals):.4f}",
                           "apfd_std": f"{statistics.pstdev(vals):.4f}"})
    d = [b - a for a, b in zip(v2g, v4g)]
    gen_stats.append({"generator": gen, "comparison": "Proposed vs pointwise-features-only",
                       "mean_diff": f"{statistics.mean(d):+.4f}", "A12": f"{a12(v2g, v4g):.3f}",
                       "wilcoxon_p": f"{wilcoxon(d).pvalue:.2e}"})
write_table("MAIN_table6_per_generator", "Table 6: Per-generator breakdown", table_gen,
            ["generator", "tool", "apfd_mean", "apfd_std"])
write_table("MAIN_table6b_per_generator_stats", "Table 6b: Per-generator significance",
            gen_stats, ["generator", "comparison", "mean_diff", "A12", "wilcoxon_p"])

with open(os.path.join(OUT, "ALL_TABLES_FINAL.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print("\nwrote ALL_TABLES_FINAL.md")

# ============================================================
# FIGURES (renamed/cleaned)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
names = [r["tool"] for r in main_table]
vals = [float(r["apfd"]) for r in main_table]
colors = ["#2b6cb0" if "KSERESNET" in n else "#718096" for n in names]
ax.barh(names, vals, color=colors)
ax.set_xlabel("Mean APFD")
ax.set_title("Main results: baseline vs. proposed vs. 2026 competitors (held-out split)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "FIG1_main_results.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4))
stages = ["KSERESNET\n(baseline)", "KSERESNET-V2\n(proposed)", "RoadFury\n(SOTA)"]
stage_vals = [statistics.mean(base_apfd), statistics.mean(prop_apfd), statistics.mean(rf_apfd)]
ax.plot(stages, stage_vals, marker="o", linewidth=2, color="#2b6cb0")
for i, v in enumerate(stage_vals):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_ylabel("Mean APFD")
ax.set_title("Baseline -> proposed -> state of the art")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "FIG2_trajectory.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(gens))
width = 0.3
for i, (label, key, color) in enumerate([("RoadFury", "RoadFury", "#718096"), ("KSERESNET-V2 (proposed)", "KSERESNET-V4", "#2b6cb0")]):
    means = []
    for gen in gens:
        rg = load(f"results_gen_{gen}.csv")
        means.append(statistics.mean(series(rg, key, "apfd")))
    ax.bar(x + i * width, means, width, label=label, color=color)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(gens)
ax.set_ylabel("Mean APFD")
ax.set_title("Per-generator APFD")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "FIG3_per_generator.png"), dpi=150)
plt.close()

print("wrote 3 figures to", OUT, "(FIG4 channel importance unchanged, see fig4_channel_importance.png)")
