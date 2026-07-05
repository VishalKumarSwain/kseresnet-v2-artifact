"""
Consolidates every experiment run in this study into paper-ready tables
(CSV + Markdown) and figures, with Wilcoxon signed-rank p-values and
Vargha-Delaney A12 effect sizes for every key comparison.
Output: ../../../paper_results/
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

OUT = "../../../paper_results"
os.makedirs(OUT, exist_ok=True)
md_lines = []


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, tool, metric, success_only=True):
    return [float(r[metric]) for r in rows if r["tool"] == tool and (not success_only or r["success"] == "True")]


def a12(x, y):
    """Vargha-Delaney A12 for paired samples of equal length: P(y>x) + 0.5*P(y==x)."""
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
        "tool": tool, "n": n_total, "success_rate": 100 * n_ok / n_total if n_total else 0,
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
# TABLE 1: main comparison, full real pool, seed=42
# ============================================================
r1 = load("results_v1_real.csv")
r2 = load("results_v2_real.csv")
r3 = load("results_2026_competitors_real.csv")
main_tools = [("KSERESNET", r1, "2026 (ours)"), ("KSERESNET-V2", r2, "2026 (ours)")] + \
    [(t, r3, "2026") for t in ["RoadFury", "EagleSemble", "RTE4SDC", "ITEP4SDC", "Random"]]

table1 = []
for tool, rows, edition in main_tools:
    s = summarize(rows, tool)
    display_name = "KSERESNET-V1" if tool == "KSERESNET" else tool
    table1.append({
        "tool": display_name, "edition": edition,
        "apfd": f"{s['apfd_mean']:.3f}", "apfdc": f"{s['apfdc_mean']:.3f}",
        "t_prioritize_s": f"{s['t_mean']:.3f}", "success_rate_%": f"{s['success_rate']:.1f}",
    })
table1.sort(key=lambda r: -float(r["apfd"]))
write_table("table1_main_comparison", "Table 1: Main tool comparison (full real SensoDat, N=32,580, seed=42)",
            table1, ["tool", "edition", "apfd", "apfdc", "t_prioritize_s", "success_rate_%"])

# ============================================================
# TABLE 2: cross-edition (2025 selectors), same pool/protocol
# ============================================================
r4 = load("results_2025_selectors_real.csv")
sel_tools = ["CertiFail", "Detour", "DRVN", "ITS4SDC", "CurvatureSelector", "GraphSelector", "MLSelector", "TransformerSelector"]
table2 = []
for tool in sel_tools:
    s = summarize(r4, tool)
    table2.append({
        "tool": tool, "edition": "2025 (selection, converted)",
        "apfd": f"{s['apfd_mean']:.3f}", "apfdc": f"{s['apfdc_mean']:.3f}",
        "success_rate_%": f"{s['success_rate']:.1f}",
    })
table2.sort(key=lambda r: -float(r["apfd"]))
write_table("table2_cross_edition", "Table 2: 2025 selection tools converted to prioritization (same pool/protocol)",
            table2, ["tool", "edition", "apfd", "apfdc", "success_rate_%"])

# ============================================================
# TABLE 3: ablation (features x fine-tuning), full real pool
# ============================================================
r5 = load("results_ablation_real.csv")
abl_cells = {
    "V1 features, no fine-tune (V1)": series(r1, "KSERESNET", "apfd"),
    "V1 features, + fine-tune (V1-FT)": series(r5, "KSERESNET-V1-FT", "apfd"),
    "V2 features, no fine-tune (V2-NOFT)": series(r5, "KSERESNET-V2-NOFT", "apfd"),
    "V2 features, + fine-tune (V2)": series(r2, "KSERESNET-V2", "apfd"),
}
table3 = [{"cell": k, "apfd_mean": f"{statistics.mean(v):.4f}", "apfd_std": f"{statistics.pstdev(v):.4f}", "n": len(v)}
          for k, v in abl_cells.items()]
write_table("table3_ablation", "Table 3: Ablation — feature set x online fine-tuning (2x2)",
            table3, ["cell", "apfd_mean", "apfd_std", "n"])

# stats for ablation
v1 = abl_cells["V1 features, no fine-tune (V1)"]
v1ft = abl_cells["V1 features, + fine-tune (V1-FT)"]
v2noft = abl_cells["V2 features, no fine-tune (V2-NOFT)"]
v2 = abl_cells["V2 features, + fine-tune (V2)"]
stats_rows = []


def add_stat(label, x, y):
    d = [b - a for a, b in zip(x, y)]
    p = wilcoxon(d).pvalue if any(d) else 1.0
    stats_rows.append({
        "comparison": label, "mean_diff": f"{statistics.mean(d):+.4f}",
        "A12": f"{a12(x, y):.3f}", "wilcoxon_p": f"{p:.2e}",
    })


add_stat("V1-FT vs V1 (fine-tune effect, weak features)", v1, v1ft)
add_stat("V2 vs V2-NOFT (fine-tune effect, rich features)", v2noft, v2)
add_stat("V2-NOFT vs V1 (feature effect, no fine-tune)", v1, v2noft)
add_stat("V2 vs V1-FT (feature effect, with fine-tune)", v1ft, v2)

# ============================================================
# TABLE 4: final model validation, held-out (leakage-free), wide sweep seed=7
# ============================================================
r6 = load("results_wide_sweep.csv")  # V2, V4, RoadFury, RTE4SDC
r7 = load("results_v5_heldout.csv")  # V5
r8 = load("results_v6_heldout.csv")  # V6
table4_tools = [("RoadFury", r6), ("KSERESNET-V4", r6), ("KSERESNET-V5", r7), ("KSERESNET-V2", r6),
                 ("RTE4SDC", r6), ("KSERESNET-V6", r8)]
table4 = []
for tool, rows in table4_tools:
    s = summarize(rows, tool)
    table4.append({"tool": tool, "apfd": f"{s['apfd_mean']:.4f}", "apfd_std": f"{s['apfd_std']:.4f}",
                    "apfdc": f"{s['apfdc_mean']:.4f}", "success_rate_%": f"{s['success_rate']:.1f}"})
table4.sort(key=lambda r: -float(r["apfd"]))
write_table("table4_final_validation", "Table 4: Final model validation, held-out split (N=7,202, seed=7, sizes 10-200)",
            table4, ["tool", "apfd", "apfd_std", "apfdc", "success_rate_%"])

v2w = series(r6, "KSERESNET-V2", "apfd"); v4w = series(r6, "KSERESNET-V4", "apfd")
rfw = series(r6, "RoadFury", "apfd"); rtew = series(r6, "RTE4SDC", "apfd")
v5w = series(r7, "KSERESNET-V5", "apfd"); v6w = series(r8, "KSERESNET-V6", "apfd")
add_stat("V4 vs V2 (offline ranking-loss effect)", v2w, v4w)
add_stat("RoadFury vs V4 (remaining gap to SOTA)", v4w, rfw)
add_stat("V4 vs RTE4SDC (beats 2026 competitor)", rtew, v4w)
add_stat("V5 vs V4 (generator-stratified pairs, expect no effect)", v4w, v5w)
add_stat("V4 vs V6 (ResNet vs Transformer, same recipe)", v6w, v4w)

write_table("table5_statistical_tests", "Table 5: Statistical significance (Wilcoxon signed-rank + Vargha-Delaney A12)",
            stats_rows, ["comparison", "mean_diff", "A12", "wilcoxon_p"])

# ============================================================
# TABLE 6: per-generator breakdown
# ============================================================
gens = ["ambiegen", "frenetic", "freneticV"]
table6 = []
gen_stats = []
for gen in gens:
    rg = load(f"results_gen_{gen}.csv")
    rg5 = load(f"results_v5_gen_{gen}.csv")
    v2g = series(rg, "KSERESNET-V2", "apfd"); v4g = series(rg, "KSERESNET-V4", "apfd")
    rfg = series(rg, "RoadFury", "apfd"); v5g = series(rg5, "KSERESNET-V5", "apfd")
    for tool, vals in [("RoadFury", rfg), ("KSERESNET-V2", v2g), ("KSERESNET-V4", v4g), ("KSERESNET-V5", v5g)]:
        table6.append({"generator": gen, "tool": tool, "apfd_mean": f"{statistics.mean(vals):.4f}",
                        "apfd_std": f"{statistics.pstdev(vals):.4f}"})
    d = [b - a for a, b in zip(v2g, v4g)]
    gen_stats.append({"generator": gen, "comparison": "V4 vs V2", "mean_diff": f"{statistics.mean(d):+.4f}",
                       "A12": f"{a12(v2g, v4g):.3f}", "wilcoxon_p": f"{wilcoxon(d).pvalue:.2e}"})
write_table("table6_per_generator", "Table 6: Per-generator breakdown (Ambiegen / Frenetic / FreneticV)",
            table6, ["generator", "tool", "apfd_mean", "apfd_std"])
write_table("table6b_per_generator_stats", "Table 6b: Per-generator significance (V4 vs V2)",
            gen_stats, ["generator", "comparison", "mean_diff", "A12", "wilcoxon_p"])

# ============================================================
# TABLE 7: training-scale sensitivity (V4 vs V4B)
# ============================================================
r9 = load("results_heldout_comparison.csv")  # V4, seed=42
r10 = load("results_v4b_heldout.csv")  # V4B, seed=42
v4e = series(r9, "KSERESNET-V4", "apfd"); v4be = series(r10, "KSERESNET-V4B", "apfd")
add_stat("V4B (40 epochs) vs V4 (15 epochs) -- overfitting check", v4e, v4be)
write_table("table7_training_scale", "Table 7: Training-epoch sensitivity",
            [stats_rows[-1]], ["comparison", "mean_diff", "A12", "wilcoxon_p"])

with open(os.path.join(OUT, "ALL_TABLES.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print("\nwrote ALL_TABLES.md")

# ============================================================
# FIGURES
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
combined = [(r["tool"], float(r["apfd"]), "2025") for r in table2] + [(r["tool"], float(r["apfd"]), "2026") for r in table1]
combined.sort(key=lambda t: -t[1])
names = [c[0] for c in combined]
vals = [c[1] for c in combined]
colors = ["#2b6cb0" if "KSERESNET" in n else ("#c05621" if e == "2025" else "#718096") for n, v, e in combined]
ax.barh(names, vals, color=colors)
ax.set_xlabel("Mean APFD")
ax.set_title("APFD across all 15 tools (2025 selection + 2026 prioritization), real SensoDat")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_all_tools_apfd.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
stages = ["V1\n(baseline)", "V2\n(+features)", "V4\n(+ranking loss)", "RoadFury\n(SOTA)"]
stage_vals = [statistics.mean(series(r1, "KSERESNET", "apfd")), statistics.mean(v2w), statistics.mean(v4w), statistics.mean(rfw)]
ax.plot(stages, stage_vals, marker="o", linewidth=2, color="#2b6cb0")
for i, v in enumerate(stage_vals):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_ylabel("Mean APFD")
ax.set_title("Improvement trajectory: diagnosis -> remedy")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_improvement_trajectory.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
import numpy as np
x = np.arange(len(gens))
width = 0.25
for i, (tool, color) in enumerate([("RoadFury", "#718096"), ("KSERESNET-V2", "#63b3ed"), ("KSERESNET-V4", "#2b6cb0")]):
    means = []
    for gen in gens:
        rg = load(f"results_gen_{gen}.csv")
        means.append(statistics.mean(series(rg, tool, "apfd")))
    ax.bar(x + i * width, means, width, label=tool, color=color)
ax.set_xticks(x + width)
ax.set_xticklabels(gens)
ax.set_ylabel("Mean APFD")
ax.set_title("Per-generator APFD: aggregate improvement masks generator-level trade-offs")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_per_generator.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
channels = ["angle_change", "curvature", "seg_length", "local_curv_std", "ay", "dx", "ax", "dy"]
drops = [0.3404, 0.2715, 0.0827, 0.0362, 0.0053, 0.0048, 0.0025, 0.0016]
ax.barh(channels, drops, color=["#c05621" if d > 0.05 else "#a0aec0" for d in drops])
ax.set_xlabel("AUC drop when channel is zeroed (importance)")
ax.set_title("KSERESNET-V4 channel importance (fail/pass separation)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_channel_importance.png"), dpi=150)
plt.close()

print("wrote 4 figures to", OUT)
