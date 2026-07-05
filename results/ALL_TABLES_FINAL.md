# KSERESNET-V2: Consolidated Results


## Table 1: Main results (held-out split, N=7,202, seed=7, sizes 10-200, repeats=20)

| tool | apfd | apfd_std | apfdc | t_prioritize_s | success_rate_% |
|---|---|---|---|---|---|
| RoadFury | 0.8028 | 0.0405 | 0.8614 | 0.804 | 100.0 |
| KSERESNET-V2 (proposed) | 0.7938 | 0.0427 | 0.8545 | 0.370 | 100.0 |
| ITEP4SDC | 0.7882 | 0.0488 | 0.8571 | 0.184 | 31.2 |
| RTE4SDC | 0.7791 | 0.0416 | 0.8364 | 0.244 | 100.0 |
| EagleSemble | 0.7765 | 0.0487 | 0.8513 | 0.477 | 31.2 |
| KSERESNET (baseline) | 0.6246 | 0.0685 | 0.6679 | 0.180 | 100.0 |
| Random | 0.5134 | 0.0885 | 0.5488 | 0.079 | 100.0 |

## Table 2: Statistical significance (Wilcoxon signed-rank + Vargha-Delaney A12)

| comparison | mean_diff | A12 | wilcoxon_p |
|---|---|---|---|
| Proposed vs Baseline (main contribution) | +0.1691 | 0.997 | 7.64e-28 |
| RoadFury vs Proposed (remaining SOTA gap) | +0.0090 | 0.750 | 4.24e-13 |
| Proposed vs RTE4SDC (beats 2026 competitor) | +0.0146 | 0.800 | 5.89e-14 |

## Table 3: 2025 selection tools converted to prioritization (full pool, seed=42)

| tool | apfd | apfdc | success_rate_% |
|---|---|---|---|
| ITS4SDC | 0.707 | 0.760 | 100.0 |
| CurvatureSelector | 0.597 | 0.607 | 100.0 |
| GraphSelector | 0.587 | 0.619 | 100.0 |
| TransformerSelector | 0.560 | 0.581 | 100.0 |
| CertiFail | 0.549 | 0.572 | 100.0 |
| MLSelector | 0.546 | 0.591 | 100.0 |
| DRVN | 0.537 | 0.575 | 98.7 |
| Detour | 0.520 | 0.577 | 98.7 |

## Table 4: Ablation -- feature set x adaptation strategy (pointwise; full pool, seed=42)

| configuration | apfd_mean | apfd_std |
|---|---|---|
| Baseline features, no adaptation | 0.6161 | 0.0614 |
| Baseline features + fine-tuned | 0.6444 | 0.0741 |
| Proposed features, no adaptation | 0.7586 | 0.0538 |
| Proposed features + pointwise fine-tune | 0.7578 | 0.0547 |

## Table 5: Alternative design choices explored (not adopted)

| design_choice | result | decision |
|---|---|---|
| Online per-subject ranking-loss fine-tuning (vs. pointwise) | No effect (+0.001 APFD, p=0.98) | Not adopted -- online adaptation is data-starved (one small subject) |
| More training epochs (40 vs 15) for offline ranking loss | Slightly worse (-0.0016 APFD, p=0.0003) | Not adopted -- mild overfitting; 15 epochs is the practical optimum |
| Generator-stratified fail/pass pair sampling | No effect (-0.0004 APFD, p=0.11) | Not adopted -- did not fix the per-generator inconsistency |
| Transformer encoder backbone (same features/recipe) | Substantially worse (-0.052 APFD, p<1e-25) | Not adopted -- underperforms ResNet under our data/compute budget |

## Table 6: Per-generator breakdown

| generator | tool | apfd_mean | apfd_std |
|---|---|---|---|
| ambiegen | RoadFury | 0.7533 | 0.0557 |
| ambiegen | KSERESNET-V2 (proposed) | 0.7278 | 0.0518 |
| frenetic | RoadFury | 0.8029 | 0.0544 |
| frenetic | KSERESNET-V2 (proposed) | 0.7935 | 0.0522 |
| freneticV | RoadFury | 0.8363 | 0.0453 |
| freneticV | KSERESNET-V2 (proposed) | 0.8221 | 0.0485 |

## Table 6b: Per-generator significance

| generator | comparison | mean_diff | A12 | wilcoxon_p |
|---|---|---|---|---|
| ambiegen | Proposed vs pointwise-features-only | -0.0038 | 0.353 | 3.09e-03 |
| frenetic | Proposed vs pointwise-features-only | +0.0013 | 0.507 | 2.77e-01 |
| freneticV | Proposed vs pointwise-features-only | -0.0078 | 0.253 | 4.02e-07 |