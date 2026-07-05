# KSERESNET-V2: Diagnosing and Repairing Ranking Failures in ML-Based SDC Test Prioritization

Reproducibility artifact for the paper *"When Accuracy Misleads: Diagnosing
and Repairing Ranking Failures in ML-Based Test Prioritization for
Self-Driving Cars"* (submitted to Empirical Software Engineering).

This repository extends [KSERESNET](https://github.com/christianbirchler-org/sdc-testing-competition),
a submission to the ICST 2026 SDC Testing Tool Competition. All baseline
performance figures cited in the paper come from the competition
organizers' [independent evaluation](https://github.com/christianbirchler-org/sdc-testing-competition),
not from KSERESNET's own self-reported results -- see the paper's
Section 5.1 (RQ1) for why that distinction matters.

## What's here

| Folder | Contents |
|---|---|
| `harness/` | Independent, DB-free reimplementation of the organizers' evaluation protocol (APFD, APFD_C, time-to-fault, success rate), validated against their published results within 0.03 APFD |
| `models/proposed_v2/` | The paper's proposed model: 8-channel geometric features + SE-attention 1D-ResNet, trained with a pairwise ranking loss. Includes the trained checkpoint. |
| `models/design_decisions_not_adopted/` | The three robustness checks reported in Table 5 (more training epochs, generator-stratified pair sampling, Transformer backbone) -- kept for transparency, not because they're recommended |
| `models/baseline_reference/` | The original KSERESNET architecture/checkpoint, included only so the harness can run a head-to-head comparison; not a contribution of this repository |
| `data_splits/` | The exact 80/20 train/held-out split used in the paper, as lightweight test-ID lists (not the full feature data -- see below) |
| `results/` | All tables (CSV) and figures (PNG) reported in the paper, plus the scripts that generated them |
| `analysis/` | Statistical consolidation and channel-importance interpretability scripts |
| `paper/` | LaTeX source of the manuscript |

## Why the dataset isn't included directly

The reconstructed SensoDat corpus used in this paper is ~700MB as raw
OpenDRIVE files and 300+MB as preprocessed JSON -- too large for a git
repository. Instead, `data_splits/` contains only the **test IDs** in each
split (a few hundred KB), and `harness/build_real_pool.py` +
`data_splits/regenerate_splits.py` reconstruct the exact same split from
the public SensoDat dataset (Birchler et al., MSR 2024). This keeps the
repository small while making the split exactly reproducible.

## Reproducing the paper's main result

```bash
# 1. Rebuild the dataset (see data_splits/regenerate_splits.py for details)
python harness/build_real_pool.py --sensodat-dir <path> --out full_pool.json
python data_splits/regenerate_splits.py --pool full_pool.json

# 2. Run the proposed model's Docker container (see models/proposed_v2/Dockerfile)
docker build -t kseresnet-v2 models/proposed_v2/
docker run -d --rm -p 50051:50051 kseresnet-v2

# 3. Evaluate against the held-out split
python harness/local_compare.py --port 50051 --data heldout_split.json \
    --sizes 10,20,30,50,80,100,150,200 --repeats 20 --seed 7 --out results.csv
```

This reproduces Table 2's proposed-model row (mean APFD ≈ 0.794).
`harness/multi_tool_compare.py` extends this to run several tools
sequentially for the full head-to-head comparison (Table 2/Figure 1).

## Data availability

- SensoDat dataset: Birchler et al., "SensoDat: Simulation-based sensor
  dataset of self-driving cars," MSR 2024.
- Competition infrastructure and the four 2026 competitor tools:
  https://github.com/christianbirchler-org/sdc-testing-competition
- This repository: harness, proposed model + checkpoint, all reported
  results, and the paper source.

## Citation

See `CITATION.cff`.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
