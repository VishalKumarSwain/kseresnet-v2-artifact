These two scripts are the original KSERESNET competition submission's
development artifacts, included as evidence for the paper's explanation
(Section 3, RQ1) of why the self-reported APFD (0.8131) diverges from the
organizers' independent measurement (0.625).

- `train.py`: trains the baseline model on `sdc-test-data.json`, a
  956-test local development sample bundled with the competition
  repository (not the 32,580-test corpus used for official scoring).
- `final_score_check.py`: the self-verification script. It scores the
  model on the exact same 956-test file used for training, computing a
  single APFD over that entire fixed set in one pass, with no held-out
  split and no random-subject-sampling protocol.

Together these are concrete, checked explanations for the discrepancy:
train/evaluation leakage, a non-representative sample, and a different
(single-shot) evaluation protocol than the organizers' repeated-trial
harness used throughout the rest of this paper.
