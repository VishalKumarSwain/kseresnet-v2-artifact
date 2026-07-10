"""
Lightweight, non-SDC sanity check for RQ1's generality claim (Threats to
Validity, External validity). Uses the identical two-phase recipe
(pointwise pretrain, then pairwise ranking-loss fine-tune) as the main
paper's proposed model, but on a synthetic, non-SDC binary classification
dataset with a small MLP instead of the SE-ResNet -- the architecture is
incidental to the mechanism under test, which is a property of the
training objective, not the network or the domain.

This is NOT a substitute for a second SDC corpus/simulator replication
(no road geometry, no simulator, cannot speak to RQ4's generator-specific
finding). It is evidence that the accuracy/ranking decoupling this paper
diagnoses for KSERESNET is not an artifact specific to SensoDat.

Usage:
    python toy_generality_check.py --seed 42
    for s in 1 7 42 99 123 2026; do python toy_generality_check.py --seed $s; done
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


def apfd(labels, order):
    """order is a list/array of indices into `labels`, ranked descending by model score."""
    trial_labels = labels[order]
    m = trial_labels.sum()
    n = len(order)
    if m == 0 or m == n:
        return None
    fail_positions = [i + 1 for i, idx in enumerate(order) if labels[idx] == 1]
    return 1 - sum(fail_positions) / (n * m) + 1 / (2 * n)


def main(args):
    seed = args.seed
    torch.manual_seed(seed)

    X, y = make_classification(
        n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
        n_clusters_per_class=3, flip_y=0.12, class_sep=0.7,
        weights=[0.7, 0.3], random_state=seed)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_hold = scaler.transform(X_hold).astype(np.float32)

    Xtr, ytr, Xho = torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_hold)

    # phase 1: pointwise pretrain
    torch.manual_seed(seed)
    model_point = MLP(X.shape[1])
    opt = torch.optim.Adam(model_point.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model_point(Xtr).squeeze(1), ytr)
        loss.backward()
        opt.step()

    # phase 2: warm-start, pairwise RankNet-style fine-tune
    model_pair = MLP(X.shape[1])
    model_pair.load_state_dict(model_point.state_dict())
    opt2 = torch.optim.Adam(model_pair.parameters(), lr=5e-5)
    fail_idx = np.where(y_train == 1)[0]
    pass_idx = np.where(y_train == 0)[0]
    prng = np.random.default_rng(seed)
    for _ in range(150):
        fi = prng.choice(fail_idx, size=256, replace=True)
        pj = prng.choice(pass_idx, size=256, replace=True)
        xb = torch.cat([Xtr[fi], Xtr[pj]], dim=0)
        opt2.zero_grad()
        scores = model_pair(xb).squeeze(1)
        n = len(fi)
        loss2 = torch.nn.functional.softplus(-(scores[:n] - scores[n:])).mean()
        loss2.backward()
        opt2.step()

    with torch.no_grad():
        s_point = model_point(Xho).squeeze(1).numpy()
        s_pair = model_pair(Xho).squeeze(1).numpy()

    acc_point = accuracy_score(y_hold, (s_point > 0).astype(int))
    acc_pair = accuracy_score(y_hold, (s_pair > 0).astype(int))

    sizes = [10, 20, 30, 50, 80]
    repeats = 32
    trial_rng = np.random.default_rng(7)
    apfd_point, apfd_pair = [], []
    n_hold = len(y_hold)
    for size in sizes:
        for _ in range(repeats):
            for _ in range(20):
                idx = trial_rng.choice(n_hold, size=size, replace=False)
                if 0 < y_hold[idx].sum() < size:
                    break
            a_point = apfd(y_hold, idx[np.argsort(-s_point[idx])])
            a_pair = apfd(y_hold, idx[np.argsort(-s_pair[idx])])
            if a_point is not None and a_pair is not None:
                apfd_point.append(a_point)
                apfd_pair.append(a_pair)

    apfd_point, apfd_pair = np.array(apfd_point), np.array(apfd_pair)
    d = apfd_pair - apfd_point
    wins, ties, losses = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    a12 = (wins + 0.5 * ties) / len(d)
    p = wilcoxon(d).pvalue if np.any(d) else 1.0

    print(f"seed={seed}  N={len(d)} trials")
    print(f"held-out accuracy: pointwise={acc_point:.4f}  pairwise={acc_pair:.4f}")
    print(f"mean APFD: pointwise={apfd_point.mean():.4f}  pairwise={apfd_pair.mean():.4f}  diff={apfd_pair.mean() - apfd_point.mean():+.4f}")
    print(f"pairwise beats pointwise: wins={wins} ties={ties} losses={losses}  A12={a12:.4f}  wilcoxon p={p:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
