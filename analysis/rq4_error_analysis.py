"""
RQ4 example-level error analysis: for each generator's held-out pool,
score every test with both the intermediate model (features + pointwise
fine-tune, crash_model_v2.pth) and the proposed model (features + pairwise
ranking loss, crash_model_v4.pth), and examine individual FAIL-test rank
changes.

Two questions this answers, both reported in the paper's RQ4 section:
  1. Is there a geometric signature (curvature, angle_change) that
     distinguishes demoted FAIL tests from the rest? (No -- checked via
     decile comparison, no separation found.)
  2. When a FAIL test is demoted, are the tests that overtake it mostly
     other FAIL tests (benign redistribution of confidence among true
     failures) or PASS tests (a genuine rank inversion that costs APFD)?
     Both occur; the mix is reported per generator.

Usage:
    python rq4_error_analysis.py --data-dir <path to data/ with heldout_*.json>
                                   --v2-checkpoint <path to crash_model_v2.pth>
                                   --v4-checkpoint <path to crash_model_v4.pth>
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

MAX_POINTS = 200
IN_CH = 8
GENERATORS = ["ambiegen", "frenetic", "freneticV"]


def compute_features_v2(coords):
    if len(coords) < 3:
        return np.zeros((MAX_POINTS, IN_CH), dtype=np.float32)
    deltas = coords[1:] - coords[:-1]
    dx, dy = deltas[:, 0], deltas[:, 1]
    accel_raw = deltas[1:] - deltas[:-1]
    accel = np.vstack((np.zeros((1, 2)), accel_raw))
    ax, ay = accel[:, 0], accel[:, 1]
    seg_len = np.sqrt(dx**2 + dy**2)
    heading = np.arctan2(dy, dx)
    angle_raw = np.diff(heading)
    angle_wrap = (angle_raw + np.pi) % (2 * np.pi) - np.pi
    angle_change = np.concatenate([[0.0], angle_wrap])
    curvature = np.where(seg_len > 1e-6, angle_change / (seg_len + 1e-8), 0.0)
    n = len(curvature)
    local_curv_std = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = max(0, i - 2); e = min(n, i + 3)
        local_curv_std[i] = float(np.std(curvature[s:e]))
    raw = np.column_stack([dx, dy, ax, ay, curvature, angle_change, seg_len, local_curv_std]).astype(np.float32)
    if len(raw) >= MAX_POINTS:
        return raw[:MAX_POINTS]
    pad = np.zeros((MAX_POINTS - len(raw), IN_CH), dtype=np.float32)
    return np.vstack((raw, pad))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1).expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + res)


class ResNetPredictorV2(nn.Module):
    def __init__(self, in_channels=IN_CH):
        super().__init__()
        self.input_layer = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.res_block1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool1d(2)
        self.res_block2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool1d(2)
        self.res_block3 = ResidualBlock(32)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 50, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.res_block1(self.input_layer(x)))
        x = self.pool2(self.res_block2(x))
        x = self.res_block3(x)
        return self.fc(x)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    m = ResNetPredictorV2()
    m.load_state_dict(ckpt["state"])
    m.eval()
    return m, np.array(ckpt["means"], dtype=np.float32), np.array(ckpt["stds"], dtype=np.float32)


def score_all(model, means, stds, feats):
    Xn = (feats - means) / (stds + 1e-8)
    with torch.no_grad():
        return model(torch.FloatTensor(Xn)).squeeze(1).numpy()


def main(args):
    v2_model, v2_means, v2_stds = load_model(args.v2_checkpoint)
    v4_model, v4_means, v4_stds = load_model(args.v4_checkpoint)

    for gen in GENERATORS:
        with open(f"{args.data_dir}/heldout_{gen}.json") as f:
            raw = json.load(f)

        feats, labels = [], []
        for t in raw:
            coords = np.array([[p["x"], p["y"]] for p in t["road_points"]], dtype=np.float32)
            feats.append(compute_features_v2(coords))
            labels.append(1 if t["meta_data"]["test_info"]["test_outcome"] == "FAIL" else 0)
        feats = np.array(feats, dtype=np.float32)
        labels = np.array(labels)
        n = len(labels)

        s2 = score_all(v2_model, v2_means, v2_stds, feats)
        s4 = score_all(v4_model, v4_means, v4_stds, feats)
        rank2 = (-s2).argsort().argsort()
        rank4 = (-s4).argsort().argsort()

        fail_idx = np.where(labels == 1)[0]
        demoted = [i for i in fail_idx if rank4[i] > rank2[i]]

        # (1) geometric-signature check: decile comparison
        curv = np.array([np.mean(np.abs(feats[i, :, 4])) for i in range(n)])
        ang = np.array([np.mean(np.abs(feats[i, :, 5])) for i in range(n)])
        delta = rank4[fail_idx] - rank2[fail_idx]
        order = np.argsort(-delta)
        k = max(5, len(fail_idx) // 10)
        worst, best = fail_idx[order[:k]], fail_idx[order[-k:]]
        print(f"\n=== {gen} ===")
        print(f"geometric signature: worst-demoted decile mean|curv|={curv[worst].mean():.4f} "
              f"vs. best/promoted decile mean|curv|={curv[best].mean():.4f} "
              f"(all FAIL: {curv[fail_idx].mean():.4f}) -- {'separates' if abs(curv[worst].mean()-curv[best].mean()) > 0.005 else 'does not separate'}")

        # (2) fail-vs-fail vs fail-vs-pass overtake categorization
        pass_overtake_fracs = []
        for i in demoted:
            r2i, r4i = rank2[i], rank4[i]
            overtakers = [j for j in range(n) if rank4[j] < r4i and rank2[j] > r2i]
            if overtakers:
                pass_overtake_fracs.append(sum(1 for j in overtakers if labels[j] == 0) / len(overtakers))
        pass_overtake_fracs = np.array(pass_overtake_fracs)
        print(f"demoted FAIL tests: {len(demoted)}/{len(fail_idx)}; "
              f"{(pass_overtake_fracs > 0).mean()*100:.0f}% are overtaken by at least one PASS test "
              f"(mean overtaker pass-fraction={pass_overtake_fracs.mean():.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="directory containing heldout_<generator>.json files")
    parser.add_argument("--v2-checkpoint", default="crash_model_v2.pth")
    parser.add_argument("--v4-checkpoint", default="crash_model_v4.pth")
    main(parser.parse_args())
