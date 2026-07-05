"""
V4: V2 features/architecture, base model trained offline on the full
28.8k-test training split with a pairwise RankNet ranking loss (see
train_ranking_v4.py), warm-started from V2's checkpoint. Initialize() only
does normalization-stat blending (no online fine-tuning), consistent with
the ablation finding that per-subject online adaptation adds ~nothing.
"""
import grpc
from concurrent import futures
import numpy as np
import torch
import torch.nn as nn
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc
import gc

MAX_POINTS = 200
IN_CH = 8


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


def road_points_to_coords(road_points):
    return np.array([[float(p.x), float(p.y)] for p in road_points], dtype=np.float32)


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


class KSERESNETv4(pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        print("Loading KSERESNET-V5 (offline ranking-trained) ...")
        self.model = ResNetPredictorV2(in_channels=IN_CH)
        checkpoint = torch.load("crash_model_v5.pth", map_location="cpu")
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()
        self.means = np.array(checkpoint["means"], dtype=np.float32)
        self.stds = np.array(checkpoint["stds"], dtype=np.float32)

    def _normalize(self, feat):
        return (feat - self.means) / (self.stds + 1e-8)

    def _featurize(self, road_points):
        coords = road_points_to_coords(road_points)
        raw = compute_features_v2(coords)
        return self._normalize(raw).astype(np.float32)

    def Name(self, request, context):
        return pb2.NameReply(name="KSERESNET-V5")

    def Initialize(self, request_iterator, context):
        all_raw = []
        for oracle in request_iterator:
            coords = road_points_to_coords(oracle.testCase.roadPoints)
            if len(coords) < 3:
                continue
            all_raw.append(compute_features_v2(coords))
        if all_raw:
            stacked = np.vstack(all_raw)
            new_means = stacked.mean(axis=0).astype(np.float32)
            new_stds = np.maximum(stacked.std(axis=0), 1e-4).astype(np.float32)
            self.means = 0.7 * self.means + 0.3 * new_means
            self.stds = 0.7 * self.stds + 0.3 * new_stds
        return pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        scored = []
        with torch.no_grad():
            for tc in request_iterator:
                feat = self._featurize(tc.roadPoints)
                score = self.model(torch.FloatTensor(feat).unsqueeze(0)).item()
                scored.append((str(tc.testId), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        for tid, _ in scored:
            yield pb2.PrioritizationReply(testId=tid)
        gc.collect()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_CompetitionToolServicer_to_server(KSERESNETv4(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("KSERESNET-V5 listening on 50051 ...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
