"""
V6: same 8-channel features as V2/V4, but a Transformer encoder backbone
(2 layers, 4 heads, Pre-LN, d_model=64) instead of the SE-attention 1D-ResNet,
trained with the same two-phase recipe as V4 (pointwise pretrain -> pairwise
ranking fine-tune) for a clean architecture-only comparison.
"""
import grpc
from concurrent import futures
import math
import numpy as np
import torch
import torch.nn as nn
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc
import gc

MAX_POINTS = 200
IN_CH = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FF = 128
MAX_LEN = 201


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    def __init__(self, in_channels=IN_CH, d_model=D_MODEL, nhead=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.score_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        b = x.size(0)
        h = self.input_proj(x)
        tok = self.score_token.expand(b, -1, -1)
        h = torch.cat([tok, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h[:, 0])


class KSERESNETv6(pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        print("Loading KSERESNET-V6 (Transformer) ...")
        self.model = TransformerPredictor()
        checkpoint = torch.load("transformer_v6.pth", map_location="cpu")
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
        return pb2.NameReply(name="KSERESNET-V6")

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
    pb2_grpc.add_CompetitionToolServicer_to_server(KSERESNETv6(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("KSERESNET-V6 listening on 50051 ...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
