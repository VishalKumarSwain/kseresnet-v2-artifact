import grpc
from concurrent import futures
import time
import numpy as np
import torch
import torch.nn as nn
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc
import gc

# --- CONFIGURATION ---
MAX_POINTS     = 200
INPUT_CHANNELS = 4

# Default normalization constants (tuned on 956-road training set)
# These are overwritten at runtime when Initialize() receives oracle roads
DEFAULT_NORM = {
    "dx_mean":  -0.02, "dx_std": 0.5,
    "dy_mean":   0.20, "dy_std": 0.5,
    "accel_scale": 5.0,
}

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_raw_features(points):
    """Return raw (un-normalized) [N, 4] feature array from road points."""
    coords = np.array([[float(p.x), float(p.y)] for p in points])
    if len(coords) < 3:
        return np.zeros((MAX_POINTS, INPUT_CHANNELS))

    deltas    = coords[1:] - coords[:-1]                        # dx, dy
    accel_raw = deltas[1:] - deltas[:-1]                        # ax, ay
    accel     = np.vstack((np.zeros((1, 2)), accel_raw))

    features = np.hstack((deltas, accel))                       # [N-1, 4]

    if len(features) >= MAX_POINTS:
        return features[:MAX_POINTS]
    padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS))
    return np.vstack((features, padding))


def normalize_features(features, norm):
    """Apply per-channel normalization to a [MAX_POINTS, 4] array."""
    out = features.copy()
    out[:, 0] = (out[:, 0] - norm["dx_mean"])  / norm["dx_std"]
    out[:, 1] = (out[:, 1] - norm["dy_mean"])  / norm["dy_std"]
    out[:, 2] = out[:, 2] * norm["accel_scale"]
    out[:, 3] = out[:, 3] * norm["accel_scale"]
    return out


def compute_norm_from_features(feature_list):
    """
    Given a list of raw [MAX_POINTS, 4] arrays (from oracle roads),
    compute mean/std for dx and dy so that normalization is correct
    for whatever distribution this dataset comes from.
    """
    all_feats = np.vstack(feature_list)          # [N*200, 4]
    nonzero   = np.any(all_feats != 0, axis=1)   # ignore padding rows
    valid     = all_feats[nonzero]

    if len(valid) < 10:                          # too little data — use defaults
        return DEFAULT_NORM.copy()

    dx_mean = float(np.mean(valid[:, 0]))
    dx_std  = max(float(np.std(valid[:, 0])),  1e-6)
    dy_mean = float(np.mean(valid[:, 1]))
    dy_std  = max(float(np.std(valid[:, 1])),  1e-6)

    # Accel scale: target std ≈ 0.2  (same as trained with accel*5 when std≈0.04)
    ax_std  = max(float(np.std(valid[:, 2])), 1e-6)
    accel_scale = 0.2 / ax_std

    return {
        "dx_mean": dx_mean, "dx_std": dx_std,
        "dy_mean": dy_mean, "dy_std": dy_std,
        "accel_scale": accel_scale,
    }


# ── Architecture ──────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(channels)
        self.se    = SEBlock(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + res)


class ResNetPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=5, padding=2)
        self.res_block1  = ResidualBlock(32)
        self.pool1       = nn.MaxPool1d(2)
        self.res_block2  = ResidualBlock(32)
        self.pool2       = nn.MaxPool1d(2)
        self.res_block3  = ResidualBlock(32)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 50, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.res_block1(self.input_layer(x)))
        x = self.pool2(self.res_block2(x))
        x = self.res_block3(x)
        return self.fc(x)


# ── gRPC service ──────────────────────────────────────────────────────────────
class MyPrioritizer(pb2_grpc.CompetitionToolServicer):

    def __init__(self):
        print("Loading KSERESNET...")
        self.model = ResNetPredictor()
        try:
            self.model.load_state_dict(torch.load("crash_model.pth", map_location="cpu"))
            self.model.eval()
            print("Model loaded.")
        except Exception as e:
            print(f"ERROR loading model: {e}")

        # Start with default normalization; updated in Initialize()
        self.norm = DEFAULT_NORM.copy()

    def Name(self, request, context):
        return pb2.NameReply(name="KSERESNET")

    def Initialize(self, request_iterator, context):
        """
        Collect oracle road features → compute dataset-specific normalization.
        This makes KSERESNET work correctly on any road distribution,
        not just the 956-road training distribution.
        """
        raw_feature_list = []
        oracle_count = 0

        for oracle in request_iterator:
            raw = extract_raw_features(oracle.testCase.roadPoints)
            raw_feature_list.append(raw)
            oracle_count += 1

        if oracle_count > 0:
            self.norm = compute_norm_from_features(raw_feature_list)
            print(f"Initialize: {oracle_count} oracles → "
                  f"dx_mean={self.norm['dx_mean']:.4f}  "
                  f"dy_mean={self.norm['dy_mean']:.4f}  "
                  f"accel_scale={self.norm['accel_scale']:.4f}")
        else:
            self.norm = DEFAULT_NORM.copy()
            print("Initialize: no oracle data — using default normalization.")

        return pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        scored = []
        with torch.no_grad():
            for test in request_iterator:
                raw      = extract_raw_features(test.roadPoints)
                features = normalize_features(raw, self.norm)
                tensor   = torch.FloatTensor(features).unsqueeze(0)
                score    = self.model(tensor).item()
                scored.append((str(test.testId), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        for tid, _ in scored:
            yield pb2.PrioritizationReply(testId=tid)
        gc.collect()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_CompetitionToolServicer_to_server(MyPrioritizer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("KSERESNET listening on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
