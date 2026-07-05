"""Channel-importance (permutation/zeroing) analysis for KSERESNET-V4 on the
held-out split: for each of the 8 input channels, zero it out (its
normalized value -> 0, i.e. its mean) and measure the drop in fail/pass
separation AUC. Larger AUC drop = more important channel."""
import numpy as np
import torch
import torch.nn as nn

IN_CH = 8
CHANNEL_NAMES = ["dx", "dy", "ax", "ay", "curvature", "angle_change", "seg_length", "local_curv_std"]


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


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos = ranks[labels == 1]
    neg = ranks[labels == 0]
    # Mann-Whitney U based AUC
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    data = np.load("heldout_features.npz", allow_pickle=True)
    X, y = data["X"], data["y"]

    ckpt = torch.load("../KSERESNET_V4/crash_model_v4.pth", map_location="cpu")
    means = np.array(ckpt["means"], dtype=np.float32)
    stds = np.array(ckpt["stds"], dtype=np.float32)
    model = ResNetPredictorV2()
    model.load_state_dict(ckpt["state"])
    model.eval()

    Xn = (X - means) / (stds + 1e-8)

    # subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xn), size=min(3000, len(Xn)), replace=False)
    Xs, ys = Xn[idx], y[idx]

    with torch.no_grad():
        base_scores = model(torch.FloatTensor(Xs)).squeeze(1).numpy()
    base_auc = auc(base_scores, ys)
    print(f"baseline AUC (fail vs pass separation): {base_auc:.4f}\n")

    results = []
    for ch in range(IN_CH):
        Xp = Xs.copy()
        Xp[:, :, ch] = 0.0  # zero = channel mean after normalization
        with torch.no_grad():
            scores = model(torch.FloatTensor(Xp)).squeeze(1).numpy()
        a = auc(scores, ys)
        drop = base_auc - a
        results.append((CHANNEL_NAMES[ch], a, drop))

    results.sort(key=lambda r: -r[2])
    print(f"{'channel':<15}{'AUC when zeroed':<18}{'AUC drop':<12}")
    for name, a, drop in results:
        print(f"{name:<15}{a:<18.4f}{drop:<12.4f}")


if __name__ == "__main__":
    main()
