"""
Real (not one-shot) ranking-loss training: warm-start from V2's checkpoint,
then train with a pairwise RankNet loss over the full 28.8k-test training
split, sampling many (fail, pass) pairs per epoch in minibatches.
Saves crash_model_v4.pth in the same {"state","means","stds"} format V2 uses.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

IN_CH = 8


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


def main(args):
    data = np.load(args.features)
    X, y = data["X"], data["y"]
    print(f"loaded features: X={X.shape} y={y.shape}  fail={int(y.sum())} pass={int((1-y).sum())}")

    means = X.reshape(-1, IN_CH).mean(axis=0).astype(np.float32)
    stds = np.maximum(X.reshape(-1, IN_CH).std(axis=0), 1e-4).astype(np.float32)
    Xn = (X - means) / (stds + 1e-8)
    X_t = torch.FloatTensor(Xn)

    fail_idx = np.where(y == 1.0)[0]
    pass_idx = np.where(y == 0.0)[0]
    print(f"fail_idx={len(fail_idx)} pass_idx={len(pass_idx)}")

    model = ResNetPredictorV2(in_channels=IN_CH)
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    model.load_state_dict(base_ckpt["state"])
    print("warm-started from", args.base_checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(42)

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        fi = rng.choice(fail_idx, size=args.pairs_per_epoch, replace=True)
        pj = rng.choice(pass_idx, size=args.pairs_per_epoch, replace=True)
        perm = rng.permutation(args.pairs_per_epoch)
        fi, pj = fi[perm], pj[perm]

        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, args.pairs_per_epoch, args.batch_size):
            bi = fi[s:s + args.batch_size]
            bj = pj[s:s + args.batch_size]
            xb = torch.cat([X_t[bi], X_t[bj]], dim=0)
            opt.zero_grad()
            scores = model(xb).squeeze(1)
            n = len(bi)
            score_fail, score_pass = scores[:n], scores[n:]
            loss = torch.nn.functional.softplus(-(score_fail - score_pass)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch+1}/{args.epochs}  loss={epoch_loss/n_batches:.4f}  ({time.time()-t0:.0f}s)")

    model.eval()
    torch.save({"state": model.state_dict(), "means": means.tolist(), "stds": stds.tolist()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="train_features.npz")
    parser.add_argument("--base-checkpoint", default="crash_model_v2.pth")
    parser.add_argument("--out", default="crash_model_v4.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00005)
    main(parser.parse_args())
