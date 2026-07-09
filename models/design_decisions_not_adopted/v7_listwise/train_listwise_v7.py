"""
V7: robustness check requested in review -- same architecture, features, and
training split as V4 (train_ranking_v4.py), but a listwise (ListNet-style)
ranking loss instead of the pairwise RankNet loss. Warm-starts from V2's
pointwise checkpoint, same as V4/V5/V6.

ListNet top-1 loss over sampled lists: for a list of model scores s and
binary relevance y (FAIL=1, PASS=0), the target distribution places equal
mass on all FAIL items and zero on PASS items; the predicted distribution
is softmax(s) over the list. Loss is the cross-entropy between the two.
Lists with zero FAIL items carry no ranking signal and are skipped, the
listwise analogue of train_ranking_v4.py's fail/pass pairing.
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
    print(f"loaded features: X={X.shape} y={y.shape}  fail={int(y.sum())} pass={int((1 - y).sum())}")

    means = X.reshape(-1, IN_CH).mean(axis=0).astype(np.float32)
    stds = np.maximum(X.reshape(-1, IN_CH).std(axis=0), 1e-4).astype(np.float32)
    Xn = (X - means) / (stds + 1e-8)
    X_t = torch.FloatTensor(Xn)
    y_t = torch.FloatTensor(y)

    n_total = len(y)
    model = ResNetPredictorV2(in_channels=IN_CH)
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    model.load_state_dict(base_ckpt["state"])
    print("warm-started from", args.base_checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(42)

    lists_per_epoch = args.pairs_per_epoch // args.list_size

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_lists_used = 0
        list_idx = rng.integers(0, n_total, size=(lists_per_epoch, args.list_size))

        for s in range(0, lists_per_epoch, args.batch_lists):
            batch_lists = list_idx[s:s + args.batch_lists]
            opt.zero_grad()
            batch_loss = 0.0
            n_valid = 0
            for lst in batch_lists:
                yl = y_t[lst]
                n_fail = yl.sum()
                if n_fail.item() == 0:
                    continue
                xl = X_t[lst]
                scores = model(xl).squeeze(1)
                log_p = torch.log_softmax(scores, dim=0)
                target = yl / n_fail
                loss_i = -(target * log_p).sum()
                batch_loss = batch_loss + loss_i
                n_valid += 1
            if n_valid == 0:
                continue
            batch_loss = batch_loss / n_valid
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += batch_loss.item()
            n_lists_used += n_valid
        n_batches = max(1, (lists_per_epoch + args.batch_lists - 1) // args.batch_lists)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={epoch_loss / n_batches:.4f}  "
              f"lists_used={n_lists_used}/{lists_per_epoch}  ({time.time() - t0:.0f}s)")

    model.eval()
    torch.save({"state": model.state_dict(), "means": means.tolist(), "stds": stds.tolist()}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="train_features.npz")
    parser.add_argument("--base-checkpoint", default="crash_model_v2.pth")
    parser.add_argument("--out", default="crash_model_v7.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000, help="total items sampled per epoch (list_size * lists_per_epoch)")
    parser.add_argument("--list-size", type=int, default=32)
    parser.add_argument("--batch-lists", type=int, default=16, help="lists per optimizer step")
    parser.add_argument("--lr", type=float, default=0.00005)
    main(parser.parse_args())
