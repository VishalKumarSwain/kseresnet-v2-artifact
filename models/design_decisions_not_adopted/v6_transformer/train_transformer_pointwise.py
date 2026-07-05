"""Phase 1: pretrain the Transformer from scratch with pointwise
BCEWithLogitsLoss on the full training split (class-balanced), mirroring
how the base ResNet checkpoint was presumably trained. This gives the
ranking-loss fine-tune (phase 2) a reasonable starting point instead of
training a transformer from random init purely with a pairwise objective."""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from transformer_model import TransformerPredictor, IN_CH


def main(args):
    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"loaded: X={X.shape} y={y.shape} fail={int(y.sum())} pass={int((1-y).sum())}")

    means = X.reshape(-1, IN_CH).mean(axis=0).astype(np.float32)
    stds = np.maximum(X.reshape(-1, IN_CH).std(axis=0), 1e-4).astype(np.float32)
    Xn = (X - means) / (stds + 1e-8)
    X_t = torch.FloatTensor(Xn)
    y_t = torch.FloatTensor(y).unsqueeze(1)

    fail_idx = np.where(y == 1.0)[0]
    pass_idx = np.where(y == 0.0)[0]

    model = TransformerPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(42)

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        n_target = args.samples_per_epoch // 2
        fi = rng.choice(fail_idx, size=n_target, replace=True)
        pi = rng.choice(pass_idx, size=n_target, replace=True)
        idx = np.concatenate([fi, pi])
        rng.shuffle(idx)

        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, len(idx), args.batch_size):
            bidx = idx[s:s + args.batch_size]
            xb, yb = X_t[bidx], y_t[bidx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
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
    parser.add_argument("--out", default="transformer_pointwise.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    main(parser.parse_args())
