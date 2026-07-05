"""Phase 2: fine-tune the pointwise-pretrained Transformer with the same
pairwise RankNet loss used for V4/V5, on the full (mixed-generator) training
split -- direct architecture comparison against V4 under an identical
training recipe."""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from transformer_model import TransformerPredictor, IN_CH


def main(args):
    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"loaded: X={X.shape} y={y.shape}")

    ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    means = np.array(ckpt["means"], dtype=np.float32)
    stds = np.array(ckpt["stds"], dtype=np.float32)
    Xn = (X - means) / (stds + 1e-8)
    X_t = torch.FloatTensor(Xn)

    fail_idx = np.where(y == 1.0)[0]
    pass_idx = np.where(y == 0.0)[0]

    model = TransformerPredictor()
    model.load_state_dict(ckpt["state"])
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
    parser.add_argument("--base-checkpoint", default="transformer_pointwise.pth")
    parser.add_argument("--out", default="transformer_v6.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-epoch", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.00005)
    main(parser.parse_args())
