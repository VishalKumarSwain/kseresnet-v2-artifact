"""Precompute V2-style 8-channel features for the training split so the
ranking-loss training loop doesn't recompute curvature/etc. every epoch."""
import json
import time
import numpy as np

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


def generator_of(test_id):
    if "frenetic_v" in test_id or "freneticV" in test_id:
        return "freneticV"
    if "frenetic" in test_id:
        return "frenetic"
    if "ambiegen" in test_id:
        return "ambiegen"
    return "unknown"


def main(in_path, out_path):
    with open(in_path) as f:
        raw = json.load(f)
    N = len(raw)
    X = np.zeros((N, MAX_POINTS, IN_CH), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)
    gen = np.empty((N,), dtype=object)
    t0 = time.time()
    for i, t in enumerate(raw):
        coords = np.array([[p["x"], p["y"]] for p in t["road_points"]], dtype=np.float32)
        X[i] = compute_features_v2(coords)
        y[i] = 1.0 if t["meta_data"]["test_info"]["test_outcome"] == "FAIL" else 0.0
        gen[i] = generator_of(t["_id"]["$oid"])
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{N} in {time.time()-t0:.0f}s")
    np.savez(out_path, X=X, y=y, gen=gen)
    print(f"saved {out_path}: X={X.shape} y={y.shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
