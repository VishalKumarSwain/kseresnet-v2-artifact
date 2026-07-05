"""
Local, DB-free re-implementation of the ICST 2026 evaluation protocol.

Loads the flat sample benchmark (data/sdc-test-data.json), samples random
subjects (test suites), drives a tool's gRPC service through Initialize/
Prioritize, and scores results with the organizers' own metrics.py so
numbers are directly comparable to Table III of the competition report.

Usage:
    python local_compare.py --port 50061 --label KSERESNET-V1 \
        --data ../../../data/sdc-test-data.json \
        --sizes 10,20,30,50,80 --repeats 10 --seed 42 --out results_v1.csv
"""
import argparse
import csv
import random
import statistics
import time

import grpc

import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc
import errors
import mydata
from metrics import MetricEvaluator


def load_pool(path):
    import json
    with open(path, "r") as fp:
        raw = json.load(fp)
    pool = []
    for t in raw:
        pool.append({
            "test_id": t["_id"]["$oid"],
            "hasFailed": t["meta_data"]["test_info"]["test_outcome"] == "FAIL",
            "sim_time": t["meta_data"]["test_info"]["test_duration"],
            "road_points": [(p["x"], p["y"]) for p in t["road_points"]],
        })
    return pool


def make_subject(subject_id, test_dicts):
    pg_test_data = [
        mydata.PGTestData(
            test_case_id=i,
            sensodat_collection_id=-1,
            object_id=t["test_id"],
            has_passed=not t["hasFailed"],
            has_failed=t["hasFailed"],
            risk_factor=-1.0,
            oob=-1.0,
            max_speed_kmh=-1,
            is_valid=True,
            sensodat_file_path="",
            duration_seconds=t["sim_time"],
        )
        for i, t in enumerate(test_dicts)
    ]
    return mydata.Subject(subject_id=subject_id, pg_test_data=pg_test_data)


def sample_subject(rng, pool, size, subject_id):
    for _ in range(20):
        picked = rng.sample(pool, size)
        n_fail = sum(1 for t in picked if t["hasFailed"])
        if 0 < n_fail < size:
            break
    return make_subject(subject_id, picked)


def oracle_iterator(subject):
    for t in subject.pg_test_data:
        road_points = [
            pb2.RoadPoint(sequenceNumber=i, x=x, y=y)
            for i, (x, y) in enumerate(_lookup_points[t.object_id])
        ]
        tc = pb2.SDCTestCase(testId=t.object_id, roadPoints=road_points)
        yield pb2.Oracle(testCase=tc, hasFailed=t.has_failed)


def testcase_iterator(subject):
    for t in subject.pg_test_data:
        road_points = [
            pb2.RoadPoint(sequenceNumber=i, x=x, y=y)
            for i, (x, y) in enumerate(_lookup_points[t.object_id])
        ]
        yield pb2.SDCTestCase(testId=t.object_id, roadPoints=road_points)


def wait_for_server(channel, timeout=120):
    grpc.channel_ready_future(channel).result(timeout=timeout)


def run(args):
    global _lookup_points
    pool = load_pool(args.data)
    _lookup_points = {t["test_id"]: t["road_points"] for t in pool}

    channel = grpc.insecure_channel(f"localhost:{args.port}")
    wait_for_server(channel)
    stub = pb2_grpc.CompetitionToolStub(channel)
    tool_name = stub.Name(pb2.Empty()).name
    print(f"Connected to '{tool_name}' on port {args.port}")

    metric_eval = MetricEvaluator()
    rng = random.Random(args.seed)
    rows = []
    sizes = [int(s) for s in args.sizes.split(",")]

    trial_id = 0
    for size in sizes:
        for rep in range(args.repeats):
            trial_id += 1
            init_subj = sample_subject(rng, pool, size, subject_id=trial_id * 2)
            eval_subj = sample_subject(rng, pool, size, subject_id=trial_id * 2 + 1)

            row = {"tool": tool_name, "subject_size": size, "trial": rep, "success": False}
            try:
                t0 = time.time()
                init_resp = stub.Initialize(oracle_iterator(init_subj))
                if not init_resp.ok:
                    raise errors.InitializationError("tool returned ok=False")
                row["t_initialize"] = time.time() - t0

                t0 = time.time()
                reply_iter = stub.Prioritize(testcase_iterator(eval_subj))
                prioritized = [r.testId for r in reply_iter]
                row["t_prioritize_tests"] = time.time() - t0

                metric_eval.check_prioritization_validity(eval_subj, prioritized)

                row["apfd"] = metric_eval.compute_apfd(eval_subj, prioritized)
                row["apfdc"] = metric_eval.compute_apdfc(eval_subj, prioritized)
                row["t_first_fault"] = metric_eval.compute_time_to_first_fault(eval_subj, prioritized)
                row["t_last_fault"] = metric_eval.compute_time_to_last_fault(eval_subj, prioritized)
                row["success"] = True
                print(f"[{tool_name}] size={size} rep={rep}: OK apfd={row['apfd']:.3f} apfdc={row['apfdc']:.3f}")
            except (errors.InitializationError, errors.PrioritizationError, grpc.RpcError) as e:
                print(f"[{tool_name}] size={size} rep={rep}: FAILED ({e})")
            rows.append(row)

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "tool", "subject_size", "trial", "success",
            "t_initialize", "t_prioritize_tests", "apfd", "apfdc",
            "t_first_fault", "t_last_fault",
        ])
        writer.writeheader()
        writer.writerows(rows)

    successes = [r for r in rows if r["success"]]
    print(f"\n=== Summary for {tool_name} ===")
    print(f"Success rate: {len(successes)}/{len(rows)} ({100*len(successes)/len(rows):.1f}%)")
    for metric in ["apfd", "apfdc", "t_prioritize_tests", "t_first_fault", "t_last_fault"]:
        vals = [r[metric] for r in successes if r.get(metric) is not None]
        if vals:
            print(f"{metric}: mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f} "
                  f"min={min(vals):.4f} max={max(vals):.4f}")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", default=None, help="only used for logging; tool name comes from Name() rpc")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sizes", default="10,20,30,50,80")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="local_compare_results.csv")
    run(parser.parse_args())
