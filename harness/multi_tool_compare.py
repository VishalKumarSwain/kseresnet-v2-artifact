"""
Generalized head-to-head harness across the 2026 prioritization tools
(and, via the "select" interface branch, the 2025 selection tools).

Runs tools sequentially (one Docker container at a time) against the same
seeded random subjects drawn from a real/sample SensoDat pool, and scores
them with the organizers' own metrics.py formulas -- so results are directly
comparable across every tool, and across the earlier KSERESNET V1/V2 runs
that used local_compare.py with the same seed/sizes/repeats.

2025 "select" tools return a subset of test IDs; we convert that into a
full ordering by placing selected tests first (in emission order) and
appending the remaining, non-selected tests afterwards in the same random
order used for sampling -- this lets APFD/APFD_C be computed uniformly.
"""
import argparse
import csv
import json
import random
import statistics
import subprocess
import time

import grpc

import competition_2026_pb2 as pb2
import errors
import mydata
from metrics import MetricEvaluator

TOOLS = [
    {"name": "RoadFury",    "image": "roadfury:latest",           "interface": "prioritize", "extra_args": ["-p", "50051"]},
    {"name": "EagleSemble", "image": "eaglesemble:latest",        "interface": "prioritize", "extra_args": ["-p", "50051"]},
    {"name": "RTE4SDC",     "image": "rte4sdc:latest",            "interface": "prioritize", "extra_args": ["-p", "50051"]},
    {"name": "ITEP4SDC",    "image": "itep4sdc:latest",           "interface": "prioritize", "extra_args": ["-p", "50051"]},
    {"name": "Random",      "image": "sample_tool_python:latest", "interface": "prioritize", "extra_args": ["-p", "50051"]},
    {"name": "KSERESNET",   "image": "kseresnet:latest",          "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V2","image": "kseresnet-v2:latest",       "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V2-NOFT", "image": "kseresnet-v2-noft:latest", "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V1-FT",   "image": "kseresnet-v1-ft:latest",   "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V3",      "image": "kseresnet-v3:latest",      "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V4",      "image": "kseresnet-v4:latest",      "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V4B",     "image": "kseresnet-v4b:latest",     "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V5",      "image": "kseresnet-v5:latest",      "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V6",      "image": "kseresnet-v6:latest",      "interface": "prioritize", "extra_args": []},
    {"name": "KSERESNET-V7",      "image": "kseresnet-v7:latest",      "interface": "prioritize", "extra_args": []},
    # 2025 edition -- test *selection* tools, converted to a full ordering (selected first, rest appended)
    {"name": "CertiFail",          "image": "certifail-2025:latest",           "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "Detour",             "image": "detour-2025:latest",              "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "DRVN",               "image": "drvn-2025:latest",                "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "ITS4SDC",            "image": "its4sdc-2025:latest",             "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "CurvatureSelector",  "image": "curvature-selector-2025:latest",  "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "GraphSelector",      "image": "graph-selector-2025:latest",      "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "MLSelector",         "image": "ml-selector-2025:latest",         "interface": "select", "extra_args": ["-p", "50051"]},
    {"name": "TransformerSelector","image": "transformer-selector-2025:latest","interface": "select", "extra_args": ["-p", "50051"]},
]

PORT = 50051


def load_pool(path):
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
            test_case_id=i, sensodat_collection_id=-1, object_id=t["test_id"],
            has_passed=not t["hasFailed"], has_failed=t["hasFailed"],
            risk_factor=-1.0, oob=-1.0, max_speed_kmh=-1, is_valid=True,
            sensodat_file_path="", duration_seconds=t["sim_time"],
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


def road_points_msgs(points):
    return [pb2.RoadPoint(sequenceNumber=i, x=x, y=y) for i, (x, y) in enumerate(points)]


def oracle_iterator(subject, lookup):
    for t in subject.pg_test_data:
        tc = pb2.SDCTestCase(testId=t.object_id, roadPoints=road_points_msgs(lookup[t.object_id]))
        yield pb2.Oracle(testCase=tc, hasFailed=t.has_failed)


def testcase_iterator(subject, lookup):
    for t in subject.pg_test_data:
        yield pb2.SDCTestCase(testId=t.object_id, roadPoints=road_points_msgs(lookup[t.object_id]))


def make_calls(channel):
    name_call = channel.unary_unary("/CompetitionTool/Name",
        request_serializer=pb2.Empty.SerializeToString, response_deserializer=pb2.NameReply.FromString)
    init_call = channel.stream_unary("/CompetitionTool/Initialize",
        request_serializer=lambda m: m.SerializeToString(), response_deserializer=pb2.InitializationReply.FromString)
    prioritize_call = channel.stream_stream("/CompetitionTool/Prioritize",
        request_serializer=lambda m: m.SerializeToString(), response_deserializer=pb2.PrioritizationReply.FromString)
    select_call = channel.stream_stream("/CompetitionTool/Select",
        request_serializer=lambda m: m.SerializeToString(), response_deserializer=pb2.PrioritizationReply.FromString)
    return name_call, init_call, prioritize_call, select_call


def start_container(tool, container_name):
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    cmd = ["docker", "run", "-d", "--rm", "-p", f"{PORT}:{PORT}", "--name", container_name, tool["image"], *tool["extra_args"]]
    subprocess.run(cmd, check=True, capture_output=True)


def stop_container(container_name):
    subprocess.run(["docker", "stop", container_name], capture_output=True)


def run_tool(tool, pool, lookup, sizes, repeats, seed):
    container_name = f"cmp_{tool['name'].lower().replace('-', '_')}"
    print(f"\n>>> starting {tool['name']} ({tool['image']})")
    start_container(tool, container_name)

    channel = grpc.insecure_channel(f"localhost:{PORT}")
    try:
        grpc.channel_ready_future(channel).result(timeout=180)
    except Exception as e:
        print(f"    FAILED to become ready: {e}")
        stop_container(container_name)
        return []

    name_call, init_call, prioritize_call, select_call = make_calls(channel)
    try:
        tool_name = name_call(pb2.Empty()).name
    except Exception:
        tool_name = tool["name"]
    print(f"    connected as '{tool_name}'")

    metric_eval = MetricEvaluator()
    rng = random.Random(seed)
    rows = []
    trial_id = 0
    for size in sizes:
        for rep in range(repeats):
            trial_id += 1
            init_subj = sample_subject(rng, pool, size, subject_id=trial_id * 2)
            eval_subj = sample_subject(rng, pool, size, subject_id=trial_id * 2 + 1)
            row = {"tool": tool["name"], "subject_size": size, "trial": rep, "success": False}
            try:
                init_resp = init_call(oracle_iterator(init_subj, lookup))
                if not init_resp.ok:
                    raise errors.InitializationError("ok=False")

                t0 = time.time()
                if tool["interface"] == "prioritize":
                    reply_iter = prioritize_call(testcase_iterator(eval_subj, lookup))
                    ordering = [r.testId for r in reply_iter]
                else:
                    reply_iter = select_call(testcase_iterator(eval_subj, lookup))
                    selected = [r.testId for r in reply_iter]
                    all_ids = [t.object_id for t in eval_subj.pg_test_data]
                    remaining = [i for i in all_ids if i not in set(selected)]
                    rng.shuffle(remaining)
                    ordering = selected + remaining
                row["t_prioritize_tests"] = time.time() - t0

                metric_eval.check_prioritization_validity(eval_subj, ordering)
                row["apfd"] = metric_eval.compute_apfd(eval_subj, ordering)
                row["apfdc"] = metric_eval.compute_apdfc(eval_subj, ordering)
                row["t_first_fault"] = metric_eval.compute_time_to_first_fault(eval_subj, ordering)
                row["t_last_fault"] = metric_eval.compute_time_to_last_fault(eval_subj, ordering)
                row["success"] = True
            except (errors.InitializationError, errors.PrioritizationError, grpc.RpcError) as e:
                print(f"    size={size} rep={rep}: FAILED ({e})")
            rows.append(row)
        print(f"    size={size}: {sum(1 for r in rows if r['subject_size']==size and r['success'])}/{repeats} ok")

    stop_container(container_name)
    return rows


def summarize(tool_name, rows):
    successes = [r for r in rows if r["success"]]
    print(f"\n=== {tool_name}: {len(successes)}/{len(rows)} successful ({100*len(successes)/max(len(rows),1):.1f}%) ===")
    for metric in ["apfd", "apfdc", "t_prioritize_tests", "t_first_fault", "t_last_fault"]:
        vals = [r[metric] for r in successes if r.get(metric) is not None]
        if vals:
            print(f"  {metric}: mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f} "
                  f"min={min(vals):.4f} max={max(vals):.4f}")


def main(args):
    pool = load_pool(args.data)
    lookup = {t["test_id"]: t["road_points"] for t in pool}
    sizes = [int(s) for s in args.sizes.split(",")]

    tool_names = [t.strip() for t in args.tools.split(",")] if args.tools else [t["name"] for t in TOOLS]
    all_rows = []
    for tool in TOOLS:
        if tool["name"] not in tool_names:
            continue
        rows = run_tool(tool, pool, lookup, sizes, args.repeats, args.seed)
        summarize(tool["name"], rows)
        all_rows.extend(rows)

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "tool", "subject_size", "trial", "success",
            "t_prioritize_tests", "apfd", "apfdc", "t_first_fault", "t_last_fault",
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAll results written to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sizes", default="10,20,30,50,80")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tools", default=None, help="comma-separated subset of tool names; default = all")
    parser.add_argument("--out", default="multi_tool_results.csv")
    main(parser.parse_args())
