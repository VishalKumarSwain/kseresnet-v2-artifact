"""
Parse the real local SensoDat .xodr files into the same flat JSON schema as
data/sdc-test-data.json, so local_compare.py can be pointed at real data
without any changes. Deduplicates by test_id to guard against the
mt_set_11_freneticV folder-naming duplicate found during inspection.
"""
import argparse
import glob
import json
import time
import xml.etree.ElementTree as ET


def parse_xodr(path):
    root = ET.parse(path).getroot()
    info = root.find(".//sdc_test_info")
    if info is None:
        return None
    test_id = info.get("test_id")
    outcome = info.get("test_outcome")
    duration = info.get("test_duration")
    if test_id is None or outcome not in ("PASS", "FAIL") or duration is None:
        return None
    points = [
        {"x": float(g.get("x")), "y": float(g.get("y"))}
        for g in root.findall(".//road/planView/geometry")
    ]
    return {
        "_id": {"$oid": test_id},
        "meta_data": {"test_info": {"test_outcome": outcome, "test_duration": float(duration)}},
        "road_points": points,
    }


def main(args):
    files = glob.glob(f"{args.sensodat_dir}/**/*.xodr", recursive=True)
    print(f"found {len(files)} .xodr files")

    seen = {}
    t0 = time.time()
    errors = 0
    for i, f in enumerate(files):
        try:
            rec = parse_xodr(f)
        except ET.ParseError:
            rec = None
        if rec is None:
            errors += 1
            continue
        seen[rec["_id"]["$oid"]] = rec  # dedup by test_id, last write wins
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)} parsed, {len(seen)} unique so far, {time.time()-t0:.0f}s elapsed")

    print(f"done in {time.time()-t0:.0f}s. unique tests: {len(seen)}  unparsable: {errors}")
    fails = sum(1 for r in seen.values() if r["meta_data"]["test_info"]["test_outcome"] == "FAIL")
    print(f"FAIL: {fails}  PASS: {len(seen)-fails}")

    with open(args.out, "w") as fp:
        json.dump(list(seen.values()), fp)
    print(f"written to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensodat-dir", default="Sensodata")
    parser.add_argument("--out", default="sensodat_real_pool.json")
    main(parser.parse_args())
