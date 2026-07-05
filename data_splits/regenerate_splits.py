"""
Regenerates the exact train/held-out split used in the paper from the
public SensoDat corpus, using the fixed test-ID lists in this folder.

Usage:
    1. Obtain the SensoDat dataset (Birchler et al., MSR 2024) raw
       OpenDRIVE (.xodr) files -- see the paper's Data Availability
       statement for the public source.
    2. python ../harness/build_real_pool.py --sensodat-dir <path_to_sensodat> \
           --out full_pool.json
    3. python regenerate_splits.py --pool full_pool.json
"""
import argparse
import json


def main(args):
    pool = json.load(open(args.pool))
    by_id = {t["_id"]["$oid"]: t for t in pool}

    for name in ["train_test_ids", "heldout_test_ids"]:
        ids = json.load(open(f"{name}.json"))
        missing = [i for i in ids if i not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} ids from {name}.json not found in the supplied pool "
                  f"(expected if using a partial SensoDat download)")
        subset = [by_id[i] for i in ids if i in by_id]
        out_name = name.replace("_test_ids", "_split") + ".json"
        with open(out_name, "w") as f:
            json.dump(subset, f)
        print(f"wrote {out_name}: {len(subset)}/{len(ids)} tests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, help="output of build_real_pool.py over the full SensoDat corpus")
    main(parser.parse_args())
