#!/usr/bin/env python3

import argparse
import json
import time
from kubernetes import client, config, watch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="path to output file")
    args = parser.parse_args()

    config.load_kube_config()
    objs_api = client.CustomObjectsApi()
    # namespace = config.list_kube_config_contexts()[1]["context"].get("namespace", "default")
    # obj_args = ("esper.petuum.com", "v1", namespace, "esperjobs")
    namespace = "adaptdl"
    obj_args = ("adaptdl.petuum.com", "v1", namespace, "adaptdljobs")
    while True:
        obj_list = objs_api.list_namespaced_custom_object(*obj_args)
        record = {
            "timestamp": time.time(),
            "submitted_jobs": [],
        }
        for obj in obj_list["items"]:
            # Try to get globalBatchSize, fall back to initBatchSize
            train_status = obj.get("status", {}).get("train", {})
            batch_size_val = train_status.get("globalBatchSize", 0)

            record["submitted_jobs"].append({
                "name": obj["metadata"]["name"],
                "epoch": train_status.get("epoch", 0),
                "allocation": obj.get("status", {}).get("allocation", []),
                "batch_size": batch_size_val,
                "submission_time": obj["metadata"]["creationTimestamp"],
                "completion_time": obj.get("status", {}).get("completionTimestamp", None),
            })
        with open(args.output, "a") as f:
            json.dump(record, f)
            f.write("\n")
        time.sleep(30)
