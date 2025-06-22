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
    core_api = client.CoreV1Api()
    # namespace = config.list_kube_config_contexts()[1]["context"].get("namespace", "default")
    # obj_args = ("esper.petuum.com", "v1", namespace, "esperjobs")
    namespace = "adaptdl"
    obj_args = ("adaptdl.petuum.com", "v1", namespace, "adaptdljobs")
    while True:
        obj_list = objs_api.list_namespaced_custom_object(*obj_args)
        # Get node information
        nodes = core_api.list_node()
        total_nodes = len(nodes.items)
        ready_nodes = sum(1 for node in nodes.items if any(condition.type == "Ready" and condition.status == "True" 
                          for condition in node.status.conditions))
        
        # Get pod information for all pods in the namespace
        pods = core_api.list_namespaced_pod(namespace)
        pod_status = {pod.metadata.name: {
            "phase": pod.status.phase,
            "container_statuses": [{
                "name": container.name,
                "state": (
                    "running" if container.state and container.state.running else
                    container.state.waiting.reason if container.state and container.state.waiting and container.state.waiting.reason else
                    "waiting" if container.state and container.state.waiting else
                    "terminated" if container.state and container.state.terminated else
                    None
                ),
                "ready": container.ready
            } for container in pod.status.container_statuses] if pod.status.container_statuses else []
        } for pod in pods.items}
        
        record = {
            "timestamp": time.time(),
            "submitted_jobs": [],
            "cluster_nodes": {
                "total": total_nodes,
                "ready": ready_nodes
            }
        }
        for obj in obj_list["items"]:
            job_name = obj["metadata"]["name"]
            # Find pods associated with this job
            job_pods = {name: status for name, status in pod_status.items() 
                       if name.startswith(f"{job_name}-")}
            
            record["submitted_jobs"].append({
                "name": job_name,
                "epoch": obj.get("status", {}).get("train", {}).get("epoch", 0),
                "allocation": obj.get("status", {}).get("allocation", []),
                "batch_size": obj.get("status", {}).get("train", {}).get("batchSize", 0),
                "submission_time": obj["metadata"]["creationTimestamp"],
                "completion_time": obj.get("status", {}).get("completionTimestamp", None),
                "pod_status": job_pods
            })
        with open(args.output, "a") as f:
            json.dump(record, f)
            f.write("\n")
        time.sleep(1)
