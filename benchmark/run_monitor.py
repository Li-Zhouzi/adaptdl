#!/usr/bin/env python3

import argparse
import json
import time
from kubernetes import client, config, watch


def get_pod_status_summary(job_pods, allocation_count):
    """
    Generate a readable pod status summary.
    Returns "pod status normal" if all pods are running and ready, 
    otherwise returns detailed status.
    """
    if not job_pods:
        return "no pods found"
    
    # Check if all pods are running and ready
    all_running = True
    all_ready = True
    pod_count = len(job_pods)
    
    for pod_name, pod_info in job_pods.items():
        # Check if pod phase is running
        if pod_info["phase"] != "Running":
            all_running = False
            break
        
        # Check if all containers are ready
        for container in pod_info["container_statuses"]:
            if not container["ready"]:
                all_ready = False
                break
    
    # If all pods are running, ready, and count matches allocation
    if all_running and all_ready and pod_count == allocation_count:
        return "pod status normal"
    else:
        # Return detailed status
        status_details = []
        for pod_name, pod_info in job_pods.items():
            pod_detail = f"{pod_name}: {pod_info['phase']}"
            if pod_info["container_statuses"]:
                container_states = []
                for container in pod_info["container_statuses"]:
                    state = container["state"] or "unknown"
                    ready = "ready" if container["ready"] else "not ready"
                    container_states.append(f"{container['name']}({state},{ready})")
                pod_detail += f" [{', '.join(container_states)}]"
            status_details.append(pod_detail)
        return "; ".join(status_details)


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
    
    # Track jobs that have been logged as completed
    logged_completed_jobs = set()
    
    while True:
        obj_list = objs_api.list_namespaced_custom_object(*obj_args)
        # Get node information
        nodes = core_api.list_node().items
        total_everything = len(nodes)
        total_nodes = len(nodes)
        ready_nodes = sum(
            1 for n in nodes
            if any(c.type == "Ready" and c.status == "True" for c in n.status.conditions)
            and not n.spec.unschedulable  # Exclude nodes with scheduling disabled
        )
        
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
                "ready": ready_nodes,
                "total_including_terminating": total_everything
            }
        }
        for obj in obj_list["items"]:
            job_name = obj["metadata"]["name"]
            completion_time = obj.get("status", {}).get("completionTimestamp", None)
            
            # Skip jobs that have already been logged as completed
            if completion_time is not None and job_name in logged_completed_jobs:
                continue
            
            # If job is newly completed, mark it for logging once and add to logged set
            if completion_time is not None and job_name not in logged_completed_jobs:
                logged_completed_jobs.add(job_name)
            
            # Find pods associated with this job
            job_pods = {name: status for name, status in pod_status.items() 
                       if name.startswith(f"{job_name}-")}
            
            # Get allocation count for pod status comparison
            allocation = obj.get("status", {}).get("allocation", [])
            allocation_count = len(allocation) if allocation else 0
            
            # Get grad_params from job status
            grad_params = obj.get("status", {}).get("train", {}).get("gradParams", {})
            
            # Get progress from job status
            progress = obj.get("status", {}).get("train", {}).get("progress", None)
            
            record["submitted_jobs"].append({
                "name": job_name,
                "epoch": obj.get("status", {}).get("train", {}).get("epoch", 0),
                "allocation": allocation,
                "batch_size": obj.get("status", {}).get("train", {}).get("batchSize", 0),
                "submission_time": obj["metadata"]["creationTimestamp"],
                "completion_time": completion_time,
                "pod_status": get_pod_status_summary(job_pods, allocation_count),
                "grad_params": {
                    "norm": grad_params.get("norm", None),
                    "var": grad_params.get("var", None)
                } if grad_params else None,
                "progress": progress
            })
        with open(args.output, "a") as f:
            json.dump(record, f)
            f.write("\n")
        time.sleep(1)
