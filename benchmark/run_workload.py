#!/usr/bin/env python3

import argparse
import copy
import itertools
import os
import pandas
import subprocess
import time
import yaml
from kubernetes import client, config, watch


def build_images(models, repository):
    # repository is like "localhost:32000/adaptdl-submit"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    templates = {}

    # Get the NodePort of the adaptdl-registry service
    core_api = client.CoreV1Api()
    try:
        service = core_api.read_namespaced_service(name="adaptdl-registry", namespace="adaptdl")
        node_port = None
        for port in service.spec.ports:
            if port.port == 5000 or port.target_port == 5000 or port.name == "registry": # Match on service port 5000 or name 'registry'
                node_port = port.node_port
                break
        if node_port is None:
            raise RuntimeError("Could not find NodePort for adaptdl-registry service on port 5000/registry.")
        print(f"Found adaptdl-registry NodePort: {node_port}")
    except client.exceptions.ApiException as e:
        print(f"Error fetching adaptdl-registry service: {e}")
        raise

    for model in models:
        with open(os.path.join(models_dir, model, "adaptdljob.yaml")) as f:
            template = yaml.load(f, Loader=yaml.SafeLoader)
        dockerfile = os.path.join(models_dir, model, "Dockerfile")
        
        # image_for_push is like "localhost:32000/adaptdl-submit:cifar10"
        # This is used for local docker build and push via port-forward
        image_for_push = repository + ":" + model
        
        print(f"Building image {image_for_push}...")
        subprocess.check_call(["docker", "build", "-t", image_for_push, project_root, "-f", dockerfile])
        print(f"Pushing image {image_for_push}...")
        subprocess.check_call(["docker", "push", image_for_push])
        
        # repodigest_from_local_push is like "localhost:32000/adaptdl-submit@sha256:digest"
        repodigest_from_local_push = subprocess.check_output(
                ["docker", "image", "inspect", image_for_push, "--format={{index .RepoDigests 0}}"])
        repodigest_from_local_push = repodigest_from_local_push.decode().strip()

        # Construct the image name for Kubernetes using localhost:<NodePort>
        # Example: localhost:32000/adaptdl-submit@sha256:digest
        # We need to get the repo name part (e.g., "adaptdl-submit") and the digest part.
        
        repo_and_digest_part = repodigest_from_local_push.split('/', 1)[-1] # e.g., "adaptdl-submit@sha256:digest"
        
        # k8s_image_name is like "localhost:NODE_PORT/adaptdl-submit@sha256:digest"
        k8s_image_name = repodigest_from_local_push 
        
        print(f"Using image for Kubernetes PodSpec: {k8s_image_name}")
        template["spec"]["template"]["spec"]["containers"][0]["image"] = k8s_image_name
        templates[model] = template
    return templates


def cache_images(templates):
    # Cache job images on all nodes in the cluster.
    daemonset = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "images"},
        "spec": {
            "selector": {"matchLabels": {"name": "images"}},
            "template": {
                "metadata": {
                    "labels": {"name": "images"},
                    # Annotations from previous attempt, can be removed if not helping
                    # "annotations": {
                    #     "kubectl.kubernetes.io/default-container": "cifar10",
                    #     "alpha.image.policy.k8s.io/non-root": "true"
                    # }
                },
                "spec": {
                    "containers": [],
                    "imagePullSecrets": [{"name": "regcred"}], # Restoring this as adaptdl copy might rely on it or similar logic
                }
            }
        }
    }
    for name, template in templates.items():
        container = {
            "name": name,
            "image": template["spec"]["template"]["spec"]["containers"][0]["image"],
            "command": ["sleep", "1000000000"],
        }
        daemonset["spec"]["template"]["spec"]["containers"].append(container)
    apps_api = client.AppsV1Api()
    namespace = "adaptdl"
    
    try:
        apps_api.delete_namespaced_daemon_set("images", namespace, body=client.V1DeleteOptions())
        print("Deleted existing 'images' DaemonSet.")
        time.sleep(5) 
    except client.exceptions.ApiException as e:
        if e.status == 404:
            print("No existing 'images' DaemonSet to delete.")
        else:
            print(f"Error deleting existing DaemonSet (continuing): {e}") # Log and continue
            
    apps_api.create_namespaced_daemon_set(namespace, daemonset)
    print("Created 'images' DaemonSet for image caching.")
    
    while True:
        try:
            obj = apps_api.read_namespaced_daemon_set("images", namespace)
            ready = obj.status.number_ready if obj.status and obj.status.number_ready is not None else 0
            total = obj.status.desired_number_scheduled if obj.status and obj.status.desired_number_scheduled is not None else 0
            print("caching images on all nodes: {}/{}".format(ready, total))
            if total > 0 and ready >= total:
                print("Image caching DaemonSet is ready.")
                break
        except client.exceptions.ApiException as e:
            print(f"Error reading DaemonSet status: {e}. Retrying...")
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str, choices=["pollux", "optimus", "tiresias"])
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--repository", type=str, default="localhost:32000/adaptdl-submit")
    args = parser.parse_args()

    workload = pandas.read_csv(args.workload)

    config.load_kube_config()

    templates = build_images(["cifar10"], args.repository)
    
    cache_images(templates)

    objs_api = client.CustomObjectsApi()
    namespace = "adaptdl" 
    obj_args = ("adaptdl.petuum.com", "v1", namespace, "adaptdljobs")

    print("start workload")
    start = time.time()
    for row in workload.sort_values(by="time").itertuples():
        while time.time() - start < row.time:
            time.sleep(1)
        print("submit job {} at time {}".format(row, time.time() - start))
        job = copy.deepcopy(templates[row.application])
        job["metadata"].pop("generateName")
        job["metadata"]["name"] = row.name
        job["spec"].update({
            "application": row.application,
            "targetNumReplicas": row.num_replicas,
            "targetBatchSize": row.batch_size,
        })
        volumes = job["spec"]["template"]["spec"].setdefault("volumes", [])
        volumes.append({
            "name": "pollux",
            "persistentVolumeClaim": { "claimName": "pollux" },
        })
        mounts = job["spec"]["template"]["spec"]["containers"][0].setdefault("volumeMounts", [])
        mounts.append({
            "name": "pollux",
            "mountPath": "/pollux/checkpoint",
            "subPath": "pollux/checkpoint/" + row.name,
        })
        mounts.append({
            "name": "pollux",
            "mountPath": "/pollux/tensorboard",
            "subPath": "pollux/tensorboard/" + row.name,
        })
        env = job["spec"]["template"]["spec"]["containers"][0].setdefault("env", [])
        env.append({"name": "ADAPTDL_CHECKPOINT_PATH", "value": "/pollux/checkpoint"})
        env.append({"name": "ADAPTDL_TENSORBOARD_LOGDIR", "value": "/pollux/tensorboard"})
        env.append({"name": "APPLICATION", "value": row.application})
        if args.policy in ["tiresias"]:
            job["spec"]["minReplicas"] = job["spec"]["maxReplicas"] = row.num_replicas
            env.append({"name": "TARGET_NUM_REPLICAS", "value": str(row.num_replicas)})
        if args.policy in ["tiresias", "optimus"]:
            env.append({"name": "TARGET_BATCH_SIZE", "value": str(row.batch_size)})
        print(yaml.dump(job))
        objs_api.create_namespaced_custom_object(*obj_args, job)
