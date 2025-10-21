#!/usr/bin/env python3
"""Replay a single Pollux optimize() from a hard-coded log snippet.

Instructions:
- Paste the four "Pollux optimize inputs" lines into RAW_LOG below.
- Run this script: python3 test-pollux-optimize.py
"""

import ast
import sys

from sched.adaptdl_sched.policy.pollux import PolluxPolicy
from sched.adaptdl_sched.policy.utils import JobInfo, NodeInfo


# Paste the four lines exactly as logged. Example format:
# INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | jobs={...}
# INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | nodes={...}
# INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | base_allocations={...}
# INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | node_template={...}
RAW_LOG = """
INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | jobs={"('adaptdl', 'cifar10-111')": {'resources': {'pods': 1, 'cpu': 10000, 'memory': 40000000000, 'nvidia.com/gpu': 1}, 'creation_timestamp': '2025-10-20T20:49:32+00:00', 'min_replicas': 0, 'max_replicas': 16, 'preemptible': True, 'epoch': 15, 'application': 'cifar10', 'speedup_fn': 'SpeedupFunction', 'num_restarts': 3, 'age': 532.065504}}
INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | nodes={'ip-192-168-64-216.ec2.internal': {'resources': {'cpu': 47359, 'ephemeral-storage': 76224326324, 'memory': 196368515072, 'nvidia.com/gpu': 4, 'pods': 222}, 'preemptible': False}, 'ip-192-168-90-191.ec2.internal': {'resources': {'cpu': 47459, 'ephemeral-storage': 76224326324, 'memory': 196997660672, 'nvidia.com/gpu': 4, 'pods': 225}, 'preemptible': False}}
INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | base_allocations={"('adaptdl', 'cifar10-111')": ['ip-192-168-64-216.ec2.internal', 'ip-192-168-64-216.ec2.internal', 'ip-192-168-64-216.ec2.internal', 'ip-192-168-64-216.ec2.internal']}
INFO:adaptdl_sched.policy.pollux:Pollux optimize inputs | node_template={'resources': {'cpu': 47459, 'ephemeral-storage': 76224326324, 'memory': 196997660672, 'nvidia.com/gpu': 4, 'pods': 225}, 'preemptible': True}
"""


def parse_inputs(log_text):
    sections = {}
    for line in log_text.splitlines():
        if "Pollux optimize inputs |" not in line:
            continue
        payload = line.split("|", 1)[1]
        key, value = payload.split("=", 1)
        sections[key.strip()] = ast.literal_eval(value.strip())
    # minimal assertions to ensure we have all parts
    assert "jobs" in sections and "nodes" in sections and \
           "base_allocations" in sections and "node_template" in sections
    return (
        sections["jobs"],
        sections["nodes"],
        sections["base_allocations"],
        sections["node_template"],
    )


def build_job_infos(jobs_dict):
    jobs = {}
    for key, info in jobs_dict.items():
        job = JobInfo(
            resources=info["resources"],
            # Dummy speedup; actual goodput comes from profiled data in Pollux when enabled.
            speedup_fn=(lambda _nodes, replicas: float(replicas or 0)),
            creation_timestamp=info.get("creation_timestamp", ""),
            min_replicas=info.get("min_replicas", 0),
            max_replicas=info.get("max_replicas", 1),
            preemptible=info.get("preemptible", True),
            num_restarts=info.get("num_restarts", 0),
            age=info.get("age", 0.0),
        )
        job.application = info.get("application")
        job.epoch = info.get("epoch")
        jobs[key] = job
    return jobs


def build_node_infos(nodes_dict):
    nodes = {}
    for key, info in nodes_dict.items():
        nodes[key] = NodeInfo(
            resources=info["resources"],
            preemptible=info.get("preemptible", True),
        )
    return nodes


if __name__ == "__main__":
    if RAW_LOG.strip() == "":
        print("Paste the four Pollux input lines into RAW_LOG and rerun.")
        sys.exit(0)

    jobs_raw, nodes_raw, base_allocs_raw, node_template_raw = parse_inputs(RAW_LOG)

    jobs = build_job_infos(jobs_raw)
    nodes = build_node_infos(nodes_raw)
    base_allocations = {k: list(v) for k, v in base_allocs_raw.items()}
    node_template = NodeInfo(
        resources=node_template_raw["resources"],
        preemptible=node_template_raw.get("preemptible", True),
    )

    policy = PolluxPolicy()
    allocations, desired_nodes = policy.optimize(jobs, nodes, base_allocations, node_template)

    print("Allocations:")
    for job_key, placement in allocations.items():
        print(job_key, placement)
    print("Desired nodes:", desired_nodes)

