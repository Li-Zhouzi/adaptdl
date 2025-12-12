#!/usr/bin/env python
"""
Scale-down event watcher for tracking the timeline of node scale-down operations.

Watches for:
- T3: When placeholder pods are actually deleted
- T4: When autoscaler marks nodes for deletion
- T5: When nodes become NotReady or deleted
"""

import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
from datetime import datetime

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NAMESPACE = 'adaptdl'
PLACEHOLDER_LABEL = 'adaptdl/placeholder'


async def watch_pod_deletions():
    """Watch for placeholder pod deletion events (T3)."""
    v1 = kubernetes.client.CoreV1Api()
    w = kubernetes.watch.Watch()

    LOG.info("Starting pod deletion watcher...")
    async for event in w.stream(
        v1.list_namespaced_pod,
        namespace=NAMESPACE,
        label_selector=f"{PLACEHOLDER_LABEL}=true"
    ):
        if event['type'] == 'DELETED':
            pod = event['object']
            timestamp = time.time()
            LOG.info(
                "TIMESTAMP-3-POD-DELETED: time=%.6f, timestamp=%s, pod=%s, node=%s",
                timestamp,
                datetime.fromtimestamp(timestamp).isoformat(),
                pod.metadata.name,
                pod.spec.node_name or 'unscheduled'
            )


async def watch_node_taints():
    """Watch for node taint changes (T4)."""
    v1 = kubernetes.client.CoreV1Api()
    w = kubernetes.watch.Watch()

    LOG.info("Starting node taint watcher...")
    async for event in w.stream(v1.list_node):
        if event['type'] in ['ADDED', 'MODIFIED']:
            node = event['object']
            if node.spec.taints:
                for taint in node.spec.taints:
                    if ('ToBeDeletedByClusterAutoscaler' in taint.key or
                        'DeletionCandidateOfClusterAutoscaler' in taint.key):
                        timestamp = time.time()
                        LOG.info(
                            "TIMESTAMP-4-AUTOSCALER-MARKED-FOR-DELETION: "
                            "time=%.6f, timestamp=%s, node=%s, taint=%s",
                            timestamp,
                            datetime.fromtimestamp(timestamp).isoformat(),
                            node.metadata.name,
                            taint.key
                        )


async def watch_node_status():
    """Watch for node status changes (T5)."""
    v1 = kubernetes.client.CoreV1Api()
    w = kubernetes.watch.Watch()

    # Track node ready states
    node_states = {}

    # Get initial state
    nodes = await v1.list_node()
    for node in nodes.items:
        ready = False
        if node.status.conditions:
            for condition in node.status.conditions:
                if condition.type == 'Ready':
                    ready = (condition.status == 'True')
                    break
        node_states[node.metadata.name] = ready

    LOG.info("Starting node status watcher...")
    async for event in w.stream(v1.list_node):
        node = event['object']
        node_name = node.metadata.name

        if event['type'] in ['ADDED', 'MODIFIED']:
            ready = False
            if node.status.conditions:
                for condition in node.status.conditions:
                    if condition.type == 'Ready':
                        ready = (condition.status == 'True')
                        break

            # Check for Ready -> NotReady transition
            if node_name in node_states and node_states[node_name] and not ready:
                timestamp = time.time()
                LOG.info(
                    "TIMESTAMP-5-NODE-NOT-READY: time=%.6f, timestamp=%s, node=%s",
                    timestamp,
                    datetime.fromtimestamp(timestamp).isoformat(),
                    node_name
                )

            node_states[node_name] = ready

        elif event['type'] == 'DELETED':
            timestamp = time.time()
            LOG.info(
                "TIMESTAMP-5B-NODE-DELETED: time=%.6f, timestamp=%s, node=%s",
                timestamp,
                datetime.fromtimestamp(timestamp).isoformat(),
                node_name
            )


async def run():
    """Run all watchers."""
    LOG.info("Starting scaledown watcher on namespace: %s", NAMESPACE)
    await asyncio.gather(
        watch_pod_deletions(),
        watch_node_taints(),
        watch_node_status(),
        return_exceptions=True
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(kubernetes.config.load_kube_config())
    loop.run_until_complete(run())
