# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kubernetes_asyncio as kubernetes
from aiohttp import web
import logging
import time
from adaptdl.sched_hints import SCHED_HINTS
from adaptdl_sched.config import get_supervisor_port
from datetime import datetime


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Supervisor:
    """
    Supervisor provides a simple REST interface for several functionalities.
    Currently, it has two endpoints:
    1.  /hints for jobs to send scheduling hints.
    2.  /discover for finding the pod IPs of a job.
    """

    def __init__(self, port, host='0.0.0.0'):
        self._host = host
        self._port = port
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_discover(self, request):
        # Long-polling endpoint used for discovering pod IPs for a given job.
        namespace = request.match_info["namespace"]
        name = request.match_info["name"]
        group = request.match_info["group"]
        timeout = int(request.query.get("timeout", "30"))
        start_time = time.time()

        LOG.info("[DISCOVER_START] job=%s group=%s namespace=%s ts=%.3f",
                 name, group, namespace, start_time)

        # Fetch job allocation for node verification
        job_allocation = None
        try:
            job = await self._objs_api.get_namespaced_custom_object(
                "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name)
            job_allocation = job.get("status", {}).get("allocation", [])
            LOG.info("[DISCOVER_ALLOC] job=%s group=%s allocation=%s ts=%.3f",
                     name, group, job_allocation, time.time())
        except Exception as e:
            LOG.warning("[DISCOVER_ALLOC_FAILED] job=%s group=%s error=%s ts=%.3f",
                       name, group, str(e), time.time())
            # Continue without allocation check if fetch fails

        pod_ip_list = None
        async with kubernetes.watch.Watch() as w:
            stream = w.stream(self._core_api.list_namespaced_pod, namespace,
                              label_selector="adaptdl/job={}".format(name),
                              field_selector="status.podIP!=",
                              timeout_seconds=timeout)
            async for event in stream:
                pod = event["object"]
                event_type = event.get("type", "UNKNOWN")
                pod_name = pod.metadata.name
                pod_group = pod.metadata.annotations.get("adaptdl/group", "unknown")
                replicas = int(pod.metadata.annotations["adaptdl/replicas"])
                rank = int(pod.metadata.annotations["adaptdl/rank"])
                node_name = pod.spec.node_name if pod.spec else "unknown"
                pod_ip = pod.status.pod_ip
                del_ts = pod.metadata.deletion_timestamp

                LOG.info("[DISCOVER_EVENT] type=%s pod=%s group=%s rank=%s/%s node=%s ip=%s del_ts=%s ts=%.3f",
                         event_type, pod_name, pod_group, rank, replicas,
                         node_name, pod_ip, del_ts, time.time())

                # Defensive check 1: Skip terminating pods
                if del_ts is not None:
                    LOG.warning("[DISCOVER_SKIP] pod=%s reason=deletion_timestamp_set group=%s rank=%s node=%s ip=%s ts=%.3f",
                               pod_name, pod_group, rank, node_name, pod_ip, time.time())
                    continue

                # Defensive check 2: Skip DELETED events
                if event_type == "DELETED":
                    LOG.warning("[DISCOVER_SKIP] pod=%s reason=event_type_deleted group=%s rank=%s node=%s ip=%s ts=%.3f",
                               pod_name, pod_group, rank, node_name, pod_ip, time.time())
                    continue

                # Only process pods from the requested group
                if pod_group != group:
                    continue

                # Defensive check 3: Verify node matches allocation (if available)
                if job_allocation and len(job_allocation) > rank:
                    expected_node = job_allocation[rank]
                    # Extract hostname from node name (e.g., "ip-192-168-66-70.ec2.internal")
                    # and compare with allocation entry
                    if expected_node not in node_name:
                        LOG.warning("[DISCOVER_SKIP] pod=%s reason=node_mismatch group=%s rank=%s expected_node=%s actual_node=%s ip=%s ts=%.3f",
                                   pod_name, pod_group, rank, expected_node, node_name, pod_ip, time.time())
                        continue

                # Initialize pod_ip_list if this is the first valid pod
                if pod_ip_list is None:
                    pod_ip_list = [None] * replicas

                # Defensive check 4: Prevent overwriting already-filled slots
                if pod_ip_list[rank] is not None:
                    LOG.warning("[DISCOVER_SKIP] pod=%s reason=rank_already_filled group=%s rank=%s node=%s ip=%s current_ip=%s ts=%.3f",
                               pod_name, pod_group, rank, node_name, pod_ip, pod_ip_list[rank], time.time())
                    continue

                # Accept this pod
                pod_ip_list[rank] = pod_ip
                LOG.info("[DISCOVER_ACCEPT] pod=%s group=%s rank=%s node=%s ip=%s current_list=%s ts=%.3f",
                         pod_name, pod_group, rank, node_name, pod_ip, pod_ip_list, time.time())

                # Check if we have all IPs
                if all(pod_ip is not None for pod_ip in pod_ip_list):
                    duration = time.time() - start_time
                    LOG.info("[DISCOVER_SUCCESS] job=%s group=%s ips=%s duration=%.3f ts=%.3f",
                             name, group, pod_ip_list, duration, time.time())
                    return web.json_response(pod_ip_list)

        # Timeout
        LOG.warning("[DISCOVER_TIMEOUT] job=%s group=%s partial_ips=%s ts=%.3f",
                   name, group, pod_ip_list, time.time())
        return web.json_response(status=408)  # Timeout.

    async def _handle_report(self, request):
        namespace = request.match_info['namespace']
        name = request.match_info['name']
        hints = await request.json()
        # Drop all unrecognized fields. TODO: validate each client-sent field.
        hints = {k: hints[k] for k in SCHED_HINTS if k in hints}
        # Patch only the train field to avoid conflicts with controller.
        patch = {"status": {"train": hints}}
        LOG.info("Patch AdaptDLJob %s/%s at %s: %s", namespace, name, datetime.now(), patch)
        await self._objs_api.patch_namespaced_custom_object_status(
            "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name, patch)
        return web.Response()

    def run(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.get('/discover/{namespace}/{name}/{group}',
                    self._handle_discover),
            web.put('/hints/{namespace}/{name}', self._handle_report),
        ])
        LOG.info("%s %s", self._host, self._port)
        web.run_app(self.app, host=self._host, port=self._port)


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    supervisor = Supervisor(int(get_supervisor_port()))
    supervisor.run()
