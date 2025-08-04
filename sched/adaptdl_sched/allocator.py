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


import asyncio
import dateutil.parser
import kubernetes_asyncio as kubernetes
import logging
import time
import os
import aiohttp

from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl.sched_hints import PERF_PARAMS
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.dummy import DummyPolicy
from adaptdl_sched.policy.fixed_width import FixedWidthPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
from adaptdl_sched.resources import (get_node_unrequested, get_pod_requests,
                                     set_default_resources)
from adaptdl_sched.utils import patch_job_status
from adaptdl_sched.cluster_expander import ClusterExpander
from adaptdl_sched.config import allowed_taints
from adaptdl_sched.policy.fixed_width import APPLICATION_NAMES

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class AdaptDLAllocator(object):
    def __init__(self, expander):
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1",
                                 "", "adaptdljobs")
        self._cluster_expander = expander

        # Select the policy to use
        # Options: "pollux", "dummy", "fixed-width"
        SELECTED_POLICY = "fixed-width"  # <--- CHANGE THIS VALUE TO SWITCH POLICY

        # Width fetching configuration
        self._width_service_url = os.environ.get("WIDTH_SERVICE_URL", "http://localhost:8083")
        self._width_fetch_interval = int(os.environ.get("WIDTH_FETCH_INTERVAL", "60")) 
        self._current_width = None

        if SELECTED_POLICY == "pollux":
            self._policy = PolluxPolicy()
            self._policy_type = "pollux"
        elif SELECTED_POLICY == "dummy":
            self._policy = DummyPolicy(num_gpus_per_job=8) # Configure dummy as needed
            self._policy_type = "dummy"
        elif SELECTED_POLICY == "fixed-width":
            # Initialize with None width, will be fetched later
            self._policy = None
            self._policy_type = "fixed-width"
        else:
            raise ValueError(f"Unknown policy: {SELECTED_POLICY}")

        # lock for the two corountines in run()
        self._lock = asyncio.Lock()

    async def run(self):
        # three functionality: (1) watch for new job and start if possible.
        # (2) periodically optimize existing jobs
        # (3) periodically fetch width for fixed-width policy
        if self._policy_type == "fixed-width":
            await asyncio.gather(
                self._allocate_one_loop(),
                self._optimize_all_loop(),
                self._fetch_width_loop()
            )
        else:
            await asyncio.gather(
                self._allocate_one_loop(),
                self._optimize_all_loop()
            )

    async def _allocate_one_loop(self):
        async with kubernetes.watch.Watch() as watch:
            while True:
                async for event in watch.stream(
                        self._objs_api.list_namespaced_custom_object,
                        *self._custom_resource, timeout_seconds=60):
                    # We only consider newly-added preemptible jobs
                    # because this allocation may not be final.
                    if (event["type"] == "ADDED" and
                            event["object"]["spec"].get("preemptible", True)):
                        async with self._lock:
                            await self._allocate_one(event)

    async def _allocate_one(self, event):
        # re-read the job , compared with the previous readjob
        job = event["object"]
        namespace = job["metadata"]["namespace"]
        name = job["metadata"]["name"]

        try:
            job = await self._objs_api.get_namespaced_custom_object(
                "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                return
            raise  # unexpected

        # some other coroutine has handled this
        if job.get("status", {}).get("allocation") is not None or\
                job.get("status", {}).get("group") is not None:
            return

        namespace = job["metadata"]["namespace"]
        name = job["metadata"]["name"]
        # if this is a restarted job, skip i
        LOG.info("detected an added job %s/%s.",
                 namespace, name)
        # parse the job infomation
        job_info = self._get_job_info(job)

        # find available nodes.
        node_infos, _ = await self._find_nodes()

        # get the node to allocate
        new_allocation = self._get_policy().allocate_job(
                            job_info, node_infos)
        patch = {"status": {"allocation": new_allocation}}
        LOG.info("Patch AdaptdlJob %s/%s: %s ",
                 namespace, name, patch)
        await patch_job_status(self._objs_api, namespace, name, patch)

    async def _optimize_all_loop(self):
        while True:
            # try to gain lock
            async with self._lock:
                LOG.info(f"[TIMESTAMP: {time.time()}] Starting allocator optimization cycle")
                await self._optimize_all()

            LOG.info("Sleep for 60 seconds")
            await asyncio.sleep(60)

    async def _fetch_width_loop(self):
        """Periodically fetch width from the width service."""
        while True:
            try:
                LOG.info(f"[TIMESTAMP: {time.time()}] Starting width fetch from calculator service")
                await self._fetch_width()
            except Exception as e:
                LOG.error(f"Error fetching width: {e}")
            
            LOG.info(f"Sleep for {self._width_fetch_interval} seconds before next width fetch")
            await asyncio.sleep(self._width_fetch_interval)

    async def _fetch_width(self):
        """Fetch width from the width calculator web service and update policy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._width_service_url}/width") as response:
                    if response.status == 200:
                        data = await response.json()
                        width = data.get("width")
                        
                        # Check if width is valid (not None and no None values in dictionaries)
                        if width is not None and self._is_valid_width(width):
                            LOG.info(f"Fetched new width: {width}")
                            self._current_width = width
                            self._policy = FixedWidthPolicy(width)
                        else:
                            if width is None:
                                LOG.warning("Width calculator returned None width, keeping current width")
                            else:
                                LOG.warning(f"Width calculator returned invalid width (contains None): {width}, keeping current width")
                            # Keep current width/policy if we have one
                            if self._current_width is not None:
                                LOG.info(f"Keeping current width: {self._current_width}")
                            else:
                                LOG.warning("No current width available, using Pollux policy as fallback")
                                self._policy = PolluxPolicy()
                    else:
                        LOG.warning(f"Width calculator returned status {response.status}, keeping current width")
                        if self._current_width is None:
                            LOG.warning("No current width available, using Pollux policy as fallback")
                            self._policy = PolluxPolicy()
        except Exception as e:
            LOG.warning(f"Failed to fetch width from calculator: {e}, keeping current width")
            if self._current_width is None:
                LOG.warning("No current width available, using Pollux policy as fallback")
                self._policy = PolluxPolicy()
        
        # Always print current width after fetch attempt
        LOG.info(f"Current width after fetch: {self._current_width}")
    
    def _is_valid_width(self, width):
        """Check if width dictionary is valid (no None values)."""
        if not isinstance(width, dict):
            return False
        
        for app, epochs in width.items():
            if epochs is None or not isinstance(epochs, dict):
                return False
            for epoch, value in epochs.items():
                if value is None:
                    return False
        return True

    def _get_policy(self):
        """Get the current policy, ensuring it's initialized for fixed-width."""
        if self._policy_type == "fixed-width" and self._policy is None:
            # If we haven't fetched width yet, use Pollux as fallback
            LOG.warning("Fixed-width policy not initialized yet, using Pollux policy as fallback")
            return PolluxPolicy()
        return self._policy

    async def _optimize_all(self):
        LOG.info("Running allocator loop")
        nodes, node_template = await self._find_nodes(
                                   pod_label_selector="!adaptdl/job"
                               )
        LOG.info("Node resources: %s",
                 {k: v.resources for k, v in nodes.items()})
        jobs, prev_allocations = \
            await self._find_jobs_and_allocations()
        LOG.info("Job resources: %s",
                 {k: v.resources for k, v in jobs.items()})
        start = time.time()
        allocations = self._allocate(jobs, nodes, prev_allocations,
                                     node_template)
        duration = time.time() - start
        LOG.info("Allocations (in %.3f sec): %s", duration,
                 allocations)
        await self._update_allocations(allocations)

    async def _update_allocations(self, allocations):
        job_list = await self._objs_api.list_namespaced_custom_object(
            "adaptdl.petuum.com", "v1", "", "adaptdljobs")
        for job in job_list["items"]:
            namespace = job["metadata"]["namespace"]
            name = job["metadata"]["name"]
            job_allocation = job.get("status", {}).get("allocation", [])
            new_allocation = list(allocations.get((namespace, name), []))
            if list(job_allocation) != new_allocation:
                patch = {"status": {"allocation": new_allocation}}
                LOG.info("Patch AdaptDLJob %s/%s: %s", namespace, name, patch)
                await patch_job_status(self._objs_api, namespace, name, patch)

    async def _find_nodes(self, pod_label_selector=""):
        # NOTE: Default empty string "" label_selector qualifies all pods
        node_infos = {}
        node_list = await self._core_api.list_node()
        # Find all pods qualified by the pod_label_selector which are taking up
        # resources and subtract those resources from the available pool.
        # Apparently there's not a more efficient way to get currently
        # available resources in k8s?. We also check if we have reached the pod
        # limit on the node. This number denotes (allocatable pods -
        # Non-terminated pods) on that node.
        pod_list = await self._core_api.list_pod_for_all_namespaces(
                                label_selector=pod_label_selector)
        for node in node_list.items:
            if allowed_taints(node.spec.taints):
                resources = get_node_unrequested(node, pod_list.items)
                if not resources.get("pods"):
                    LOG.warning(f"node {node.metadata.name} "
                                "has no free pods available.")
                node_infos[node.metadata.name] = NodeInfo(resources, False)
        # For cluster autoscaling: to determine if additional nodes would be
        # helpful, add a few "virtual" nodes which only become available in
        # "eta" seconds. Currently, we only consider as many virtual nodes as
        # there are real nodes. We infer each resource to be the maximum amount
        # observed in any real node.
        max_resources = {}
        for node_name in node_infos:
            for key, val in node_infos[node_name].resources.items():
                if key not in max_resources or val > max_resources[key]:
                    max_resources[key] = val
        node_template = NodeInfo(max_resources, True)
        return node_infos, node_template

    def _get_job_info(self, job):
        job["spec"]["template"]["spec"] = \
            set_default_resources(job["spec"]["template"]["spec"])
        resources = get_pod_requests(job["spec"]["template"]["spec"])
        hints = job.get("status", {}).get("train", {})
        max_replicas = max(2 * hints.get("maxProfiledReplicas", 0), 1)
        if job["spec"].get("maxReplicas"):
            max_replicas = min(max_replicas, job["spec"]["maxReplicas"])
        min_replicas = job["spec"].get("minReplicas", 0)
        # max_replicas should be greater or equal to min_replicas
        max_replicas = max(max_replicas, min_replicas)
        preemptible = job["spec"].get("preemptible", True)
        if {"perfParams", "initBatchSize"} <= hints.keys() and preemptible:
            max_batch_size = (hints.get("maxBatchSize")
                              or hints["initBatchSize"])
            if hints.get("localBszBounds"):
                min_local_bsz = hints["localBszBounds"][0] or 1
                # Make sure max_batch_size / replicas >= min_local_bsz
                if max_batch_size < min_local_bsz * max_replicas:
                    max_replicas = int(max_batch_size / min_local_bsz)
            perf_params = PerfParams(*[hints["perfParams"][k]
                                       for k in PERF_PARAMS.keys()])
            if "gradParams" in hints:
                grad_params = GradParams(hints["gradParams"]["norm"],
                                         hints["gradParams"]["var"])
            else:
                grad_params = GradParams(0.0, 1.0)
            goodput_fn = GoodputFunction(perf_params, grad_params,
                                         hints["initBatchSize"])
            speedup_fn = SpeedupFunction(
                goodput_fn,
                hints.get("maxBatchSize"),
                hints.get("localBszBounds"),
                hints.get("gradientAccumulation", False))
        else:
            speedup_fn = lambda n, r: r  # noqa: E731
        creation_ts = dateutil.parser.isoparse(
                job["metadata"]["creationTimestamp"])
        
        job_name = job["metadata"]["name"]
        job_application = job_name.split("-")[0]
        if job_application not in APPLICATION_NAMES:
            raise ValueError(f"Unknown application: {job_application}")
        job_epoch = hints.get("epoch", 0)
        # if job_epoch is None:
        #     raise ValueError(f"Epoch is not set for job: {job_name}")
        job_info = JobInfo(
                resources, speedup_fn, creation_ts, min_replicas,
                max_replicas, preemptible)
        job_info.epoch = job_epoch
        job_info.application = job_application
        return job_info

    async def _find_jobs_and_allocations(self):
        job_list = await self._objs_api.list_namespaced_custom_object(
            "adaptdl.petuum.com", "v1", "", "adaptdljobs")
        job_infos = {}
        allocations = {}

        for job in job_list["items"]:
            if job.get("status", {}).get("phase") \
                    not in ["Pending", "Running", "Starting", "Stopping"]:
                continue

            namespace = job["metadata"]["namespace"]
            name = job["metadata"]["name"]

            if "allocation" in job.get("status", {}):
                allocations[namespace, name] = \
                    list(job["status"]["allocation"])

            job_info = self._get_job_info(job)
            job_infos[(namespace, name)] = job_info

        return job_infos, allocations

    def _allocate(self, jobs, nodes, prev_allocations, node_template):
        for job_key in list(jobs):
            job_resources = jobs[job_key].resources
            for node in nodes.values():
                if all(val <= node.resources.get(key, 0)
                       for key, val in job_resources.items()):
                    # Found a node which can fit a replica of this job.
                    break
            else:
                # No node can fit a replica of this job.
                # TODO: propagate this to the controller so the job is Failed.
                LOG.warning("Job %s cannot be scheduled!", job_key)
                jobs.pop(job_key)
        allocations = {}
        if not jobs:
            # There are no jobs, let the expander shrink the cluster.
            self._cluster_expander.fit([])
        elif jobs and nodes:
            allocations, desired_nodes = self._get_policy().optimize(
                jobs, nodes, prev_allocations, node_template)
            if desired_nodes < len(nodes):
                active_nodes = list(set.union(*map(set, allocations.values())))
            else:
                active_nodes = list(nodes)
                while len(active_nodes) < desired_nodes:
                    active_nodes.append(f"~{desired_nodes-len(active_nodes)}")
            self._cluster_expander.fit(active_nodes)
            LOG.info("Active nodes: %s", active_nodes)
        elif jobs:
            # Expand job ASG from zero nodes.
            # Assumption is AdaptDL is running on a different ASG
            self._cluster_expander.fit(['~1'])
        return allocations


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    expander = ClusterExpander()
    allocator = AdaptDLAllocator(expander)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        expander.run(),
        allocator.run(),
    ))
    loop.close()
