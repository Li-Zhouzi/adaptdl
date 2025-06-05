import logging
import math
from collections import OrderedDict

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class DummyPolicy(object):
    def __init__(self, num_gpus_per_job=1):
        self._num_gpus_per_job = num_gpus_per_job
        LOG.info(f"DummyPolicy initialized with {self._num_gpus_per_job} GPUs per job.")

    def allocate_job(self, job_info, nodes):
        """
        Copied from Pollux Policy.
        """
        job_resources = job_info.resources
        min_replicas = max(job_info.min_replicas, 1)
        node_list = []
        nodes = self._sort_nodes(nodes)
        for node_name, node in nodes.items():
            # number of replica fit in this node
            replica_this = min(node.resources.get(key, 0) // val
                               for key, val in job_resources.items())
            if replica_this >= min_replicas:
                node_list = [node_name] * min_replicas
                return node_list
        else:
            return []
    
    def _sort_nodes(self, nodes):
        return OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible,
                                                  kv[0])))

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        """
        Optimizes allocations for all jobs by assigning a fixed number of GPUs per job.
        First preserves existing allocations that already have the correct number of GPUs,
        then assigns remaining jobs if resources are available.
        """
        new_allocations = {}
        # Track available GPUs on each node
        available_gpus = {node_name: node.resources.get("nvidia.com/gpu", 0) 
                         for node_name, node in nodes.items()}
        
        # First pass: preserve existing allocations that already have the correct number of GPUs
        for job_key, prev_alloc in prev_allocations.items():
            if job_key not in jobs:
                continue  # Job no longer exists
                
            job_info = jobs[job_key]
            gpus_per_replica = job_info.resources.get("nvidia.com/gpu", 1)
            if gpus_per_replica == 0:
                # This job requests 0 GPUs per replica.
                raise ValueError(f"Job {job_key} requests 0 GPUs per replica.")
            assert gpus_per_replica == 1, f"Job {job_key} requests {gpus_per_replica} GPUs per replica, which is not 1."
                
            # Calculate total GPUs this job had in its previous allocation
            gpus_in_prev_alloc = len(prev_alloc) * gpus_per_replica
                
            if gpus_in_prev_alloc == self._num_gpus_per_job:
                # Check if this previous allocation can be preserved
                can_preserve = True
                # Count how many replicas were on each node in the previous allocation for this job
                replicas_on_nodes_map = {}
                for node_name_from_prev in prev_alloc:
                    replicas_on_nodes_map[node_name_from_prev] = \
                        replicas_on_nodes_map.get(node_name_from_prev, 0) + 1
                
                # Check if current nodes have enough resources for these previously allocated replicas
                for node_name_val, num_replicas_on_node in replicas_on_nodes_map.items():
                    gpus_needed_on_this_node = num_replicas_on_node * gpus_per_replica
                    if available_gpus.get(node_name_val, 0) < gpus_needed_on_this_node:
                        can_preserve = False
                        break
                
                if can_preserve:
                    new_allocations[job_key] = prev_alloc
                    # Deduct the GPUs from available_gpus. This iterates once per replica in prev_alloc.
                    for node_name_from_prev in prev_alloc:
                        available_gpus[node_name_from_prev] -= gpus_per_replica
        
        # Second pass: assign remaining jobs
        for job_key, job_info in jobs.items():
            if job_key in new_allocations:
                continue  # Already allocated or handled (e.g. 0-GPU job)
                
            gpus_per_replica = job_info.resources.get("nvidia.com/gpu", 1)
            assert gpus_per_replica == 1, f"Job {job_key} requests {gpus_per_replica} GPUs per replica, which is not 1."
                
            num_replicas = self._num_gpus_per_job // gpus_per_replica        
            # Try to allocate the job
            current_alloc = []
            for node_name, gpus in available_gpus.items():
                while len(current_alloc) < num_replicas and gpus >= gpus_per_replica:
                    current_alloc.append(node_name)
                    gpus -= gpus_per_replica
                    available_gpus[node_name] = gpus
            
            new_allocations[job_key] = current_alloc
            if len(current_alloc) < num_replicas:
                LOG.warning(f"Job {job_key}: wanted {num_replicas} replicas, got {len(current_alloc)}")
        
        # Calculate desired number of nodes based on total GPUs needed
        total_gpus_needed = len(jobs) * self._num_gpus_per_job
        gpus_per_node = node_template.resources.get("nvidia.com/gpu", 1)
        desired_nodes = math.ceil(total_gpus_needed / gpus_per_node)        
        LOG.info(f"DummyPolicy optimize results: {new_allocations}, desired_nodes: {desired_nodes}")
        return new_allocations, desired_nodes 