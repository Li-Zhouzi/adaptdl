import logging
import math
from collections import OrderedDict

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class DummyPolicy(object):
    def __init__(self, num_gpus_per_job=1):
        # Support both single integer and list of integers
        if isinstance(num_gpus_per_job, int):
            self._num_gpus_per_job = [num_gpus_per_job]
        else:
            self._num_gpus_per_job = num_gpus_per_job
        self._job_gpu_assignments = {}  # Track which job uses which GPU count
        
        # DEBUGGING BRANCH: the following three flags are only used for bert debugging.
        self._post400_flip_done = False  # Track the one-time downscale to 4 after 400
        self._post400_back_done = False  # Track the immediate upscale back to 8 after the downscale
        self._current_flip_cycle = None   # Track the 200-progress window where flip sequence applies
        LOG.info(f"DummyPolicy initialized with GPU options: {self._num_gpus_per_job}")

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
        Optimizes allocations for all jobs. Each job gets a fixed number of GPUs assigned
        when it first enters the system (using the first available number from the options list).
        This assignment stays fixed for the lifetime of the job.
        """

        # Only for debugging purposes
        assert len(jobs) == 1, "DEBUGGING BRANCH"
        job = list(jobs.values())[0]
        assert job.application == "cifar10", "DEBUGGING BRANCH"
        LOG.info(f"DummyPolicy optimize: progress={job.progress}")
        if job.progress < 2000:
            assigned_gpu = 12
        elif job.progress < 3000:
            # Before 200, use 4 GPUs
            assigned_gpu = 8
            # Reset post-400 sequence flags before crossing 400
            self._post400_flip_done = False
            self._post400_back_done = False
            self._current_flip_cycle = None
        elif job.progress < 8000:
            # [200, 400): use 8 GPUs
            assigned_gpu = 12
            # Reset post-400 sequence flags before crossing 400
            self._post400_flip_done = False
            self._post400_back_done = False
        elif job.progress < 10000:
            assigned_gpu = 8
        else:
            assigned_gpu = 4
        available_gpus = {node_name: node.resources.get("nvidia.com/gpu", 0) 
                        for node_name, node in nodes.items()}
        current_alloc = []
        for node_name, gpus in available_gpus.items():
            while len(current_alloc) < assigned_gpu and gpus >= 1:
                current_alloc.append(node_name)
                gpus -= 1
                available_gpus[node_name] = gpus
        # Return shape must be (allocations_dict, desired_nodes)
        job_key = list(jobs.keys())[0]
        gpus_per_node = node_template.resources.get("nvidia.com/gpu", 1)
        return {job_key: current_alloc}, math.ceil(assigned_gpu / 4) # DEBUGGING BRANCH

        new_allocations = {}
        # Track available GPUs on each node
        available_gpus = {node_name: node.resources.get("nvidia.com/gpu", 0) 
                         for node_name, node in nodes.items()}
        
        # Clean up assignments for jobs that no longer exist
        jobs_to_remove = [job_key for job_key in self._job_gpu_assignments if job_key not in jobs]
        for job_key in jobs_to_remove:
            del self._job_gpu_assignments[job_key]
        
        # Assign GPU counts to new jobs
        for job_key in jobs:
            if job_key not in self._job_gpu_assignments:
                # This is a new job, assign it a GPU count
                # Find which GPU counts are currently in use
                gpu_counts_in_use = set(self._job_gpu_assignments.values())
                
                # Find the first available GPU count from the options
                assigned_gpu_count = None
                if len(self._num_gpus_per_job) == 1:
                    assigned_gpu_count = self._num_gpus_per_job[0]
                else:
                    for gpu_count in self._num_gpus_per_job:
                        if gpu_count not in gpu_counts_in_use:
                            assigned_gpu_count = gpu_count
                            break
                
                if assigned_gpu_count is None:
                    # All options are in use, default to the first option
                    raise ValueError(f"Job {job_key}: All GPU counts in use, assigning {self._num_gpus_per_job[0]} GPUs (may share with other jobs)")
                    assigned_gpu_count = self._num_gpus_per_job[0]
                    LOG.warning(f"Job {job_key}: All GPU counts in use, assigning {assigned_gpu_count} GPUs (may share with other jobs)")
                
                self._job_gpu_assignments[job_key] = assigned_gpu_count
                LOG.info(f"Job {job_key}: Assigned {assigned_gpu_count} GPUs")
        
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
                
            # Get the fixed GPU assignment for this job
            target_gpu_count = self._job_gpu_assignments[job_key]
            
            # Calculate total GPUs this job had in its previous allocation
            gpus_in_prev_alloc = len(prev_alloc) * gpus_per_replica

            if job_info.max_replicas == 1:
                # This job has never run before, so the grad and perf params are none, which may lead to a bad bsz.
                target_gpu_count = 1
                
            if gpus_in_prev_alloc == target_gpu_count:
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
                continue  # Already allocated
                
            gpus_per_replica = job_info.resources.get("nvidia.com/gpu", 1)
            assert gpus_per_replica == 1, f"Job {job_key} requests {gpus_per_replica} GPUs per replica, which is not 1."
            
            # Get the fixed GPU assignment for this job
            target_gpu_count = self._job_gpu_assignments[job_key]
            if job_info.max_replicas == 1:
                # This job has never run before, so the grad and perf params are none, which may lead to a bad bsz.
                target_gpu_count = 1
            num_replicas = target_gpu_count // gpus_per_replica
            # Try to allocate the job
            current_alloc = []
            for node_name, gpus in available_gpus.items():
                while len(current_alloc) < num_replicas and gpus >= gpus_per_replica:
                    current_alloc.append(node_name)
                    gpus -= gpus_per_replica
                    available_gpus[node_name] = gpus
            
            
            if len(current_alloc) < num_replicas:
                new_allocations[job_key] = []
                LOG.warning(f"Job {job_key}: wanted {num_replicas} replicas, got 0.")
            else:
                new_allocations[job_key] = current_alloc
        
        # Calculate desired number of nodes based on total GPUs needed
        total_gpus_needed = sum(self._job_gpu_assignments.values())
        gpus_per_node = node_template.resources.get("nvidia.com/gpu", 1)
        desired_nodes = math.ceil(total_gpus_needed / gpus_per_node)        
        LOG.info(f"DummyPolicy optimize results: allocations={new_allocations}, "
                f"gpu_assignments={self._job_gpu_assignments}, desired_nodes={desired_nodes}")
        return new_allocations, desired_nodes 