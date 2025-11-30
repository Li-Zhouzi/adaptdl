import logging
import math
from collections import OrderedDict

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

APPLICATION_NAMES = ["bert", "cifar10", "ncf", "imagenet", "deepspeech2", "yolov3"]

class FixedWidthPolicy(object):
    '''
    This policy is used to allocate jobs to nodes with a fixed width.
    The width is the number of GPUs per job: width[job_type][epoch] = num_gpus
    '''
    def __init__(self, width):
        self.width = width
        self.max_nodes = 40

    def _sort_nodes(self, nodes):
        return OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible,
                                                  kv[0])))

    def allocate_job(self, job_info, nodes):
        """
        Copied from Pollux Policy. allocate the min number of replicas to a new arrival.
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

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        '''
        Give each job width[job_type][epoch] GPUs. Prioritize jobs that already have the correct number of GPUs in prev_allocations. 
        For those jobs that don't have the correct number of GPUs, allocate to the best effort.
        Ask for total number of desired nodes.
        New logic 11/7/2025: the jobs start with 1 GPU (when grad_params and perf_params are None). It runs on 1 GPU for one rescaling time,
        and after the training starts, its params are updated, and thus, max_replicas is not 1, and it gets the width[job_type][epoch] GPUs. 
        However, the desired number of GPUs is always width[job_type][epoch], even when the job is running on 1 GPU. 
        This whole process is because the bsz is only updated when an epoch is finished or the job is rescaled.
        This logic is disabled on Nov 30.
        '''
        new_allocations = {}
        # Track available GPUs on each node
        available_gpus = {node_name: node.resources.get("nvidia.com/gpu", 0) 
                         for node_name, node in nodes.items()}
        total_gpus_needed = 0
        # First pass: preserve existing allocations that already have the correct number of GPUs
        for job_key, prev_alloc in prev_allocations.items():
            if job_key not in jobs:
                continue  # Job no longer exists
                
            job_info = jobs[job_key]
            gpus_per_replica = job_info.resources.get("nvidia.com/gpu", 1)
            assert gpus_per_replica == 1, f"Job {job_key} requests {gpus_per_replica} GPUs per replica, which is not 1."
                
            # Calculate total GPUs this job had in its previous allocation
            gpus_in_prev_alloc = len(prev_alloc)
            print("Here: ", job_info.application, job_info.epoch)
            gpu_wanted = self.width[job_info.application][str(job_info.epoch)]
            # if job_info.max_replicas == 1:
            #     # This job has never run before, so the grad and perf params are none, which may lead to a bad bsz.
            #     gpu_wanted = 1
            
            # only fulfill the job if it has the correct number of GPUs
            if gpus_in_prev_alloc == gpu_wanted:
                new_allocations[job_key] = prev_alloc
                # Deduct the GPUs from available_gpus. This iterates once per replica in prev_alloc.
                for node_name_from_prev in prev_alloc:
                    available_gpus[node_name_from_prev] -= gpus_per_replica
                total_gpus_needed += self.width[job_info.application][str(job_info.epoch)]
        
        # Second pass: assign remaining jobs
        for job_key, job_info in jobs.items():
            if job_key in new_allocations:
                continue  # Already allocated or handled (e.g. 0-GPU job)
                
            gpus_per_replica = job_info.resources.get("nvidia.com/gpu", 1)
            assert gpus_per_replica == 1, f"Job {job_key} requests {gpus_per_replica} GPUs per replica, which is not 1."
                
            gpu_wanted = self.width[job_info.application][str(job_info.epoch)]     
            # if job_info.max_replicas == 1:
            #     # This job has never run before, so the grad and perf params are none, which may lead to a bad bsz.
            #     gpu_wanted = 1
            # Try to allocate the job
            current_alloc = []
            for node_name, gpus in available_gpus.items():
                while len(current_alloc) < gpu_wanted and gpus >= gpus_per_replica:
                    current_alloc.append(node_name)
                    gpus -= gpus_per_replica
                    available_gpus[node_name] = gpus
            
            new_allocations[job_key] = current_alloc
            total_gpus_needed += self.width[job_info.application][str(job_info.epoch)]
            if len(current_alloc) < gpu_wanted:
                LOG.warning(f"Job {job_key}: wanted {gpu_wanted} GPUs, got {len(current_alloc)}")

        desired_nodes = math.ceil(total_gpus_needed / node_template.resources.get("nvidia.com/gpu", 1))

        # Calculate actual nodes used in allocations
        actual_nodes_used = len(set.union(*map(set, new_allocations.values()))) if new_allocations else 0

        # Desired nodes should be at least the actual nodes used (due to imperfect bin-packing)
        desired_nodes = max(desired_nodes, actual_nodes_used)

        LOG.info(f"FixedWidthPolicy optimize results: {new_allocations}, desired_nodes: {desired_nodes}")
        if desired_nodes > self.max_nodes:
            desired_nodes = self.max_nodes
            LOG.warning(f"Desired nodes {desired_nodes} is greater than max nodes {self.max_nodes}, setting to max nodes {self.max_nodes}")
        return new_allocations, desired_nodes 
    


    