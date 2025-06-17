import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
from collections import defaultdict

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class WidthCalculator:
    """
    WidthCalculator aggregates profile data from all active jobs.
    """

    def __init__(self):
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1", "", "adaptdljobs")
        
        # Local dictionary to maintain job data
        self._profile_data = {} # _profile_data[application] = profile of the job
        self._goodput_profile_data = {} # _goodput_profile_data[application][epoch] = goodput

    async def run(self):
        """Main loop that periodically aggregates job profiles."""
        LOG.info("Starting WidthCalculator")
        
        # Main aggregation loop
        iter_cnt = 0
        while True:
            await self._update_job_data()
            iter_cnt += 1
            if iter_cnt % 30 == 0:
                self._compute_width() # compute width every 300 seconds
                iter_cnt = 0
            await asyncio.sleep(10)

    async def _update_job_data(self):
        """Find all active jobs and extract their data."""
        """To update the profiles"""
        try:
            job_list = await self._objs_api.list_namespaced_custom_object(
                *self._custom_resource)
        except Exception as e:
            LOG.error(f"Failed to list AdaptDL jobs: {e}")
            return

        active_jobs = []
        for job in job_list["items"]:
            if job.get("status", {}).get("phase") in ["Pending", "Running", "Starting", "Stopping"]:
                active_jobs.append(job)

        LOG.info(f"Found {len(active_jobs)} active jobs")

        # Extract data from each job
        for job in active_jobs:
            hints = job.get("status", {}).get("train", {})
            epoch = hints.get("epoch", None)
            job_name = job["metadata"]["name"]
            application = job_name.split("-")[0]
            if epoch is None or application is None:
                raise ValueError(f"Epoch or application is not set for job {job_name}")
            
            if "new_profile" in hints:
                if application not in self._profile_data:
                    self._profile_data[application] = {}
                if epoch not in self._profile_data[application]:
                    self._profile_data[application][epoch] = {}
                
                for key, value in hints["new_profile"].items():
                    if key not in self._profile_data[application][epoch]:
                    # key is (num_nodes, num_replicas, atomic_bsz)
                        self._profile_data[application][epoch][key] = value
                    else:
                        self._profile_data[application][epoch][key]["optim_step_time"] += value["optim_step_time"]
                        self._profile_data[application][epoch][key]["optim_sync_time"] += value["optim_sync_time"]
                        self._profile_data[application][epoch][key]["optim_count"] += value["optim_count"]
                
            if "new_goodput_profile" in hints:
                if application not in self._goodput_profile_data:
                    self._goodput_profile_data[application] = {}

                for key, value in hints["new_goodput_profile"].items():
                    # key is (num_nodes, num_replicas)
                    if key not in self._goodput_profile_data[application][epoch]:
                        self._goodput_profile_data[application][epoch][key] = value
                    else:
                        self._goodput_profile_data[application][epoch][key] = value # overwrite the existing value
            
            LOG.info(f"Updated data for job {job_name}: {application}-{epoch}")
            
    async def _compute_width(self):
        """Compute the width of the job."""
        return 1


async def main():
    """Main entry point for the width calculator."""
    logging.basicConfig(level=logging.INFO)
    kubernetes.config.load_incluster_config()
    
    calculator = WidthCalculator()
    await calculator.run()


if __name__ == "__main__":
    asyncio.run(main())
