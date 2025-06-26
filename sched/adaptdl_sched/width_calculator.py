import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
import sys
import os
import pickle
from collections import defaultdict
from global_profile_state import PartialMetricsState

# Add adaptdl to path for importing checkpoint functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'adaptdl'))

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
        
        # Partial metrics state for checkpointing
        self._partial_metrics_state = PartialMetricsState()

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
            
            # Process new_profile data
            if "new_profile" in hints:
                if application not in self._profile_data:
                    self._profile_data[application] = {}
                if epoch not in self._profile_data[application]:
                    self._profile_data[application][epoch] = {}
                
                for key, value in hints["new_profile"].items():
                    if key not in self._profile_data[application][epoch]:
                        self._profile_data[application][epoch][key] = {
                            "optim_step_time": value["optim_step_time"],
                            "optim_sync_time": value["optim_sync_time"],
                            "optim_count": value["optim_count"]
                        }
                    else:
                        # Accumulate with existing data
                        self._profile_data[application][epoch][key]["optim_step_time"] += value["optim_step_time"]
                        self._profile_data[application][epoch][key]["optim_sync_time"] += value["optim_sync_time"]
                        self._profile_data[application][epoch][key]["optim_count"] += value["optim_count"]
                
            # Process new_goodput_profile data
            if "new_goodput_profile" in hints:
                if application not in self._goodput_profile_data:
                    self._goodput_profile_data[application] = {}
                if epoch not in self._goodput_profile_data[application]:
                    self._goodput_profile_data[application][epoch] = {}

                for key, value in hints["new_goodput_profile"].items():
                    if key not in self._goodput_profile_data[application][epoch]:
                        self._goodput_profile_data[application][epoch][key] = {
                            "goodput": value["goodput"],
                            "cnt": value["cnt"]
                        }
                    else:
                        # Update with latest goodput value and accumulate count
                        self._goodput_profile_data[application][epoch][key]["goodput"] = value["goodput"]
                        self._goodput_profile_data[application][epoch][key]["cnt"] += value["cnt"]
            
            LOG.info(f"Updated data for job {job_name}: {application}-{epoch}")
            
    async def _compute_width(self):
        """Compute the width of the job."""
        LOG.info("Computing width based on aggregated profiles")
        LOG.info(f"Profile data: {self._profile_data}")
        LOG.info(f"Goodput profile data: {self._goodput_profile_data}")
        
        # Update the partial metrics state with aggregated data
        self._update_partial_metrics_state()
        
        # Save the partial metrics checkpoint
        self._save_partial_metrics_checkpoint()
        
        return 1
    
    def _update_partial_metrics_state(self):
        """Update the partial metrics state with aggregated profile data."""
        # Clear existing profile data
        self._partial_metrics_state.profile.clear()
        
        # Aggregate profile data from all applications and epochs
        for application, epochs in self._profile_data.items():
            for epoch, profiles in epochs.items():
                for key, value in profiles.items():
                    # Convert key to tuple format if it's a string
                    if isinstance(key, str):
                        try:
                            # Parse the key string to extract components
                            key_clean = key.strip("()")
                            parts = [int(x.strip()) for x in key_clean.split(",")]
                            if len(parts) == 3:
                                key_tuple = tuple(parts)
                                self._partial_metrics_state.profile[key_tuple]["optim_step_time"] += value["optim_step_time"]
                                self._partial_metrics_state.profile[key_tuple]["optim_sync_time"] += value["optim_sync_time"]
                                self._partial_metrics_state.profile[key_tuple]["optim_count"] += value["optim_count"]
                        except (ValueError, AttributeError):
                            LOG.warning(f"Could not parse profile key: {key}")
                            continue
                    else:
                        # Key is already in tuple format
                        self._partial_metrics_state.profile[key]["optim_step_time"] += value["optim_step_time"]
                        self._partial_metrics_state.profile[key]["optim_sync_time"] += value["optim_sync_time"]
                        self._partial_metrics_state.profile[key]["optim_count"] += value["optim_count"]
        
        # For now, set perf_params to None - you can compute this based on your logic
        # self._partial_metrics_state.perf_params = computed_perf_params
        
        LOG.info(f"Updated partial metrics state with {len(self._partial_metrics_state.profile)} profile entries")
    
    def _save_partial_metrics_checkpoint(self):
        """Save the partial metrics state to a checkpoint file."""
        try:
            # Import checkpoint path from adaptdl
            from adaptdl.env import checkpoint_path
            
            checkpoint_dir = checkpoint_path()
            if checkpoint_dir is None:
                LOG.warning("Checkpoint path is None, cannot save partial metrics checkpoint")
                return
            
            checkpoint_file = os.path.join(checkpoint_dir, self._partial_metrics_state.name)
            
            with open(checkpoint_file, "wb") as f:
                self._partial_metrics_state.save(f)
            
            LOG.info(f"Successfully saved partial metrics checkpoint to {checkpoint_file}")
            
        except ImportError as e:
            LOG.warning(f"Could not import adaptdl checkpoint functions: {e}")
        except Exception as e:
            LOG.error(f"Failed to save partial metrics checkpoint: {e}")


async def main():
    """Main entry point for the width calculator."""
    logging.basicConfig(level=logging.INFO)
    kubernetes.config.load_incluster_config()
    
    calculator = WidthCalculator()
    await calculator.run()


if __name__ == "__main__":
    asyncio.run(main())
