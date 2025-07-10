import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
import sys
import os
import pickle

# Add adaptdl to path for importing checkpoint functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'adaptdl'))

# Import after adding to path
from adaptdl.global_profile_state import GlobalProfileState

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class WidthCalculator:
    """
    WidthCalculator loads global profiler state and computes width based on aggregated profiles.
    """

    def __init__(self):
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1", "", "adaptdljobs")
        
        # Global profile state for loading checkpoint data
        self._global_state = GlobalProfileState()

    async def run(self):
        """Main loop that periodically computes width."""
        LOG.info("Starting WidthCalculator")
        
        # Main computation loop - call _compute_width every 5 minutes (300 seconds)
        while True:
            await self._compute_width()
            await asyncio.sleep(300)  # Sleep for 5 minutes

    async def _compute_width(self):
        """Compute the width of the job."""
        LOG.info("Computing width based on global profiler state")
        
        # Load global profiler state from checkpoint
        self._load_global_profiler_state()
        
        # Log the loaded state for debugging
        LOG.info(f"Loaded global profiles for applications: {list(self._global_state.global_profiles.keys())}")
        LOG.info(f"Loaded global perf_params for applications: {list(self._global_state.global_perf_params.keys())}")
        
        # TODO: Implement the remaining width computation logic here
        # The global state is now loaded and available in self._global_state
        # You can access:
        # - self._global_state.global_profiles[application][key] for profile data
        # - self._global_state.global_perf_params[application] for performance parameters
        
        return 1
    
    def _load_global_profiler_state(self):
        """Load the global profiler state from checkpoint."""
        # Try to load from the pollux checkpoint path
        checkpoint_path = "/pollux/checkpoint"
        state_name = "global-profile-state"
        checkpoint_file = os.path.join(checkpoint_path, state_name)
        
        if not os.path.exists(checkpoint_file):
            LOG.warning(f"Global profile state file not found: {checkpoint_file}")
            LOG.info("Starting with empty global profiler state")
            return
        
        LOG.info(f"Loading global profiler state from: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            self._global_state.load(f)
        LOG.info("Successfully loaded global profiler state from checkpoint")


async def main():
    """Main entry point for the width calculator."""
    logging.basicConfig(level=logging.INFO)
    kubernetes.config.load_incluster_config()
    
    calculator = WidthCalculator()
    await calculator.run()


if __name__ == "__main__":
    asyncio.run(main())
