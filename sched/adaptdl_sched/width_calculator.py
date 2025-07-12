import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
import sys
import os
import pickle
from _compute_width import get_width
from aiohttp import web

# Add adaptdl to path for importing checkpoint functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'adaptdl'))

# Import after adding to path
from adaptdl.global_profile_state import GlobalProfileState
from adaptdl_sched.config import get_width_calculator_port, get_checkpoint_path


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class WidthCalculator:
    """
    WidthCalculator loads global profiler state and computes width based on aggregated profiles.
    Also provides a web service to expose the current width.
    """

    def __init__(self, budget):
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1", "", "adaptdljobs")
        
        # Global profile state for loading checkpoint data
        self._global_state = GlobalProfileState()
        
        self.budget = budget
        self.width = None

    def get_width(self):
        """
        Get the current computed width.
        
        Returns:
            dict or None: The current width dictionary or None if not computed yet
        """
        return self.width

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_width(self, request):
        """
        HTTP endpoint to get the current width.
        
        Returns:
            JSON response with width data or error message
        """
        try:
            if self.width is None:
                return web.json_response({
                    "error": "Width not available",
                    "message": "Width calculator has not computed width yet or computation failed"
                }, status=503)
            
            return web.json_response({
                "width": self.width,
            })
        except Exception as e:
            LOG.error(f"Error getting width: {e}")
            return web.json_response({
                "error": "Internal server error",
                "message": str(e)
            }, status=500)

    async def run(self):
        """Main loop that periodically computes width and serves web requests."""
        LOG.info("Starting WidthCalculator with web service")
        
        # Start web service
        app = web.Application()
        app.router.add_get('/healthz', self._handle_healthz)
        app.router.add_get('/width', self._handle_width)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", int(get_width_calculator_port()))
        await site.start()
        LOG.info("Width service started on port %s", get_width_calculator_port())
        
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
        width = get_width(self._global_state, self.budget)
        self.width = width
    
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
    
    # Get budget from environment variable or use default
    budget = int(os.environ.get("WIDTH_CALCULATOR_BUDGET", "100"))
    calculator = WidthCalculator(budget)
    await calculator.run()


if __name__ == "__main__":
    asyncio.run(main())
