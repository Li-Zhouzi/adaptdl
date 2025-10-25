import asyncio
import kubernetes_asyncio as kubernetes
import logging
import time
import os
import aiohttp
from adaptdl_sched._compute_width import get_width
from aiohttp import web
from adaptdl_sched.config import get_width_calculator_port, get_global_profiler_url
from ._configs import APPLICATIONS, ARRIVAL_RATE

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class WidthCalculator:
    """
    WidthCalculator fetches goodput data from the global profiler and computes width based on that data.
    Also provides a web service to expose the current width.
    """

    def __init__(self, budget):
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1", "", "adaptdljobs")
        
        self.budget = budget
        self.width = None
        
        # Global profiler URL configuration
        self._global_profiler_url = get_global_profiler_url()
        LOG.info(f"Global profiler URL: {self._global_profiler_url}")

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
            if self.width is not None: # for the experiment, it is perfect info case, so never compute width again.
                await asyncio.sleep(60)
                continue
            await self._compute_width()
            await asyncio.sleep(60)  # Sleep for 1 minutes

    async def _compute_width(self):
        """Compute the width of the job."""
        LOG.info(f"[TIMESTAMP: {time.time()}] Starting width calculation")
        LOG.info("Computing width based on goodput data from global profiler")
        time_start = time.time()
        
        # Fetch goodput data from global profiler
        goodput_dict = await self._fetch_goodput_from_global_profiler()
        time_fetch = time.time() - time_start
        LOG.info(f"Time taken to fetch goodput data: {time_fetch} seconds")
        time_start = time.time()
        
        if goodput_dict is None:
            LOG.error("Failed to fetch goodput data from global profiler")
            self.width = None
            return
        
        # Validate the goodput dictionary has all required applications and epochs
        if not self._check_goodput_dict(goodput_dict):
            LOG.error("Goodput data validation failed - cannot compute width")
            self.width = None
            return
        
        # Log the fetched goodput data for debugging
        # LOG.info(f"Fetched goodput data for applications: {list(goodput_dict.keys())}")
        # for app in goodput_dict.keys():
        #     LOG.info(f"  Application {app} has {len(goodput_dict[app])} epochs")
        
        try:
            width = get_width(goodput_dict, self.budget)
            LOG.info(f"Computed width: {width}")
        except Exception as e:
            LOG.error(f"Error computing width: {e}")
            width = None
        time_compute = time.time() - time_start
        LOG.info(f"Time taken to compute width: {time_compute} seconds")
        self.width = width

    async def _fetch_goodput_from_global_profiler(self):
        """
        Fetch goodput dictionary from the global profiler.
        
        Returns:
            dict: Goodput dictionary or None if fetch failed
        """
        try:
            LOG.info(f"Fetching goodput data from: {self._global_profiler_url}/get_goodput")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._global_profiler_url}/get_goodput") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            goodput_dict = data.get("goodput", {})
                            LOG.info(f"Successfully fetched goodput data for {len(goodput_dict)} applications")
                            return goodput_dict
                        else:
                            LOG.error(f"Global profiler returned error: {data.get('message', 'Unknown error')}")
                            return None
                    else:
                        LOG.error(f"Global profiler returned status {response.status}")
                        return None
                        
        except Exception as e:
            LOG.error(f"Error fetching goodput data from global profiler: {e}")
            return None

    def _check_goodput_dict(self, goodput_dict):
        """
        Check whether the goodput dictionary contains all required applications and epochs.
        
        For all applications in ARRIVAL_RATE with arrival_rate > 0, validates that
        goodput_dict[app][epoch] exists for all epoch in range(max_epochs).
        
        Args:
            goodput_dict (dict): The goodput dictionary from global profiler
            
        Returns:
            bool: True if all required data is present, False otherwise
        """
        if not goodput_dict:
            LOG.error("Goodput dictionary is empty")
            return False
        
        missing_data = []
        
        # Check each application with positive arrival rate
        for app_name, arrival_rate in ARRIVAL_RATE.items():
            if arrival_rate <= 0:
                continue  # Skip applications with zero arrival rate
                
            # Check if application exists in goodput dictionary
            if app_name not in goodput_dict:
                missing_data.append(f"Application '{app_name}' not found in goodput dictionary")
                continue
                
            # Get expected number of epochs for this application
            app_config = APPLICATIONS[app_name]
            max_epochs = app_config.max_epochs
            
            # Check if all epochs exist
            missing_epochs = []
            for epoch in range(max_epochs):
                if str(epoch) not in goodput_dict[app_name] and epoch not in goodput_dict[app_name]:
                    # Check both string and integer keys since JSON might convert to strings
                    missing_epochs.append(epoch)
            
            if missing_epochs:
                missing_data.append(f"Application '{app_name}' missing epochs: {missing_epochs}")
        
        if missing_data:
            LOG.error("Goodput dictionary validation failed:")
            for issue in missing_data:
                LOG.error(f"  - {issue}")
            return False
        
        LOG.info("Goodput dictionary validation passed - all required applications and epochs present")
        return True

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
