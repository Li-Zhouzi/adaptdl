from aiohttp import web
import logging
from datetime import datetime
import os
import asyncio
import time
from adaptdl_sched.config import get_global_profiler_port, get_checkpoint_path
from adaptdl_sched.global_profile_state import GlobalProfileState
from adaptdl.checkpoint import save_state


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class GlobalProfiler:
    """
    GlobalProfiler provides a REST interface for collecting profiling data
    from distributed training jobs. Currently, it has two endpoints:
    1. /healthz for health checks.
    2. /profile for receiving profiling data from jobs.
    """

    def __init__(self, port, host='0.0.0.0'):
        self._host = host
        self._port = port
        # Initialize the global profile state
        self._global_state = GlobalProfileState()
        # Load existing state if available
        try:
            from adaptdl.checkpoint import load_state
            load_state(self._global_state)
            LOG.info("Loaded existing global profile state")
        except Exception as e:
            LOG.info("No existing global profile state found, starting fresh: %s", e)

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_profile(self, request):
        # Endpoint for receiving profile data from jobs.
        try:
            profile_data = await request.json()
            LOG.info("Received profile data at %s: %s", datetime.now(), profile_data)
            
            # Extract application type from the profile data
            # You can modify this to extract the application type as needed
            application = profile_data.get('application')
            LOG.info("Application: %s", application)
            
            # Extract the actual profile data (excluding metadata like application)
            actual_profile_data = {k: v for k, v in profile_data.items() 
                                 if k != 'application'}
            
            # Update the global profile state
            self._global_state.update_profile(application, actual_profile_data)
            
            # Check if it's time to fit perf_params
            if self._global_state.should_fit_perf_params():
                LOG.info("Fitting perf_params for all applications")
                self._global_state.fit_all_perf_params()
                
                # Save the state to persistent storage
                try:
                    save_state(self._global_state, sync=False)
                    LOG.info("Saved global profile state to persistent storage")
                except Exception as e:
                    LOG.error("Error saving global profile state: %s", e)
            
            return web.json_response({"status": "success", "application": application})
        except Exception as e:
            LOG.error("Error processing profile data: %s", e)
            return web.json_response({"status": "error", "message": str(e)}, status=400)

    def run(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.post('/profile', self._handle_profile),
        ])
        
        LOG.info("GlobalProfiler starting on %s:%s", self._host, self._port)
        web.run_app(self.app, host=self._host, port=self._port)


if __name__ == "__main__":
    logging.basicConfig()
    
    # Set checkpoint path environment variable
    os.environ["ADAPTDL_CHECKPOINT_PATH"] = get_checkpoint_path()
    
    # Get port from config
    port = int(get_global_profiler_port())
    
    profiler = GlobalProfiler(port)
    profiler.run()
