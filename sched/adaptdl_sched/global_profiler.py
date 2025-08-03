from aiohttp import web
import logging
from datetime import datetime
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from adaptdl_sched.config import get_global_profiler_port, get_checkpoint_path
from adaptdl.global_profile_state import GlobalProfileState
from adaptdl.checkpoint import save_state
from ._configs import APPLICATIONS, NUM_GPU_PER_NODE
from adaptdl.goodput import GoodputFunction

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class GlobalProfiler:
    """
    GlobalProfiler provides a REST interface for collecting profiling data
    from distributed training jobs. Currently, it has two endpoints:
    1. /healthz for health checks.
    2. /profile for receiving profiling data from jobs.
    """

    def __init__(self, port, host='0.0.0.0', grad_params_alpha=0.1):
        self._host = host
        self._port = port
        self._grad_params_alpha = grad_params_alpha
        # Initialize the global profile state
        self._global_state = GlobalProfileState()
        
        # Initialize thread pool executor for heavy computations
        self._executor = ThreadPoolExecutor(max_workers=2)
        LOG.info("Initialized ThreadPoolExecutor with 2 worker threads")
        
        # Load existing state if available - try multiple sources
        state_loaded = False
        
        # First, try to load from default checkpoint path
        try:
            from adaptdl.checkpoint import load_state
            load_state(self._global_state)
            LOG.info("Loaded existing global profile state from default checkpoint")
            state_loaded = True
        except Exception as e:
            LOG.info("No global profile state found at default checkpoint: %s", e)
        
        # If default checkpoint loading failed, try width-calculator-init directory
        if not state_loaded:
            state_loaded = self._load_from_width_calculator_init()
        
        # If both failed, start fresh
        if not state_loaded:
            LOG.info("Starting with fresh global profile state")
    
    def __del__(self):
        """Cleanup executor on shutdown."""
        if hasattr(self, '_executor'):
            LOG.info("Shutting down ThreadPoolExecutor")
            self._executor.shutdown(wait=True)

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_profile(self, request):
        # Endpoint for receiving profile data from jobs.
        profile_data = await request.json()
        LOG.info("Received profile data at %s: %s", datetime.now(), profile_data)
        
        # Extract application type from the profile data
        # You can modify this to extract the application type as needed
        application = profile_data.get('application')
        
        # Extract the actual profile data (excluding metadata like application)
        actual_profile_data = {k: v for k, v in profile_data.items() 
                                if k != 'application'}
        
        # Always update the global profile state (for perf params fitting)
        self._global_state.update_profile(application, actual_profile_data, alpha=self._grad_params_alpha)
        
        # Note: perf_params fitting has been moved to _compute_goodput_with_fitting()
        # to avoid blocking profile reports
        
        return web.json_response({"status": "success", "application": application})

    async def _handle_get_goodput(self, request):
        """
        Endpoint for retrieving goodput data.
        
        Returns:
            JSON response containing goodput dictionary or error message
        """
        try:
            LOG.info("Received goodput request at %s", datetime.now())
            
            # Run the heavy computation in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            LOG.info("Dispatching goodput computation to executor thread pool")
            
            goodput_dict = await loop.run_in_executor(
                self._executor,
                self._compute_goodput_with_fitting
            )
            
            LOG.info("Successfully generated goodput dictionary for %d applications", len(goodput_dict))
            
            return web.json_response({
                "status": "success",
                "goodput": goodput_dict,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            LOG.error("Error generating goodput data: %s", str(e))
            return web.json_response({
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }, status=500)

    def _save_to_width_calculator_init(self):
        """Save a copy of the global profile state to width-calculator-init directory."""
        import os
        from adaptdl.env import checkpoint_path
        
        # Get the base checkpoint path from environment (same as adaptdl logic)
        base_checkpoint_path = checkpoint_path()
        
        if base_checkpoint_path is None:
            LOG.warning("No checkpoint path available, cannot save to width-calculator-init")
            return
            
        # Create width-calculator-init directory as sibling to checkpoint directory
        base_dir = os.path.dirname(base_checkpoint_path)  # e.g., /pollux
        width_calc_dir = os.path.join(base_dir, "width-calculator-init")
        
        
        os.makedirs(width_calc_dir, exist_ok=True)
        
        # Save the global state to the width calculator init directory
        checkpoint_file = os.path.join(width_calc_dir, "global-profile-state")
        
        try:
            with open(checkpoint_file, "wb") as f:
                self._global_state.save(f)
            LOG.info(f"Successfully saved global profile state to {checkpoint_file}")
        except Exception as e:
            LOG.error(f"Failed to save global profile state to {checkpoint_file}: {e}")

    def _load_from_width_calculator_init(self):
        """Load global profile state from width-calculator-init directory if it exists."""
        import os
        from adaptdl.env import checkpoint_path
        
        # Get the base checkpoint path from environment (same as adaptdl logic)
        base_checkpoint_path = checkpoint_path()
        
        if base_checkpoint_path is None:
            LOG.warning("No checkpoint path available, cannot load from width-calculator-init")
            return False
            
        # Create width-calculator-init directory path as sibling to checkpoint directory
        base_dir = os.path.dirname(base_checkpoint_path)  # e.g., /pollux
        width_calc_dir = os.path.join(base_dir, "width-calculator-init")
        checkpoint_file = os.path.join(width_calc_dir, "global-profile-state")
        
        if not os.path.exists(checkpoint_file):
            LOG.info(f"Width-calculator-init state file not found: {checkpoint_file}")
            return False
        
        try:
            with open(checkpoint_file, "rb") as f:
                self._global_state.load(f)
            LOG.info(f"Successfully loaded global profile state from width-calculator-init: {checkpoint_file}")
            return True
        except Exception as e:
            LOG.error(f"Failed to load global profile state from width-calculator-init {checkpoint_file}: {e}")
            return False

    def _compute_goodput_with_fitting(self):
        """
        Compute goodput with perf_params fitting if needed.
        This runs in the executor thread pool to avoid blocking the main event loop.
        """
        # Check if it's time to fit perf_params (every 60 seconds)
        if self._global_state.should_fit_perf_params():
            LOG.info(f"[TIMESTAMP: {time.time()}] Starting perf_params fitting in executor thread")
            fit_start_time = time.time()
            self._global_state.fit_all_perf_params()
            
            # Save the state to persistent storage
            save_state(self._global_state, sync=False)
            LOG.info("Saved global profile state to persistent storage")
            
            # Also save a copy to width-calculator-init directory
            self._save_to_width_calculator_init()
            LOG.info("Saved copy of global profile state to width-calculator-init directory")
            
            fit_duration = time.time() - fit_start_time
            LOG.info(f"[TIMESTAMP: {time.time()}] Completed perf_params fitting in {fit_duration:.3f} seconds")
        
        # Now compute and return goodput
        return self._load_goodput_function()
    
    def _load_goodput_function(self):
        """
        Load goodput functions and create a goodput dictionary using global profile state data.
        
        Args:
            global_profile_state: The global profile state containing profiles, perf_params, and grad_params
            
        Returns:
            dict: A dictionary where goodput_dict[app][epoch][num_replica] contains the optimized goodput
        """
        goodput_dict = {}
        global_profile_state = self._global_state
        
        # Validate that all applications have required data
        for application in global_profile_state.global_perf_params.keys():            
            # Get application configuration
            
            perf_params = global_profile_state.global_perf_params[application]
            profile = global_profile_state.global_profiles[application]
            
            goodput_dict[application] = {}
            app_config = APPLICATIONS[application]

            # For each epoch that has grad_params
            for epoch in global_profile_state.global_grad_params[application].keys():
                grad_params = global_profile_state.global_grad_params[application][epoch]
            

                # Create GoodputFunction with the global profile data
                goodput_fn = GoodputFunction(perf_params, grad_params, app_config.init_batch_size)
                
                goodput_dict[application][epoch] = {}
                
                # Calculate optimal goodput for replicas 1-64
                for num_replicas in range(1, 65):
                    # Check if we have profiled goodput for this configuration
                    if (application in global_profile_state.global_goodput_profile and
                        epoch in global_profile_state.global_goodput_profile[application] and
                        num_replicas in global_profile_state.global_goodput_profile[application][epoch]):
                        # Use profiled goodput
                        # NOTE: The goodput we store is already gain/second (actual goodput).
                        # The progress in the original implementation was incorrectly scaled by init_batch_size.
                        # This is fixed in _compute_width, where init_batch_size is not divided.
                        optimal_goodput = global_profile_state.global_goodput_profile[application][epoch][num_replicas]
                    else:
                        # Use prediction
                        num_nodes = max(1, (num_replicas + NUM_GPU_PER_NODE - 1) // NUM_GPU_PER_NODE)
                        # Optimize for the best goodput using application-specific config
                        optimal_goodput, _, _ = goodput_fn.optimize(
                            num_nodes, num_replicas, 
                            max_batch_size=app_config.max_batch_size,
                            atomic_bsz_range=(app_config.min_local_bsz, app_config.max_local_bsz),
                            accumulation=app_config.gradient_accumulation
                        )
                    
                    goodput_dict[application][epoch][num_replicas] = optimal_goodput
        return goodput_dict

    def run(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.post('/profile', self._handle_profile),
            web.get('/get_goodput', self._handle_get_goodput),
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
