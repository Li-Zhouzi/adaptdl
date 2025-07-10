import pickle
import time
import numpy as np
from adaptdl.checkpoint import State
from adaptdl.goodput import fit_perf_params


class GlobalProfileState(State):
    """
    A global profile state that contains profile and perf_params for each application type.
    This maintains global profiles and perf_params across different jobs and applications.
    """
    def __init__(self, name="global-profile-state"):
        super().__init__(name)
        # global_profiles[application][key] contains the profile for that application and configuration
        # Simple dictionary structure: {application: {key: profile_dict}}
        self.global_profiles = {}
        # global_perf_params[application] contains the perf_params for that application type
        self.global_perf_params = {}
        # global_grad_params[application][epoch] contains the grad_params for that application and epoch
        self.global_grad_params = {}
        self.global_size = {}
        self.last_fit_time = time.time()

    def _get_or_create_profile(self, application, key):
        """Get existing profile or create a new one with proper defaults."""
        if application not in self.global_profiles:
            self.global_profiles[application] = {}
        
        if key not in self.global_profiles[application]:
            self.global_profiles[application][key] = {
                "accum_step_time": 0.0,
                "accum_count": 0,
                "optim_step_time": 0.0,
                "optim_sync_time": 0.0,
                "optim_count": 0
            }
        
        return self.global_profiles[application][key]

    def update_profile(self, application, profile_data, alpha=0.1):
        """
        Update the global profile for a specific application type.
        This follows the same logic as in _metrics.py profile_step_commit.
        
        Args:
            application (str): The application type identifier
            profile_data (dict): Profile data containing the profile information
            alpha (float): Alpha parameter for exponential moving average of grad_params
        """
        key = (profile_data["num_nodes"], profile_data["num_replicas"], profile_data["atomic_bsz"])
        profile = self._get_or_create_profile(application, key)

        if profile_data.get("accumulation_step", False):
            profile["accum_step_time"] += profile_data.get("step_time", 0.0)
            profile["accum_count"] += 1
        else:
            profile["optim_step_time"] += profile_data.get("step_time", 0.0)
            profile["optim_sync_time"] += profile_data.get("sync_time", 0.0)
            profile["optim_count"] += 1
        
        # Update grad_params if provided in profile_data
        if "grad_norm_sqr" in profile_data and "grad_variance" in profile_data:
            epoch = profile_data.get("epoch", 0)  # Default to epoch 0 if not provided
            
            # Initialize application and epoch if not exists
            if application not in self.global_grad_params:
                self.global_grad_params[application] = {}
            if epoch not in self.global_grad_params[application]:
                self.global_grad_params[application][epoch] = (0.0, 0.0)  # Default values
            
            # Get current values
            old_norm_sqr, old_variance = self.global_grad_params[application][epoch]
            new_norm_sqr = profile_data["grad_norm_sqr"]
            new_variance = profile_data["grad_variance"]
            
            # Apply exponential moving average
            updated_norm_sqr = (1 - alpha) * old_norm_sqr + alpha * new_norm_sqr
            updated_variance = (1 - alpha) * old_variance + alpha * new_variance
            
            self.global_grad_params[application][epoch] = (updated_norm_sqr, updated_variance)


    def fit_perf_params_for_application(self, application):
        """
        TODO: NEED TO CHECK THIS FUNCTION
        Fit performance parameters for a specific application using its profile data.
        This follows the same logic as _fit_perf_params in _metrics.py.
        
        Args:
            application (str): The application type identifier
        """
        if application not in self.global_profiles:
            return
            
        profile = self.global_profiles[application]
        # Filter profile to only include entries with optim_count > 0
        profile = {k: v for k, v in profile.items() if v.get("optim_count", 0) > 0}
        
        if not profile:
            return
            
        # Convert profile into numpy arrays
        num_nodes, num_replicas, atomic_bsz = (np.array(k) for k in zip(*profile))
        accum_step_time = np.array([v.get("accum_step_time", 0.0) for v in profile.values()])
        accum_count = np.array([v.get("accum_count", 0) for v in profile.values()])
        optim_step_time = np.array([v.get("optim_step_time", 0.0) for v in profile.values()])
        optim_sync_time = np.array([v.get("optim_sync_time", 0.0) for v in profile.values()])
        optim_count = np.array([v.get("optim_count", 0) for v in profile.values()])
        
        # Ensure all optim_count > 0
        if not np.all(optim_count > 0):
            return
            
        # Non-sync time during optimization steps should be approximately equal to
        # accumulation step time, combine those data points
        if not np.all(optim_step_time >= optim_sync_time):
            return
            
        accum_step_time += optim_step_time - optim_sync_time
        accum_count += optim_count
        accum_step_time /= accum_count
        optim_step_time /= optim_count
        
        # Fit the performance parameters
        perf_params = fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                                    accum_step_time, optim_step_time)
        self.global_perf_params[application] = perf_params

    def should_fit_perf_params(self):
        """
        Check if it's time to fit perf_params (every 300 seconds).
        
        Returns:
            bool: True if it's time to fit perf_params
        """
        current_time = time.time()
        return current_time - self.last_fit_time > 60

    def fit_all_perf_params(self):
        """
        Fit perf_params for all applications and update the last fit time.
        """
        for application in self.global_profiles.keys():
            self.fit_perf_params_for_application(application)
        self.last_fit_time = time.time()

    def save(self, fileobj):
        """Save global_profiles and global_perf_params to the checkpoint."""
        pickle.dump(self.global_profiles, fileobj)
        pickle.dump(self.global_perf_params, fileobj)
        pickle.dump(self.global_grad_params, fileobj)
        pickle.dump(self.last_fit_time, fileobj)
        pickle.dump(self.global_size, fileobj)
    
    def load(self, fileobj):
        """Load global_profiles and global_perf_params from the checkpoint."""
        self.global_profiles = pickle.load(fileobj)
        self.global_perf_params = pickle.load(fileobj)
        self.global_grad_params = pickle.load(fileobj)
        self.global_size = pickle.load(fileobj)
        try:
            self.last_fit_time = pickle.load(fileobj)
        except EOFError:
            # Backward compatibility: if last_fit_time doesn't exist, set to current time
            self.last_fit_time = time.time() 