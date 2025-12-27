# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import json
import pickle
import time
import requests
import logging
import os

import numpy as np

import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.goodput import GoodputFunction, fit_perf_params
from adaptdl.sched_hints import SCHED_HINTS, PERF_PARAMS, post_sched_hints


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def report_train_metrics(epoch, loss, **kwargs):
    if adaptdl.env.replica_rank() > 0:
        return
    with open(adaptdl.env.checkpoint_path() + "/train.txt", "a") as f:
        json.dump(dict(
            time=time.time(),
            progress=get_progress(),
            epoch=epoch,
            loss=loss,
            **kwargs
        ), f)
        f.write("\n")


def report_valid_metrics(epoch, loss, **kwargs):
    if adaptdl.env.replica_rank() > 0:
        return
    with open(adaptdl.env.checkpoint_path() + "/valid.txt", "a") as f:
        json.dump(dict(
            time=time.time(),
            progress=get_progress(),
            epoch=epoch,
            loss=loss,
            **kwargs
        ), f)
        f.write("\n")


def profile_step_start(atomic_bsz):
    state = _metrics_state()
    state.atomic_bsz = atomic_bsz
    state.step_start = time.time()
    state.sync_time = 0.0


def profile_sync_time(sync_time):
    _metrics_state().sync_time += sync_time


_PREV_REPORT = None
_PREV_PROGRESS = None      # Track previous progress value for rate calculation
_PREV_REPORT_TIME = None   # Track previous report timestamp
_STEP_COUNT = 0            # Count optimizer steps between reports
_LAST_GAIN = None          # Track most recent gain value
_LAST_GRAD_PARAMS = None   # Track most recent local grad_params


def profile_step_commit(epoch, batch_size, accumulation_step=False):
    global _PREV_REPORT, _STEP_COUNT
    state = _metrics_state()
    step_time = time.time() - state.step_start
    num_nodes = adaptdl.env.num_nodes()
    num_replicas = adaptdl.env.num_replicas()
    key = (num_nodes, num_replicas, state.atomic_bsz)

    # Don't update local profile, but report to global profiler
    profile_data = {
        "accumulation_step": accumulation_step,
        "step_time": step_time,
        "sync_time": state.sync_time,
        "num_nodes": num_nodes,
        "num_replicas": num_replicas,
        "atomic_bsz": state.atomic_bsz,
        "epoch": epoch,
    }
    
    # Include current grad_params if available
    if state.grad_params is not None:
        profile_data["grad_norm_sqr"] = state.grad_params[0]
        profile_data["grad_variance"] = state.grad_params[1]
    
    # Include init_batch_size for proper goodput calculation
    if state.init_batch_size is not None:
        profile_data["init_batch_size"] = state.init_batch_size
    
    if adaptdl.env.replica_rank() == 0:
        _report_global_profile(profile_data)

    _update_grad_params_from_global_profiler(epoch)

    # Start here for job wise profile
    # if accumulation_step:
    #     state.profile[key]["accum_step_time"] += step_time
    #     state.profile[key]["accum_count"] += 1
    # else:
    #     state.profile[key]["optim_step_time"] += step_time
    #     state.profile[key]["optim_sync_time"] += state.sync_time
    #     state.profile[key]["optim_count"] += 1
    # del state.atomic_bsz
    # del state.step_start
    # del state.sync_time
    # End here for job wise profile


    if not accumulation_step:
        _STEP_COUNT += 1  # Count optimizer steps (not accumulation steps)

        if _PREV_REPORT is None:
            _PREV_REPORT = time.time()
        if adaptdl.env.replica_rank() == 0 and time.time() - _PREV_REPORT > 1:
            # _fit_perf_params() # if type wise profile, comment this line
            _report_sched_hints(epoch, batch_size)
            _PREV_REPORT = time.time()


_GRAD_PARAM_DICT = {}


def update_grad_params(edp_key, grad_norm_sqr, grad_variance):
    return # for now, skip the whole grad params update. Always use global grad params.
    # global _GRAD_PARAM_DICT
    # _GRAD_PARAM_DICT[edp_key] = np.asarray([grad_norm_sqr, grad_variance])
    # grad_params = sum(_GRAD_PARAM_DICT.values())
    # _metrics_state().grad_params = (grad_params[0], grad_params[1])

def update_progress(progress):
    _metrics_state().progress = progress


def update_diagnostic_metrics(gain, grad_params):
    """Update diagnostic metrics for progress growth analysis."""
    global _LAST_GAIN, _LAST_GRAD_PARAMS
    _LAST_GAIN = gain
    _LAST_GRAD_PARAMS = grad_params


def get_progress():
    return _metrics_state().progress


def set_batch_size(init_batch_size, max_batch_size, local_bsz_bounds,
                   gradient_accumulation):
    state = _metrics_state()
    state.init_batch_size = init_batch_size
    state.max_batch_size = max_batch_size
    state.local_bsz_bounds = local_bsz_bounds
    state.gradient_accumulation = gradient_accumulation


def get_goodput_fn():
    state = _metrics_state()
    # print(state.grad_params, state.perf_params)
    if state.grad_params is None or state.perf_params is None:
        return None
    return GoodputFunction(state.perf_params, state.grad_params,
                           state.init_batch_size)


def _fit_perf_params():
    state = _metrics_state()
    profile = {k: v for k, v in state.profile.items() if v.get("optim_count")}
    # Convert profile into numpy arrays.
    num_nodes, num_replicas, atomic_bsz = (np.array(k) for k in zip(*profile))
    accum_step_time = np.array([v.get("accum_step_time", 0.0)
                                for v in profile.values()])
    accum_count = np.array([v.get("accum_count", 0) for v in profile.values()])
    optim_step_time = np.array([v.get("optim_step_time", 0.0)
                                for v in profile.values()])
    optim_sync_time = np.array([v.get("optim_sync_time", 0.0)
                                for v in profile.values()])
    optim_count = np.array([v.get("optim_count", 0) for v in profile.values()])
    assert np.all(optim_count > 0)
    # Non-sync time during optimization steps should be approximately equal to
    # accumulation step time, combine those data points.
    assert np.all(optim_step_time >= optim_sync_time)
    accum_step_time += optim_step_time - optim_sync_time
    accum_count += optim_count
    accum_step_time /= accum_count
    optim_step_time /= optim_count
    state.perf_params = fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                                        accum_step_time, optim_step_time)


def _report_sched_hints(epoch, batch_size):
    global _PREV_PROGRESS, _PREV_REPORT_TIME, _STEP_COUNT, _LAST_GAIN, _LAST_GRAD_PARAMS

    assert adaptdl.env.replica_rank() == 0
    state = _metrics_state()
    
    # Scheduling hints
    sched_hints = SCHED_HINTS.copy()
    
    # Only add perfParams if available, otherwise skip this entry
    if state.perf_params is not None:
        sched_hints["perfParams"] = {k: v for (k, v) in
                                     zip(PERF_PARAMS.keys(),
                                     state.perf_params)}
    
    sched_hints["maxBatchSize"] = state.max_batch_size
    sched_hints["localBszBounds"] = state.local_bsz_bounds
    sched_hints["initBatchSize"] = state.init_batch_size
    if state.grad_params:
        sched_hints["gradParams"] = {}
        sched_hints["gradParams"]["norm"] = state.grad_params[0]
        sched_hints["gradParams"]["var"] = state.grad_params[1]
    sched_hints["maxProfiledReplicas"] = max(key[1] for key in state.profile)
    sched_hints["gradientAccumulation"] = state.gradient_accumulation
    sched_hints["epoch"] = epoch
    sched_hints["batchSize"] = batch_size
    sched_hints["progress"] = state.progress

    # Compute diagnostic metrics
    current_time = time.time()

    # Compute progress rate (progress/second)
    if _PREV_PROGRESS is not None and _PREV_REPORT_TIME is not None:
        time_delta = current_time - _PREV_REPORT_TIME
        progress_delta = state.progress - _PREV_PROGRESS
        if time_delta > 0:
            sched_hints["progressRate"] = progress_delta / time_delta
            sched_hints["throughput"] = _STEP_COUNT / time_delta
            sched_hints["stepTime"] = time_delta / _STEP_COUNT if _STEP_COUNT > 0 else None
        else:
            sched_hints["progressRate"] = 0.0
            sched_hints["throughput"] = 0.0
            sched_hints["stepTime"] = None
    else:
        sched_hints["progressRate"] = None
        sched_hints["throughput"] = None
        sched_hints["stepTime"] = None

    # Add current gain and local grad_params
    sched_hints["currentGain"] = _LAST_GAIN
    if _LAST_GRAD_PARAMS is not None:
        sched_hints["localGradParams"] = {
            "sqr": _LAST_GRAD_PARAMS[0],  # sqr_avg
            "var": _LAST_GRAD_PARAMS[1]   # var_avg
        }

    # Print diagnostics to console for immediate visibility
    progress_rate_str = f"{sched_hints['progressRate']:.4f}" if sched_hints.get('progressRate') is not None else 'N/A'
    throughput_str = f"{sched_hints['throughput']:.2f}" if sched_hints.get('throughput') is not None else 'N/A'
    gain_str = f"{_LAST_GAIN:.4f}" if _LAST_GAIN is not None else 'N/A'
    sqr_str = f"{_LAST_GRAD_PARAMS[0]:.2e}" if _LAST_GRAD_PARAMS is not None else 'N/A'
    var_str = f"{_LAST_GRAD_PARAMS[1]:.2e}" if _LAST_GRAD_PARAMS is not None else 'N/A'

    print(f"[DIAGNOSTIC] Job: {adaptdl.env.job_id()}, Epoch: {epoch}, Progress: {state.progress:.2f}, "
          f"ProgressRate: {progress_rate_str}/s, "
          f"Throughput: {throughput_str} steps/s, "
          f"Gain: {gain_str}, "
          f"LocalGradParams: sqr={sqr_str}, var={var_str}")

    # Update tracking variables for next iteration
    _PREV_PROGRESS = state.progress
    _PREV_REPORT_TIME = current_time
    _STEP_COUNT = 0  # Reset counter after reporting

    post_sched_hints(sched_hints, adaptdl.env.job_id())


def _report_global_profile(profile_data):
    """Report profile data to the global profiler."""
    application = adaptdl.env.job_id().split("-")[0].split("/")[-1]
    post_global_profile(profile_data, application)


class _MetricsState(adaptdl.checkpoint.State):
    def __init__(self):
        super().__init__("adaptdl-metrics")
        self.profile = collections.defaultdict(collections.Counter)
        self.perf_params = None
        self.grad_params = None
        self.init_batch_size = None
        self.max_batch_size = None
        self.local_bsz_bounds = None
        self.gradient_accumulation = False
        self.progress = 0.0  # Progress in scale-invariant iterations.
        self.last_fetch_global_time = 0.0  # Track when we last fetched global profiler state


    def save(self, fileobj):
        pickle.dump(self.profile, fileobj)
        pickle.dump(self.perf_params, fileobj)
        pickle.dump(self.grad_params, fileobj)
        pickle.dump(self.init_batch_size, fileobj)
        pickle.dump(self.max_batch_size, fileobj)
        pickle.dump(self.local_bsz_bounds, fileobj)
        pickle.dump(self.gradient_accumulation, fileobj)
        pickle.dump(self.progress, fileobj)
        pickle.dump(self.last_fetch_global_time, fileobj)

        
    def load(self, fileobj):
        self.profile = pickle.load(fileobj)
        self.perf_params = pickle.load(fileobj)
        self.grad_params = pickle.load(fileobj)
        self.init_batch_size = pickle.load(fileobj)
        self.max_batch_size = pickle.load(fileobj)
        self.local_bsz_bounds = pickle.load(fileobj)
        self.gradient_accumulation = pickle.load(fileobj)
        self.progress = pickle.load(fileobj)
        # Handle backward compatibility - if last_fetch_global_time doesn't exist in checkpoint
        try:
            self.last_fetch_global_time = pickle.load(fileobj)
        except EOFError:
            self.last_fetch_global_time = 0.0

def _update_grad_params_from_global_profiler(epoch):
    """Update grad_params from global profiler state for the current application and epoch."""
    # Check if global profiler state exists, if not retrieve it
    if not hasattr(_load_global_profiler_state, '_GLOBAL_PROFILE_STATE') or \
       _load_global_profiler_state._GLOBAL_PROFILE_STATE is None:
        _load_global_profiler_state(_metrics_state())
    
    global_state = _load_global_profiler_state._GLOBAL_PROFILE_STATE
    if global_state is None:
        raise Exception("Global profile state not available")
        return
    
    # Get application from job_id
    application = adaptdl.env.job_id().split("-")[0].split("/")[-1]
    
    # Check if global_grad_params exists in the global state
    if not hasattr(global_state, 'global_grad_params'):
        raise Exception("global_grad_params not found in global state")
        return
    
    # Look for grad_params for this application and epoch
    if application in global_state.global_grad_params:
        app_grad_params = global_state.global_grad_params[application]
        if epoch in app_grad_params:
            # Update the metrics state with the grad_params for this epoch
            grad_params = app_grad_params[epoch]
            _metrics_state().grad_params = (grad_params[0], grad_params[1])
            # print(f"Updated grad_params for application {application}, epoch {epoch}: {grad_params}")
        else:
            raise Exception(f"No grad_params found for application {application}, epoch {epoch}")
    else:
        raise Exception(f"No grad_params found for application {application}")

def _metrics_state():
    global _METRICS_STATE
    if _METRICS_STATE is None:
        _METRICS_STATE = _MetricsState()
        print("loading state")
        adaptdl.checkpoint.load_state(_METRICS_STATE)
        print("retrieving global profiler state")
        _load_global_profiler_state(_METRICS_STATE)

    # else:
        # Check if we need to refresh global profiler state (every 60 seconds)
        # current_time = time.time()
        # if current_time - _METRICS_STATE.last_fetch_global_time > 60.0:
        #     print("retrieving global profiler state")
        #     _load_global_profiler_state(_METRICS_STATE)
        #     _METRICS_STATE.last_fetch_global_time = current_time
    
    return _METRICS_STATE


def _load_global_profiler_state(metrics_state):
    """Helper function to load global profiler state."""
    print("Attempting to import GlobalProfileState...")
    from adaptdl.global_profile_state import GlobalProfileState
    print("Successfully imported GlobalProfileState")
    
    # Use a singleton pattern for GlobalProfileState to avoid registration conflicts
    global _GLOBAL_PROFILE_STATE
    if not hasattr(_load_global_profiler_state, '_GLOBAL_PROFILE_STATE'):
        _load_global_profiler_state._GLOBAL_PROFILE_STATE = None
    
    if _load_global_profiler_state._GLOBAL_PROFILE_STATE is None:
        _load_global_profiler_state._GLOBAL_PROFILE_STATE = GlobalProfileState()
        print("Created GlobalProfileState instance")
    
    global_state = _load_global_profiler_state._GLOBAL_PROFILE_STATE
    
    # Try to load the state from the global checkpoint path
    import os
    global_checkpoint_path = "/pollux/global-checkpoint"
    if os.path.exists(global_checkpoint_path):
        try:
            import pickle
            # Get the state name from the global checkpoint dictionary
            from adaptdl.checkpoint import _STATES_TO_NAMES
            state_name = _STATES_TO_NAMES.get(global_state, "global-profile-state")
            checkpoint_file = os.path.join(global_checkpoint_path, state_name)
            print(f"Loading from global checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "rb") as f:
                global_state.load(f)
            print("Loaded global profiler state from global checkpoint")
        except Exception as e:
            print(f"Failed to load from global checkpoint: {e}")
            return
    else:
        raise Exception("Global checkpoint path not found")
    
    # Get application from job_id
    application = adaptdl.env.job_id().split("-")[0].split("/")[-1]
    print("application: ", application)
    print("keys in global_profile_state: ", global_state.global_profiles.keys())
    
    # Overwrite perf_params if available for this application
    if application in global_state.global_perf_params:
        metrics_state.perf_params = global_state.global_perf_params[application]
        print("Loaded global perf_params for application ", application)
    
    # Overwrite profile if available for this application
    if application in global_state.global_profiles:
        # Convert the global profile back to defaultdict structure
        global_profile = global_state.global_profiles[application]
        metrics_state.profile = collections.defaultdict(collections.Counter)
        for key, profile_data in global_profile.items():
            metrics_state.profile[key] = profile_data
        print("Loaded global profile for application ", application)
        print("length of profile: ", len(metrics_state.profile))

    # Load grad_params for epoch 0 during initialization
    from adaptdl.torch.epoch import current_epoch
    if current_epoch() is None:
        epoch = 0
    else:
        epoch = current_epoch()
    _update_grad_params_from_global_profiler(epoch)


_METRICS_STATE = None


def post_global_profile(profile_data, application="default"):
    """
    Post profile data to the global profiler.
    
    Args:
        profile_data (dict): Profile data with keys as (num_nodes, num_replicas, atomic_bsz) tuples
        application (str): Application identifier for the global profiler
    """
    url = adaptdl.env.global_profiler_url()
    # print("sent profile to global profiler, url: ", url, application, profile_data)
    if not url or url == "":
        return  # skip if global profiler URL is not set
    
    headers = {"Content-Type": "application/json"}
    try:
        # Prepare the data in the format expected by the global profiler
        data = {
            "application": application,
            **profile_data
        }
        
        response = requests.post(url=f"{url}/profile",
                               data=json.dumps(data),
                               headers=headers)
        if response.status_code != 200:
            LOG.warning(f"Global profiler returned {response.status_code}")
    except Exception as e:
        LOG.warning(f"Failed to post to global profiler: {e}")
