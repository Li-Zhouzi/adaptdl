import os
import sys
sys.path.insert(0, os.path.abspath("adaptdl"))

from sched.adaptdl_sched._compute_width import get_width
from sched.adaptdl_sched._configs import ARRIVAL_RATE, APPLICATIONS
from adaptdl.global_profile_state import GlobalProfileState
from adaptdl.goodput import GoodputFunction
from adaptdl.checkpoint import load_state

BUDGET = 20
CKP_PATH = './checkpoint/global-profile-state'

NUM_GPU_PER_NODE = 1 

def load_goodput_function(global_profile_state):
    goodput_dict = {}
    
    # Validate that all applications have required data
    for application in global_profile_state.global_perf_params.keys():            
        # Get application configuration
        
        perf_params = global_profile_state.global_perf_params[application]
        profile = global_profile_state.global_profiles[application]
        
        goodput_dict[application] = {}
        app_config = APPLICATIONS[application]

        # For each epoch that has grad_params
        for epoch_str in global_profile_state.global_grad_params[application].keys():
            epoch = int(epoch_str)
            grad_params = global_profile_state.global_grad_params[application][epoch_str]
            

            # Create GoodputFunction with the global profile data
            goodput_fn = GoodputFunction(perf_params, grad_params, app_config.init_batch_size)
            
            goodput_dict[application][epoch] = {}
            
            # Calculate optimal goodput for replicas 1-64
            for num_replicas in range(1, 65):
                num_nodes = max(1, (num_replicas + NUM_GPU_PER_NODE - 1) // NUM_GPU_PER_NODE)
                # Optimize for the best goodput using application-specific config
                optimal_goodput, _, _ = goodput_fn.optimize(
                    num_nodes, num_replicas, 
                    max_batch_size=app_config.max_batch_size,
                    atomic_bsz_range=(app_config.min_local_bsz, app_config.max_local_bsz),
                    accumulation=app_config.gradient_accumulation,
                    profile=profile
                )
                
                goodput_dict[application][epoch][num_replicas] = optimal_goodput
    return goodput_dict



if __name__ == '__main__':
    global_profile_state = GlobalProfileState()
    with open(CKP_PATH, "rb") as f:
        global_profile_state.load(f)
    goodput_dict = load_goodput_function(global_profile_state)
    # print(goodput_dict)
    width = get_width(goodput_dict, BUDGET)
    print(width)




