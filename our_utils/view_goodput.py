import os
import sys
import pickle

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("adaptdl"))

import matplotlib.pyplot as plt

from sched.adaptdl_sched._configs import APPLICATIONS
from adaptdl.global_profile_state import GlobalProfileState
from adaptdl.goodput import GoodputFunction

CKP_PATH = './checkpoint/global-profile-state'

NUM_GPU_PER_NODE = 1 

def load_goodput_function(global_profile_state):
    goodput_dict = {}
    
    # Validate that all applications have required data
    for application in global_profile_state.global_perf_params.keys():            
        # Get application configuration
        
        perf_params = global_profile_state.global_perf_params[application]
        profile = global_profile_state.global_profiles[application]
        # if application == "bert":
        #     print(profile)
        
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


def plot_goodput_curve(goodput_dict, application, epoch):
    """
    Plots the goodput curve for a given application and epoch.
    Args:
        goodput_dict (dict): A dictionary containing goodput values.
        application (str): The name of the application to plot.
        epoch (int): The epoch number to plot.
    """
    if application not in goodput_dict or epoch not in goodput_dict[application]:
        print(f"No data found for application '{application}' and epoch {epoch}.")
        return

    goodputs_for_epoch = goodput_dict[application][epoch]
    
    replicas = sorted(goodputs_for_epoch.keys())
    goodputs = [goodputs_for_epoch[r] for r in replicas]

    plt.figure()
    plt.plot(replicas, goodputs, marker='o', linestyle='-')
    plt.xlabel("Number of Replicas")
    plt.ylabel("Optimal Goodput")
    plt.title(f"Goodput Curve for {application} (Epoch {epoch})")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()
    
    # plot_filename = f"{application}_epoch_{epoch}_goodput.png"
    # plt.savefig(plot_filename)
    # print(f"Plot saved to {plot_filename}")
    # plt.close()


if __name__ == '__main__':
    goodput_cache_path = "./our_utils/goodput_dict.pkl"
    if os.path.exists(goodput_cache_path):
        print("Loading goodput dictionary from cache.")
        with open(goodput_cache_path, "rb") as f:
            goodput_dict = pickle.load(f)
    else:
        print("Generating goodput dictionary...")
        global_profile_state = GlobalProfileState()
        with open(CKP_PATH, "rb") as f:
            global_profile_state.load(f)
        goodput_dict = load_goodput_function(global_profile_state)
        with open(goodput_cache_path, "wb") as f:
            pickle.dump(goodput_dict, f)
        print(f"Goodput dictionary saved to {goodput_cache_path}")

    if goodput_dict:
        app_to_plot = "deepspeech2"
        epoch_to_plot = 10
        plot_goodput_curve(goodput_dict, app_to_plot, epoch_to_plot)
