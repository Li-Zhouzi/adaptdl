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

def load_goodput_function(global_profile_state, return_predictions=False):
    """
    Load goodput functions and create a goodput dictionary using global profile state data.
    This is consistent with the implementation in global_profiler.py.
    
    Args:
        global_profile_state: Global profile state containing profiles and parameters
        return_predictions: If True, also return a dict with predicted values for all points
    
    Returns:
        dict or tuple: A dictionary where goodput_dict[app][epoch][num_replica] contains the goodput value
                      If return_predictions=True, returns (goodput_dict, predicted_dict)
    """
    goodput_dict = {}
    predicted_dict = {} if return_predictions else None
    
    # Validate that all applications have required data
    for application in global_profile_state.global_perf_params.keys():            
        # Get application configuration
        if application not in APPLICATIONS:
            print(f"Warning: Application {application} not found in APPLICATIONS config, skipping...")
            continue
            
        perf_params = global_profile_state.global_perf_params[application]
        profile = global_profile_state.global_profiles[application]
        
        goodput_dict[application] = {}
        if return_predictions:
            predicted_dict[application] = {}
        app_config = APPLICATIONS[application]

        # For each epoch that has grad_params
        if application not in global_profile_state.global_grad_params:
            print(f"Warning: No grad_params found for application {application}, skipping...")
            continue
            
        for epoch in global_profile_state.global_grad_params[application].keys():
            grad_params = global_profile_state.global_grad_params[application][epoch]
            
            # Create GoodputFunction with the global profile data
            goodput_fn = GoodputFunction(perf_params, grad_params, app_config.init_batch_size)
            
            goodput_dict[application][epoch] = {}
            if return_predictions:
                predicted_dict[application][epoch] = {}
            
            # Calculate optimal goodput for replicas 1-64
            for num_replicas in range(1, 65):
                # Always calculate prediction if requested
                if return_predictions:
                    num_nodes = max(1, (num_replicas + NUM_GPU_PER_NODE - 1) // NUM_GPU_PER_NODE)
                    predicted_goodput, _, _ = goodput_fn.optimize(
                        num_nodes, num_replicas, 
                        max_batch_size=app_config.max_batch_size,
                        atomic_bsz_range=(app_config.min_local_bsz, app_config.max_local_bsz),
                        accumulation=app_config.gradient_accumulation
                    )
                    predicted_dict[application][epoch][num_replicas] = predicted_goodput
                
                # Check if we have profiled goodput for this configuration
                if (application in global_profile_state.global_goodput_profile and
                    epoch in global_profile_state.global_goodput_profile[application] and
                    num_replicas in global_profile_state.global_goodput_profile[application][epoch]):
                    # Use profiled goodput
                    optimal_goodput = global_profile_state.global_goodput_profile[application][epoch][num_replicas]
                else:
                    # Use prediction
                    if return_predictions:
                        optimal_goodput = predicted_dict[application][epoch][num_replicas]
                    else:
                        num_nodes = max(1, (num_replicas + NUM_GPU_PER_NODE - 1) // NUM_GPU_PER_NODE)
                        optimal_goodput, _, _ = goodput_fn.optimize(
                            num_nodes, num_replicas, 
                            max_batch_size=app_config.max_batch_size,
                            atomic_bsz_range=(app_config.min_local_bsz, app_config.max_local_bsz),
                            accumulation=app_config.gradient_accumulation
                        )
                
                goodput_dict[application][epoch][num_replicas] = optimal_goodput
    
    if return_predictions:
        return goodput_dict, predicted_dict
    return goodput_dict


def plot_goodput_curve(goodput_dict, application, epoch, global_profile_state=None, predicted_dict=None):
    """
    Plots the goodput curve for a given application and epoch.
    Shows both predicted and profiled values when available.
    
    Args:
        goodput_dict (dict): A dictionary containing goodput values.
        application (str): The name of the application to plot.
        epoch (int): The epoch number to plot.
        global_profile_state: Optional global profile state to show profiled data
        predicted_dict: Optional dictionary with predicted values
    """
    if application not in goodput_dict or epoch not in goodput_dict[application]:
        print(f"No data found for application '{application}' and epoch {epoch}.")
        return

    goodputs_for_epoch = goodput_dict[application][epoch]
    
    replicas = sorted(goodputs_for_epoch.keys())
    
    plt.figure(figsize=(10, 6))
    
    # Plot predicted values if available
    if predicted_dict and application in predicted_dict and epoch in predicted_dict[application]:
        predicted_goodputs = [predicted_dict[application][epoch][r] for r in replicas]
        plt.plot(replicas, predicted_goodputs, 'b-', marker='o', markersize=4, 
                label='Predicted', alpha=0.7)
    
    # Plot profiled values
    if global_profile_state and application in global_profile_state.global_goodput_profile:
        if epoch in global_profile_state.global_goodput_profile[application]:
            profiled_replicas = []
            profiled_goodputs = []
            for r in replicas:
                if r in global_profile_state.global_goodput_profile[application][epoch]:
                    profiled_replicas.append(r)
                    profiled_goodputs.append(global_profile_state.global_goodput_profile[application][epoch][r])
            if profiled_replicas:
                plt.scatter(profiled_replicas, profiled_goodputs, color='red', s=80, 
                           label='Profiled', zorder=5, edgecolors='darkred', linewidth=1.5)
    
    plt.xlabel("Number of Replicas")
    plt.ylabel("Goodput (gain/second)")
    plt.title(f"Goodput Curve for {application} (Epoch {epoch})")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def plot_multi_epoch_goodput(goodput_dict, application, epochs, global_profile_state=None, predicted_dict=None):
    """
    Plots goodput curves for multiple epochs in a single figure with subplots.
    
    Args:
        goodput_dict (dict): A dictionary containing goodput values.
        application (str): The name of the application to plot.
        epochs (list): List of epoch numbers to plot (up to 6).
        global_profile_state: Optional global profile state to show profiled data
        predicted_dict: Optional dictionary with predicted values
    """
    # Limit to 6 epochs
    epochs = epochs[:6]
    n_epochs = len(epochs)
    
    if n_epochs == 0:
        print("No epochs to plot.")
        return
    
    # Create subplot grid
    rows = 2 if n_epochs > 3 else 1
    cols = min(3, n_epochs) if n_epochs > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_epochs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # First pass: collect all goodput values to determine common y-axis scale
    all_goodputs = []
    
    for epoch in epochs:
        if application in goodput_dict and epoch in goodput_dict[application]:
            goodputs_for_epoch = goodput_dict[application][epoch]
            all_goodputs.extend(goodputs_for_epoch.values())
            
            # Also collect predicted values if available
            if predicted_dict and application in predicted_dict and epoch in predicted_dict[application]:
                all_goodputs.extend(predicted_dict[application][epoch].values())
            
            # Also collect profiled values
            if global_profile_state and application in global_profile_state.global_goodput_profile:
                if epoch in global_profile_state.global_goodput_profile[application]:
                    all_goodputs.extend(global_profile_state.global_goodput_profile[application][epoch].values())
    
    # Determine common y-axis limits
    if all_goodputs:
        y_max = max(all_goodputs) * 1.1  # Add 10% padding
    else:
        y_max = 1.0
    
    # Second pass: plot with common scale
    for idx, epoch in enumerate(epochs):
        ax = axes[idx]
        
        if application not in goodput_dict or epoch not in goodput_dict[application]:
            ax.text(0.5, 0.5, f'No data for epoch {epoch}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Epoch {epoch}')
            ax.set_ylim(0, y_max)  # Set same scale even for empty plots
            continue
        
        goodputs_for_epoch = goodput_dict[application][epoch]
        replicas = sorted(goodputs_for_epoch.keys())
        
        # Plot predicted values if available
        if predicted_dict and application in predicted_dict and epoch in predicted_dict[application]:
            predicted_goodputs = [predicted_dict[application][epoch][r] for r in replicas]
            ax.plot(replicas, predicted_goodputs, 'b-', marker='o', markersize=3, 
                   label='Predicted', alpha=0.7)
        
        # Plot profiled values
        if global_profile_state and application in global_profile_state.global_goodput_profile:
            if epoch in global_profile_state.global_goodput_profile[application]:
                profiled_replicas = []
                profiled_goodputs = []
                for r in replicas:
                    if r in global_profile_state.global_goodput_profile[application][epoch]:
                        profiled_replicas.append(r)
                        profiled_goodputs.append(global_profile_state.global_goodput_profile[application][epoch][r])
                if profiled_replicas:
                    ax.scatter(profiled_replicas, profiled_goodputs, color='red', s=60, 
                              label='Profiled', zorder=5, edgecolors='darkred', linewidth=1)
        
        ax.set_xlabel("Number of Replicas")
        ax.set_ylabel("Goodput (gain/second)")
        ax.set_title(f'Epoch {epoch}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, y_max)  # Set common y-axis scale
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_epochs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Goodput Curves for {application}", fontsize=16)
    plt.tight_layout()
    plt.show()


def print_goodput_summary(global_profile_state, predicted_dict=None):
    """
    Print a summary of profiled goodput values with predicted values side by side.
    """
    print("\n=== Profiled vs Predicted Goodput Summary ===")
    if not hasattr(global_profile_state, 'global_goodput_profile') or not global_profile_state.global_goodput_profile:
        print("No profiled goodput data found.")
        return
    
    for app in sorted(global_profile_state.global_goodput_profile.keys()):
        print(f"\nApplication: {app}")
        for epoch in sorted(global_profile_state.global_goodput_profile[app].keys()):
            print(f"  Epoch {epoch}:")
            replica_data = global_profile_state.global_goodput_profile[app][epoch]
            for replica in sorted(replica_data.keys()):
                profiled_goodput = replica_data[replica]
                # Get predicted value if available
                predicted_str = "N/A"
                ratio_str = ""
                if predicted_dict and app in predicted_dict and epoch in predicted_dict[app] and replica in predicted_dict[app][epoch]:
                    predicted_goodput = predicted_dict[app][epoch][replica]
                    predicted_str = f"{predicted_goodput:.4f}"
                    ratio = profiled_goodput / predicted_goodput if predicted_goodput > 0 else 0
                    ratio_str = f" (ratio: {ratio:.2f})"
                print(f"    Replicas={replica}: profiled={profiled_goodput:.4f}, predicted={predicted_str}{ratio_str}")
    print()


if __name__ == '__main__':
    # Always load fresh data to get the latest profiled goodput
    print("Loading global profile state...")
    global_profile_state = GlobalProfileState()
    
    if not os.path.exists(CKP_PATH):
        print(f"Checkpoint file not found at: {CKP_PATH}")
        sys.exit(1)
    
    with open(CKP_PATH, "rb") as f:
        global_profile_state.load(f)
    
    # Generate goodput dictionary with predictions first to get predicted values
    print("\nGenerating goodput dictionary with predictions...")
    goodput_dict, predicted_dict = load_goodput_function(global_profile_state, return_predictions=True)
    
    # Print summary of profiled vs predicted goodput
    print_goodput_summary(global_profile_state, predicted_dict)
    
    # Save to cache
    goodput_cache_path = "./our_utils/goodput_dict.pkl"
    with open(goodput_cache_path, "wb") as f:
        pickle.dump((goodput_dict, predicted_dict), f)
    print(f"Goodput dictionary saved to {goodput_cache_path}")
    
    # Print available applications and epochs
    print("\nAvailable applications and epochs:")
    for app in sorted(goodput_dict.keys()):
        epochs = sorted(goodput_dict[app].keys())
        print(f"  {app}: epochs {epochs}")
    
    # Plot goodput curves
    if goodput_dict:
        app_to_plot = "cifar10"
        
        # Allow command line arguments
        if len(sys.argv) > 1:
            app_to_plot = sys.argv[1]
        
        # Check if single epoch or multi-epoch plot
        if len(sys.argv) > 2:
            if sys.argv[2] == "multi":
                # Multi-epoch plot: python view_goodput.py cifar10 multi 0 1 2 3 4 5
                epochs_to_plot = []
                for i in range(3, min(len(sys.argv), 9)):  # Max 6 epochs
                    try:
                        epochs_to_plot.append(int(sys.argv[i]))
                    except ValueError:
                        pass
                if not epochs_to_plot:
                    # Default to first 6 epochs
                    available_epochs = sorted(goodput_dict[app_to_plot].keys())
                    epochs_to_plot = available_epochs[:6]
                print(f"\nPlotting multi-epoch goodput curves for {app_to_plot}, epochs: {epochs_to_plot}")
                plot_multi_epoch_goodput(goodput_dict, app_to_plot, epochs_to_plot, 
                                       global_profile_state, predicted_dict)
            else:
                # Single epoch plot
                epoch_to_plot = int(sys.argv[2])
                print(f"\nPlotting goodput curve for {app_to_plot} epoch {epoch_to_plot}")
                plot_goodput_curve(goodput_dict, app_to_plot, epoch_to_plot, 
                                 global_profile_state, predicted_dict)
        else:
            # Default: plot first 6 epochs in multi-plot
            available_epochs = sorted(goodput_dict[app_to_plot].keys())
            epochs_to_plot = available_epochs[:6]
            print(f"\nPlotting multi-epoch goodput curves for {app_to_plot}, epochs: {epochs_to_plot}")
            plot_multi_epoch_goodput(goodput_dict, app_to_plot, epochs_to_plot, 
                                   global_profile_state, predicted_dict)
