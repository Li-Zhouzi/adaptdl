import json
import sys
import os
import glob
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Application configurations from _configs.py
APPLICATIONS = {
    "bert": {"dataset_size": 97077, "max_epochs": 1},
    "cifar10": {"dataset_size": 50000, "max_epochs": 30},
    "ncf": {"dataset_size": 1000000, "max_epochs": 10},
    "imagenet": {"dataset_size": 1281167, "max_epochs": 90},
    "deepspeech2": {"dataset_size": 4074, "max_epochs": 15},
    "yolov3": {"dataset_size": 14041, "max_epochs": 50}
}

def process_single_log(log_file_path):
    """Process a single log file to extract metrics for dummy policy experiment."""
    
    # Extract expected GPU count from filename (e.g., "3gpu.txt" -> 3)
    filename = os.path.basename(log_file_path)
    expected_gpus = int(filename.replace('gpu.txt', ''))
    
    # Data structures to track metrics
    epochs = {}  # epoch_num -> epoch_info
    startup_metrics = {
        'node_preparation_time': None,      # Time from total nodes to ready nodes
        'job_allocation_change_time': None, # Time from ready nodes to allocation
        'image_building_time': None,        # Time from allocation to pod status normal
        'rescaling_time': None,             # Time from pod status normal to progress increase
        'first_timestamp': None,
        'total_nodes_timestamp': None,     # When total nodes reach expected
        'ready_nodes_timestamp': None,     # When ready nodes reach expected
        'allocation_timestamp': None,      # When allocation reaches expected
        'pod_normal_timestamp': None,      # When pod status becomes normal
        'progress_growth_timestamp': None, # When progress starts growing after pod normal
        'allocation_reached': False,
        'pod_normal_reached': False,
        'last_progress': None
    }
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Define threshold for monitor gap detection (e.g., 100 seconds)
    MONITOR_GAP_THRESHOLD = 100  # seconds
    monitor_gaps = []  # List of (start_time, end_time) tuples
    
    # Process each log entry
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        
        # Track first timestamp
        if startup_metrics['first_timestamp'] is None:
            startup_metrics['first_timestamp'] = timestamp
            
        # Check for monitor gaps
        if i > 0:
            prev_entry = json.loads(lines[i-1].strip())
            prev_timestamp = prev_entry['timestamp']
            time_diff = timestamp - prev_timestamp
            
            if time_diff > MONITOR_GAP_THRESHOLD:
                monitor_gaps.append((prev_timestamp, timestamp))
        
        # Check cluster nodes
        cluster_nodes = log_entry.get('cluster_nodes', {})
        total_nodes = cluster_nodes.get('total', 0)
        ready_nodes = cluster_nodes.get('ready', 0)
        
        # Track when total nodes reach expected count
        if total_nodes >= expected_gpus and startup_metrics['total_nodes_timestamp'] is None:
            startup_metrics['total_nodes_timestamp'] = timestamp
            
        # Track when ready nodes reach expected count  
        if ready_nodes >= expected_gpus and startup_metrics['ready_nodes_timestamp'] is None:
            startup_metrics['ready_nodes_timestamp'] = timestamp
        
        # Process job information
        for job in log_entry.get('submitted_jobs', []):
            job_name = job['name']
            epoch = job['epoch']
            allocation = job.get('allocation', [])
            progress = job.get('progress', 0)
            pod_status = job.get('pod_status', '')
            
            # Track when allocation reaches expected count
            if len(allocation) == expected_gpus and startup_metrics['allocation_timestamp'] is None:
                startup_metrics['allocation_timestamp'] = timestamp
                startup_metrics['allocation_reached'] = True
                
            # Track when pod status becomes normal after allocation
            if (startup_metrics['allocation_reached'] and 
                startup_metrics['pod_normal_timestamp'] is None and
                'normal' in pod_status.lower()):
                startup_metrics['pod_normal_timestamp'] = timestamp
                startup_metrics['pod_normal_reached'] = True
                
            # Track when progress starts growing after pod status is normal
            if (startup_metrics['pod_normal_reached'] and 
                startup_metrics['progress_growth_timestamp'] is None and
                progress is not None and 
                startup_metrics['last_progress'] is not None and
                progress > startup_metrics['last_progress']):
                startup_metrics['progress_growth_timestamp'] = timestamp
                
            # Update last progress
            if progress is not None:
                startup_metrics['last_progress'] = progress
            
            # Initialize epoch tracking
            if epoch not in epochs:
                epochs[epoch] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'progress_history': [],
                    'allocation_history': [],
                    'is_starting': False,  # Mark if epoch didn't run with full GPUs
                    'idle_time_at_end': 0,
                    'duration': 0,
                    'max_allocation': 0,
                    'has_monitor_gap': False  # Mark if epoch was affected by monitor gap
                }
            
            epoch_info = epochs[epoch]
            epoch_info['last_seen'] = timestamp
            epoch_info['progress_history'].append((timestamp, progress))
            epoch_info['allocation_history'].append((timestamp, len(allocation)))
            epoch_info['max_allocation'] = max(epoch_info['max_allocation'], len(allocation))
            
            # Check if this epoch ever ran with less than expected GPUs
            if len(allocation) < expected_gpus:
                epoch_info['is_starting'] = True
                
    # Mark epochs that were affected by monitor gaps
    for epoch_num, epoch_info in epochs.items():
        for gap_start, gap_end in monitor_gaps:
            # If the epoch was active during a monitor gap
            if epoch_info['first_seen'] <= gap_end and epoch_info['last_seen'] >= gap_start:
                epoch_info['has_monitor_gap'] = True
                break
    
    # Calculate startup timing metrics
    if startup_metrics['total_nodes_timestamp'] and startup_metrics['ready_nodes_timestamp']:
        startup_metrics['node_preparation_time'] = (
            startup_metrics['ready_nodes_timestamp'] - startup_metrics['total_nodes_timestamp']
        )
    
    if startup_metrics['ready_nodes_timestamp'] and startup_metrics['allocation_timestamp']:
        startup_metrics['job_allocation_change_time'] = (
            startup_metrics['allocation_timestamp'] - startup_metrics['ready_nodes_timestamp']
        )
    
    if startup_metrics['allocation_timestamp'] and startup_metrics['pod_normal_timestamp']:
        startup_metrics['image_building_time'] = (
            startup_metrics['pod_normal_timestamp'] - startup_metrics['allocation_timestamp']
        )
        
    if startup_metrics['pod_normal_timestamp'] and startup_metrics['progress_growth_timestamp']:
        startup_metrics['rescaling_time'] = (
            startup_metrics['progress_growth_timestamp'] - startup_metrics['pod_normal_timestamp']
        )
    
    # Calculate epoch durations
    for epoch_num, epoch_info in epochs.items():
        epoch_info['duration'] = epoch_info['last_seen'] - epoch_info['first_seen']
    
    return epochs, startup_metrics, expected_gpus

def calculate_goodput(epochs, app_name, expected_gpus):
    """Calculate goodput for normal epochs."""
    if app_name not in APPLICATIONS:
        return None
        
    dataset_size = APPLICATIONS[app_name]["dataset_size"]
    goodputs = {}
    
    for epoch_num, epoch_info in epochs.items():
        # Only calculate goodput for normal epochs
        if not epoch_info['is_starting'] and not epoch_info['has_monitor_gap'] and epoch_info['duration'] > 0:
            goodput = dataset_size / epoch_info['duration']
            goodputs[epoch_num] = goodput
    
    return goodputs

def build_goodput_dict(base_dir):
    """Build goodput dictionary for all applications and GPU configurations."""
    goodput_dict = {}
    startup_times_dict = defaultdict(list)
    
    # Get all application directories
    app_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for app_name in app_dirs:
        if app_name not in APPLICATIONS:
            print(f"Warning: Application {app_name} not found in APPLICATIONS config")
            continue
            
        goodput_dict[app_name] = {}
        app_dir = os.path.join(base_dir, app_name)
        
        # Get all GPU log files
        log_files = glob.glob(os.path.join(app_dir, "*gpu.txt"))
        
        for log_file in sorted(log_files):
            epochs, startup_metrics, expected_gpus = process_single_log(log_file)
            goodputs = calculate_goodput(epochs, app_name, expected_gpus)
            
            # Store startup metrics
            startup_times_dict[app_name].append({
                'gpus': expected_gpus,
                'metrics': startup_metrics
            })
            
            # Store goodput values by epoch
            if goodputs:
                for epoch_num, goodput_value in goodputs.items():
                    if epoch_num not in goodput_dict[app_name]:
                        goodput_dict[app_name][epoch_num] = {}
                    goodput_dict[app_name][epoch_num][expected_gpus] = goodput_value
    
    # Handle missing epochs with extrapolation
    for app_name in goodput_dict:
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        
        # Find which GPU counts we have data for
        all_gpu_counts = set()
        for epoch_data in goodput_dict[app_name].values():
            all_gpu_counts.update(epoch_data.keys())
        all_gpu_counts = sorted(list(all_gpu_counts))
        
        if not all_gpu_counts:
            continue
            
        # Fill missing epochs
        for epoch in range(max_epochs):
            if epoch not in goodput_dict[app_name]:
                goodput_dict[app_name][epoch] = {}
            
            # For each GPU count, fill missing values
            for gpu_count in all_gpu_counts:
                if gpu_count not in goodput_dict[app_name][epoch]:
                    # Find nearest epoch with data
                    found = False
                    # First try next epochs
                    for next_epoch in range(epoch + 1, max_epochs):
                        if (next_epoch in goodput_dict[app_name] and 
                            gpu_count in goodput_dict[app_name][next_epoch]):
                            goodput_dict[app_name][epoch][gpu_count] = goodput_dict[app_name][next_epoch][gpu_count]
                            found = True
                            break
                    
                    # If not found, try previous epochs
                    if not found:
                        for prev_epoch in range(epoch - 1, -1, -1):
                            if (prev_epoch in goodput_dict[app_name] and 
                                gpu_count in goodput_dict[app_name][prev_epoch]):
                                goodput_dict[app_name][epoch][gpu_count] = goodput_dict[app_name][prev_epoch][gpu_count]
                                break
    
    return goodput_dict, startup_times_dict

def plot_goodput_functions(goodput_dict):
    """Create figure with 6 subplots showing goodput functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define which epochs to plot for each application
    plot_configs = [
        ("cifar10", 1),
        ("cifar10", 5),
        ("cifar10", 10),
        ("deepspeech2", 1),
        ("deepspeech2", 5),
        ("deepspeech2", 10)
    ]
    
    for idx, (app_name, epoch) in enumerate(plot_configs):
        ax = axes[idx]
        
        if app_name in goodput_dict and epoch in goodput_dict[app_name]:
            # Get data points
            gpu_counts = sorted(goodput_dict[app_name][epoch].keys())
            goodputs = [goodput_dict[app_name][epoch][gpu] for gpu in gpu_counts]
            
            if gpu_counts and goodputs:
                # Plot data points
                ax.plot(gpu_counts, goodputs, 'o-', markersize=8, linewidth=2)
                
                # Interpolate for smooth curve
                if len(gpu_counts) > 2:
                    gpu_range = np.linspace(min(gpu_counts), max(gpu_counts), 100)
                    f = interp1d(gpu_counts, goodputs, kind='linear', fill_value='extrapolate')
                    ax.plot(gpu_range, f(gpu_range), '--', alpha=0.5)
                
                ax.set_xlabel('Number of GPUs')
                ax.set_ylabel('Goodput (samples/sec)')
                ax.set_title(f'{app_name} - Epoch {epoch}')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, max(gpu_counts) + 1))
            else:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{app_name} - Epoch {epoch}')
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(f'{app_name} - Epoch {epoch}')
    
    plt.tight_layout()
    plt.savefig('goodput_functions.png', dpi=300)
    plt.close()

def print_startup_times_table(startup_times_dict):
    """Print startup timing table for all applications."""
    print("\n" + "="*100)
    print("STARTUP TIMING SUMMARY - ALL APPLICATIONS")
    print("="*100)
    
    for app_name in sorted(startup_times_dict.keys()):
        print(f"\n{app_name.upper()}")
        print("-"*80)
        print(f"{'GPUs':<6} {'Node Preparation':<18} {'Job Allocation':<18} {'Image Building':<18} {'Rescaling':<18}")
        print(f"{'':6} {'Time (s)':<18} {'Change Time (s)':<18} {'Time (s)':<18} {'Time (s)':<18}")
        print("-"*80)
        
        for item in sorted(startup_times_dict[app_name], key=lambda x: x['gpus']):
            gpus = item['gpus']
            m = item['metrics']
            
            node_prep = f"{m['node_preparation_time']:.2f}" if m['node_preparation_time'] is not None else "N/A"
            job_alloc = f"{m['job_allocation_change_time']:.2f}" if m['job_allocation_change_time'] is not None else "N/A"
            image_build = f"{m['image_building_time']:.2f}" if m['image_building_time'] is not None else "N/A"
            rescaling = f"{m['rescaling_time']:.2f}" if m['rescaling_time'] is not None else "N/A"
            
            print(f"{gpus:<6} {node_prep:<18} {job_alloc:<18} {image_build:<18} {rescaling:<18}")

def print_goodput_functions(goodput_dict):
    """Print goodput functions in a copy-paste friendly format."""
    print("\n" + "="*80)
    print("GOODPUT FUNCTIONS (Copy-Paste Format)")
    print("="*80)
    
    # Print as Python dictionary
    print("\n# Python Dictionary Format:")
    print("goodput_functions = {")
    for app_name in sorted(goodput_dict.keys()):
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        print(f"    '{app_name}': {{")
        for epoch in sorted(goodput_dict[app_name].keys()):
            if epoch < max_epochs:  # Only include epochs up to max_epochs
                gpu_goodputs = goodput_dict[app_name][epoch]
                if gpu_goodputs:
                    print(f"        {epoch}: {dict(sorted(gpu_goodputs.items()))},")
        print("    },")
    print("}")
    
    # Print as table format for each application
    print("\n# Table Format:")
    for app_name in sorted(goodput_dict.keys()):
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        print(f"\n{app_name.upper()} Goodput Table (max_epochs={max_epochs}):")
        print("-" * 60)
        
        # Get all epochs and GPU counts
        all_epochs = [e for e in sorted(goodput_dict[app_name].keys()) if e < max_epochs]
        all_gpus = set()
        for epoch in all_epochs:
            all_gpus.update(goodput_dict[app_name][epoch].keys())
        all_gpus = sorted(list(all_gpus))
        
        if not all_gpus:
            continue
            
        # Print header
        header = "Epoch\\GPUs"
        for gpu in all_gpus:
            header += f"\t{gpu}"
        print(header)
        
        # Print data
        for epoch in all_epochs:
            row = f"{epoch}"
            for gpu in all_gpus:
                if gpu in goodput_dict[app_name][epoch]:
                    row += f"\t{goodput_dict[app_name][epoch][gpu]:.2f}"
                else:
                    row += "\t-"
            print(row)
    
    # Print as CSV format
    print("\n# CSV Format:")
    for app_name in sorted(goodput_dict.keys()):
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        print(f"\n{app_name}_goodput.csv (max_epochs={max_epochs}):")
        print("epoch,num_gpus,goodput")
        for epoch in sorted(goodput_dict[app_name].keys()):
            if epoch < max_epochs:  # Only include epochs up to max_epochs
                for gpu, goodput in sorted(goodput_dict[app_name][epoch].items()):
                    print(f"{epoch},{gpu},{goodput:.2f}")

def main():
    base_dir = "./experiment_results/dummy"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found")
        sys.exit(1)
    
    print("Processing all application logs...")
    goodput_dict, startup_times_dict = build_goodput_dict(base_dir)
    
    # Print goodput functions in copy-paste format
    print_goodput_functions(goodput_dict)
    
    # Print startup times table
    print_startup_times_table(startup_times_dict)
    
    # Create goodput plots
    print("\nCreating goodput function plots...")
    plot_goodput_functions(goodput_dict)
    print("Plots saved to goodput_functions.png")

if __name__ == "__main__":
    main()