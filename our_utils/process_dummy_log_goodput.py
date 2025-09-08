import json
import sys
import os
import glob
import math
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Configuration
NUM_GPU_PER_NODE = 4

# Application configurations from _configs.py
APPLICATIONS = {
    "bert": {"dataset_size": 97077, "max_epochs": 1},
    "cifar10": {"dataset_size": 50000, "max_epochs": 100},
    "ncf": {"dataset_size": 1000000, "max_epochs": 10},
    "imagenet": {"dataset_size": 1281167, "max_epochs": 90},
    "deepspeech2": {"dataset_size": 4074, "max_epochs": 15},
    "yolov3": {"dataset_size": 14041, "max_epochs": 50}
}

def process_single_log(log_file_path):
    """Process a single log file to extract metrics for dummy policy experiment."""
    
    # Extract GPU configuration from filename
    filename = os.path.basename(log_file_path)
    
    # Handle both single GPU files (e.g., "3gpu.txt", "12gpu.txt") and parallel files (e.g., "parallel_1_2_4gpu.txt")
    if filename.startswith('parallel_'):
        # Extract list of GPU counts from parallel filename
        gpu_str = filename.replace('parallel_', '').replace('gpu.txt', '')
        expected_gpus_list = [int(x) for x in gpu_str.split('_')]
        is_parallel = True
    else:
        # Single GPU count - extract the number before "gpu.txt"
        import re
        match = re.match(r'(\d+)gpu\.txt', filename)
        if match:
            expected_gpus = int(match.group(1))
            expected_gpus_list = [expected_gpus]
            is_parallel = False
        else:
            print(f"Warning: Could not parse GPU count from filename: {filename}")
            return {}
    
    # Data structures to track metrics per job
    jobs_data = {}  # job_name -> {epochs: {}, startup_metrics: {}, expected_gpus: int}
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Define threshold for monitor gap detection (e.g., 100 seconds)
    MONITOR_GAP_THRESHOLD = 100  # seconds
    monitor_gaps = []  # List of (start_time, end_time) tuples
    
    # Track global first timestamp
    global_first_timestamp = None
    
    # Process each log entry
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        
        # Track first timestamp globally
        if global_first_timestamp is None:
            global_first_timestamp = timestamp
            
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
        
        # Process job information
        for job in log_entry.get('submitted_jobs', []):
            job_name = job['name']
            epoch = job['epoch']
            allocation = job.get('allocation', [])
            progress = job.get('progress', 0)
            pod_status = job.get('pod_status', '')
            
            # Initialize job data if not exists
            if job_name not in jobs_data:
                # Determine expected GPUs for this job
                if is_parallel:
                    # In parallel mode, try to match job allocation to expected GPU counts
                    job_expected_gpus = None
                    for gpu_count in expected_gpus_list:
                        # Find the first stable allocation that matches
                        if len(allocation) == gpu_count:
                            job_expected_gpus = gpu_count
                            break
                    # If no match yet, store allocation for later determination
                    if job_expected_gpus is None:
                        job_expected_gpus = len(allocation) if allocation else None
                else:
                    # In single mode, all jobs should have the same expected GPUs
                    job_expected_gpus = expected_gpus_list[0]
                
                jobs_data[job_name] = {
                    'epochs': {},
                    'expected_gpus': job_expected_gpus,
                    'startup_metrics': {
                        'first_timestamp': timestamp,
                        'allocation_timestamp': None,
                        'pod_normal_timestamp': None,
                        'progress_growth_timestamp': None,
                        'allocation_reached': False,
                        'pod_normal_reached': False,
                        'last_progress': None
                    }
                }
            
            job_info = jobs_data[job_name]
            epochs = job_info['epochs']
            startup_metrics = job_info['startup_metrics']
            
            # Update expected GPUs if we have a stable allocation
            if job_info['expected_gpus'] is None and len(allocation) > 0:
                job_info['expected_gpus'] = len(allocation)
            elif job_info['expected_gpus'] != len(allocation) and len(allocation) > 0:
                # Check if this is a more stable allocation
                if len(allocation) in expected_gpus_list:
                    job_info['expected_gpus'] = len(allocation)
            
            job_expected_gpus = job_info['expected_gpus']
            
            # Track when allocation reaches expected count
            if (job_expected_gpus and len(allocation) == job_expected_gpus and 
                startup_metrics['allocation_timestamp'] is None):
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
                    'is_affected': False,  # Mark if epoch is affected by scaling/gaps/wrong allocation
                    'idle_time_at_end': 0,
                    'duration': 0,
                    'max_allocation': 0,
                    'reached_expected': False,  # Track if epoch ever reached expected GPUs
                    'has_monitor_gap': False,  # Mark if epoch was affected by monitor gap
                    'has_wrong_allocation': False,  # Mark if epoch had wrong GPU count
                    'has_rescaling': False,  # Mark if epoch had rescaling during execution
                    'stable_start_time': None,  # When epoch reached stable allocation
                    'stable_allocation_duration': 0  # Duration with correct allocation
                }
            
            epoch_info = epochs[epoch]
            epoch_info['last_seen'] = timestamp
            epoch_info['progress_history'].append((timestamp, progress))
            epoch_info['allocation_history'].append((timestamp, len(allocation)))
            epoch_info['max_allocation'] = max(epoch_info['max_allocation'], len(allocation))
            
            # Check if this epoch reached expected GPU count
            if job_expected_gpus and len(allocation) >= job_expected_gpus:
                epoch_info['reached_expected'] = True
                # Track when stable allocation started
                if epoch_info['stable_start_time'] is None:
                    epoch_info['stable_start_time'] = timestamp
            
            # Check for wrong allocation (not expected GPU count)
            if job_expected_gpus and len(allocation) != job_expected_gpus:
                epoch_info['has_wrong_allocation'] = True
                
            # Check for rescaling within the same epoch
            if len(epoch_info['allocation_history']) > 1:
                prev_allocation_size = epoch_info['allocation_history'][-2][1]
                current_allocation_size = len(allocation)
                if prev_allocation_size != current_allocation_size:
                    epoch_info['has_rescaling'] = True
                
    # Mark epochs that were affected by monitor gaps for all jobs
    for job_name, job_info in jobs_data.items():
        epochs = job_info['epochs']
        for epoch_num, epoch_info in epochs.items():
            for gap_start, gap_end in monitor_gaps:
                # If the epoch was active during a monitor gap
                if epoch_info['first_seen'] <= gap_end and epoch_info['last_seen'] >= gap_start:
                    epoch_info['has_monitor_gap'] = True
                    break
    
        # Calculate startup timing metrics for each job
        startup_metrics = job_info['startup_metrics']
        if startup_metrics['allocation_timestamp'] and startup_metrics['pod_normal_timestamp']:
            startup_metrics['image_building_time'] = (
                startup_metrics['pod_normal_timestamp'] - startup_metrics['allocation_timestamp']
            )
            
        if startup_metrics['pod_normal_timestamp'] and startup_metrics['progress_growth_timestamp']:
            startup_metrics['rescaling_time'] = (
                startup_metrics['progress_growth_timestamp'] - startup_metrics['pod_normal_timestamp']
            )
    
        # Calculate epoch durations and mark affected epochs
        for epoch_num, epoch_info in epochs.items():
            epoch_info['duration'] = epoch_info['last_seen'] - epoch_info['first_seen']
            
            # Calculate stable allocation duration
            if epoch_info['stable_start_time'] is not None:
                epoch_info['stable_allocation_duration'] = epoch_info['last_seen'] - epoch_info['stable_start_time']
            
            # Mark epoch as affected if any of the conditions are met
            epoch_info['is_affected'] = (
                epoch_num == 0 or                      # Epoch 0 is always affected (startup)
                epoch_info['has_wrong_allocation'] or  # Wrong GPU allocation
                epoch_info['has_monitor_gap'] or       # Monitor gaps
                epoch_info['has_rescaling']            # Rescaling during epoch
            )
            
            # Debug output for affected epochs
            if epoch_info['is_affected']:
                reasons = []
                if epoch_num == 0:
                    reasons.append("startup_epoch")
                if epoch_info['has_wrong_allocation']:
                    reasons.append("wrong_allocation")
                if epoch_info['has_monitor_gap']:
                    reasons.append("monitor_gap") 
                if epoch_info['has_rescaling']:
                    reasons.append("rescaling")
                print(f"    Epoch {epoch_num} marked as affected: {', '.join(reasons)}")
    
    return jobs_data

def calculate_goodput(epochs, app_name, expected_gpus):
    """Calculate goodput for unaffected epochs only."""
    if app_name not in APPLICATIONS:
        return None
        
    dataset_size = APPLICATIONS[app_name]["dataset_size"]
    goodputs = {}
    
    for epoch_num, epoch_info in epochs.items():
        # Only calculate goodput for unaffected epochs
        if not epoch_info['is_affected'] and epoch_info['duration'] > 0:
            # Use stable allocation duration if available, otherwise use total duration
            if epoch_info['stable_allocation_duration'] > 0:
                goodput = dataset_size / epoch_info['stable_allocation_duration']
            else:
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
        
        # Get all GPU log files (both single and parallel)
        log_files = glob.glob(os.path.join(app_dir, "*gpu.txt"))
        
        # Debug: Print found log files
        print(f"\nFound log files for {app_name}:")
        for f in sorted(log_files):
            print(f"  - {os.path.basename(f)}")
        
        for log_file in sorted(log_files):
            jobs_data = process_single_log(log_file)
            
            # Process each job in the log file
            for job_name, job_info in jobs_data.items():
                epochs = job_info['epochs']
                expected_gpus = job_info['expected_gpus']
                startup_metrics = job_info['startup_metrics']
                
                if expected_gpus is None:
                    print(f"Warning: Could not determine expected GPUs for job {job_name} in {log_file}")
                    continue
                
                # Debug: Show what GPU count was extracted
                print(f"  Job {job_name}: {expected_gpus} GPUs")
                
                # Extract base app name from job name (e.g., "cifar10-0" -> "cifar10")
                job_app_name = job_name.rsplit('-', 1)[0]
                
                # Make sure we're processing the right application
                if job_app_name != app_name:
                    continue
                
                goodputs = calculate_goodput(epochs, app_name, expected_gpus)
                
                # Store startup metrics
                startup_times_dict[app_name].append({
                    'gpus': expected_gpus,
                    'job_name': job_name,
                    'metrics': startup_metrics
                })
                
                # Store goodput values by epoch and track affected status
                if goodputs:
                    for epoch_num, goodput_value in goodputs.items():
                        if epoch_num not in goodput_dict[app_name]:
                            goodput_dict[app_name][epoch_num] = {}
                        # Store with job name if multiple jobs have same GPU count
                        if expected_gpus in goodput_dict[app_name][epoch_num]:
                            # Average if we have multiple measurements for same GPU count
                            existing_val = goodput_dict[app_name][epoch_num][expected_gpus]
                            goodput_dict[app_name][epoch_num][expected_gpus] = (existing_val + goodput_value) / 2
                        else:
                            goodput_dict[app_name][epoch_num][expected_gpus] = goodput_value
                
                # Also track which epochs have affected jobs (for any GPU count)
                for epoch_num, epoch_info in epochs.items():
                    if epoch_info['is_affected']:
                        # Mark this epoch as having affected jobs
                        if 'affected_epochs' not in goodput_dict[app_name]:
                            goodput_dict[app_name]['affected_epochs'] = set()
                        goodput_dict[app_name]['affected_epochs'].add(epoch_num)
    
    # Handle affected epochs and interpolation
    for app_name in goodput_dict:
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        
        # Get affected epochs
        affected_epochs = goodput_dict[app_name].get('affected_epochs', set())
        if affected_epochs:
            print(f"\n{app_name}: Removing data for affected epochs: {sorted(affected_epochs)}")
        
        # Remove all data for affected epochs
        for epoch_num in list(affected_epochs):
            if epoch_num in goodput_dict[app_name]:
                del goodput_dict[app_name][epoch_num]
                print(f"  Removed all GPU data for epoch {epoch_num}")
        
        # Clean up the affected_epochs tracking
        if 'affected_epochs' in goodput_dict[app_name]:
            del goodput_dict[app_name]['affected_epochs']
        
        # Find which GPU counts we have data for
        all_gpu_counts = set()
        for epoch_key, epoch_data in goodput_dict[app_name].items():
            if isinstance(epoch_key, int):  # Skip non-epoch keys
                all_gpu_counts.update(epoch_data.keys())
        all_gpu_counts = sorted(list(all_gpu_counts))
        
        if not all_gpu_counts:
            continue
            
        print(f"  Available GPU counts: {all_gpu_counts}")
        
        # Fill missing epochs (including affected ones)
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
                            isinstance(next_epoch, int) and
                            gpu_count in goodput_dict[app_name][next_epoch]):
                            goodput_dict[app_name][epoch][gpu_count] = goodput_dict[app_name][next_epoch][gpu_count]
                            found = True
                            break
                    
                    # If not found, try previous epochs
                    if not found:
                        for prev_epoch in range(epoch - 1, -1, -1):
                            if (prev_epoch in goodput_dict[app_name] and 
                                isinstance(prev_epoch, int) and
                                gpu_count in goodput_dict[app_name][prev_epoch]):
                                goodput_dict[app_name][epoch][gpu_count] = goodput_dict[app_name][prev_epoch][gpu_count]
                                found = True
                                break
                    
                    if found and epoch in affected_epochs:
                        print(f"    Interpolated {gpu_count} GPUs for affected epoch {epoch}")
        
        # Summary of interpolated epochs
        interpolated_epochs = [e for e in affected_epochs if e < max_epochs]
        if interpolated_epochs:
            print(f"  Interpolated {len(interpolated_epochs)} affected epochs: {sorted(interpolated_epochs)}")
    
    return goodput_dict, startup_times_dict

def plot_goodput_functions(goodput_dict):
    """Create figure with 6 subplots showing goodput functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define which epochs to plot for each application
    plot_configs = [
        ("cifar10", 0),
        ("cifar10", 7),
        ("cifar10", 15),("cifar10", 30),("cifar10", 60),
        ("cifar10", 90)
    ]
    
    for idx, (app_name, epoch) in enumerate(plot_configs):
        ax = axes[idx]
        
        if app_name in goodput_dict and epoch in goodput_dict[app_name]:
            # Get data points
            gpu_counts = sorted(goodput_dict[app_name][epoch].keys())
            goodputs = [goodput_dict[app_name][epoch][gpu] for gpu in gpu_counts]
            
            if gpu_counts and goodputs:
                # Plot data points
                ax.plot(gpu_counts, goodputs, 'o-', markersize=8, linewidth=2, label='Actual')
                
                # Add linear scaling reference line from 1 GPU point
                if 1 in goodput_dict[app_name][epoch]:
                    base_goodput = goodput_dict[app_name][epoch][1]
                    max_gpu = max(gpu_counts)
                    linear_gpus = range(1, max_gpu + 1)
                    linear_goodputs = [base_goodput * gpu for gpu in linear_gpus]
                    ax.plot(linear_gpus, linear_goodputs, '--', alpha=0.7, color='red', 
                           label='Linear scaling from 1 GPU', linewidth=2)
                
                # Interpolate for smooth curve of actual data
                if len(gpu_counts) > 2:
                    gpu_range = np.linspace(min(gpu_counts), max(gpu_counts), 100)
                    f = interp1d(gpu_counts, goodputs, kind='linear', fill_value='extrapolate')
                    ax.plot(gpu_range, f(gpu_range), ':', alpha=0.5, color='blue')
                
                ax.set_xlabel('Number of GPUs')
                ax.set_ylabel('Goodput (samples/sec)')
                ax.set_title(f'{app_name} - Epoch {epoch}')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, max(gpu_counts) + 1))
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
                
                # Add legend for the first subplot
                if idx == 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{app_name} - Epoch {epoch}')
                ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(f'{app_name} - Epoch {epoch}')
            ax.set_ylim(bottom=0)
    
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
        print(f"{'Job':<15} {'GPUs':<6} {'Image Building':<18} {'Rescaling':<18}")
        print(f"{'Name':<15} {'':6} {'Time (s)':<18} {'Time (s)':<18}")
        print("-"*80)
        
        for item in sorted(startup_times_dict[app_name], key=lambda x: (x['gpus'], x.get('job_name', ''))):
            gpus = item['gpus']
            job_name = item.get('job_name', 'unknown')
            m = item['metrics']
            
            image_build = f"{m['image_building_time']:.2f}" if m.get('image_building_time') is not None else "N/A"
            rescaling = f"{m['rescaling_time']:.2f}" if m.get('rescaling_time') is not None else "N/A"
            
            print(f"{job_name:<15} {gpus:<6} {image_build:<18} {rescaling:<18}")

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
    # Allow passing base directory as command line argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "./experiment_results/dummy-goodput-12xlarge"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found")
        print(f"Usage: python {sys.argv[0]} [base_directory]")
        print(f"Default: ./experiment_results/dummy-goodput-12xlarge")
        sys.exit(1)
    
    print(f"Processing all application logs from: {base_dir}")
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