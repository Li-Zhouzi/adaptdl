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
    "bert": {"dataset_size": 97077, "max_epochs": 2},
    "cifar10": {"dataset_size": 50000, "max_epochs": 100},
    "ncf": {"dataset_size": 1000000, "max_epochs": 10},
    "imagenet": {"dataset_size": 1281167, "max_epochs": 90},
    "deepspeech2": {"dataset_size": 4074, "max_epochs": 80},
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
    
    # For parallel mode, pre-determine job to GPU mapping
    job_to_expected_gpus = {}
    if is_parallel:
        print(f"Pre-determining job-to-GPU mapping for parallel file: {filename}")
        for i, line in enumerate(lines):
            log_entry = json.loads(line.strip())
            submitted_jobs = log_entry.get('submitted_jobs', [])
            
            # Check if we have exactly the expected number of jobs
            if len(submitted_jobs) == len(expected_gpus_list):
                # Check if their allocations match expected GPU list
                allocations = [len(job.get('allocation', [])) for job in submitted_jobs]
                allocations.sort()
                expected_gpus_list_sorted = sorted(expected_gpus_list)
                
                if allocations == expected_gpus_list_sorted:
                    print(f"Found matching allocation at line {i}: {allocations}")
                    
                    # Map each job to its GPU count
                    for job in submitted_jobs:
                        job_name = job['name']
                        allocation_size = len(job.get('allocation', []))
                        job_to_expected_gpus[job_name] = allocation_size
                    
                    # Verify for next 10 lines that jobs maintain correct allocations
                    verification_passed = True
                    for verify_i in range(i + 1, min(i + 11, len(lines))):
                        verify_entry = json.loads(lines[verify_i].strip())
                        for job in verify_entry.get('submitted_jobs', []):
                            job_name = job['name']
                            if job_name in job_to_expected_gpus:
                                current_allocation = len(job.get('allocation', []))
                                expected = job_to_expected_gpus[job_name]
                                if current_allocation != expected:
                                    verification_passed = False
                                    break
                        if not verification_passed:
                            break
                    
                    if verification_passed:
                        print(f"Verification passed. Job-to-GPU mapping: {job_to_expected_gpus}")
                        break
                    else:
                        print(f"Verification failed, continuing search...")
                        job_to_expected_gpus = {}
        
        if not job_to_expected_gpus:
            pass  # Will handle this case in job processing
    else:
        # For non-parallel mode, all jobs get the same expected GPU count
        single_expected_gpus = expected_gpus_list[0]
    
    # Define threshold for monitor gap detection (e.g., 100 seconds)
    MONITOR_GAP_THRESHOLD = 20  # seconds
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
                print(f"Monitor gap detected: {prev_timestamp} to {timestamp}")
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
            batch_size = job.get('batch_size', None)
            pod_status = job.get('pod_status', '')
            
            # Initialize job data if not exists
            if job_name not in jobs_data:
                # Determine expected GPUs for this job
                if is_parallel:
                    # Use pre-computed mapping
                    job_expected_gpus = job_to_expected_gpus.get(job_name)
                    if job_expected_gpus is None:
                        print(f"Warning: No expected GPU count found for job {job_name}")
                        continue
                else:
                    # In single mode, all jobs should have the same expected GPUs
                    job_expected_gpus = single_expected_gpus
                
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
                    'allocation_history': [],
                    'batch_sizes': set(),
                    'is_affected': False,  # Mark if epoch is affected by scaling/gaps/wrong allocation
                    'duration': 0,
                    'has_monitor_gap': False,  # Mark if epoch was affected by monitor gap
                    'has_wrong_allocation': False,  # Mark if epoch had wrong GPU count
                    'has_rescaling': False,  # Mark if epoch had rescaling during execution
                }
            
            epoch_info = epochs[epoch]
            epoch_info['last_seen'] = timestamp
            epoch_info['allocation_history'].append((timestamp, len(allocation)))
            # Track batch sizes only when allocation equals expected GPUs for the job
            if (isinstance(batch_size, (int, float)) and batch_size > 0 and
                job_expected_gpus and len(allocation) == job_expected_gpus):
                epoch_info['batch_sizes'].add(int(batch_size))
            
            # Check for wrong allocation (not expected GPU count)
            if job_expected_gpus and len(allocation) != job_expected_gpus:
                epoch_info['has_wrong_allocation'] = True
                
            # Check for rescaling within the same epoch
            if len(epoch_info['allocation_history']) > 1:
                prev_allocation_size = epoch_info['allocation_history'][-2][1]
                current_allocation_size = len(allocation)
                if prev_allocation_size != current_allocation_size:
                    epoch_info['has_rescaling'] = True
                
    # Mark epochs that were affected by monitor gaps for all jobs,
    # and additionally mark the immediate next epoch as affected.
    for job_name, job_info in jobs_data.items():
        epochs = job_info['epochs']
        direct_gap_epochs = set()
        for epoch_num, epoch_info in epochs.items():
            for gap_start, gap_end in monitor_gaps:
                # If the epoch was active during a monitor gap
                if epoch_info['first_seen'] <= gap_end and epoch_info['last_seen'] >= gap_start:
                    epoch_info['has_monitor_gap'] = True
                    direct_gap_epochs.add(epoch_num)
                    break
        # After identifying epochs directly overlapping gaps, mark the next epoch as affected as well
        if direct_gap_epochs:
            sorted_epochs = sorted(epochs.keys())
            index_by_epoch = {e: i for i, e in enumerate(sorted_epochs)}
            for e in sorted(direct_gap_epochs):
                idx = index_by_epoch.get(e)
                if idx is not None and idx + 1 < len(sorted_epochs):
                    next_epoch = sorted_epochs[idx + 1]
                    epochs[next_epoch]['has_monitor_gap'] = True
    
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
        # First, get all timestamps for this job to find previous timestamps
        all_timestamps = []
        for epoch_data in epochs.values():
            for timestamp, _ in epoch_data['allocation_history']:
                all_timestamps.append(timestamp)
        all_timestamps = sorted(set(all_timestamps))
        
        for epoch_num, epoch_info in epochs.items():
            # Find the timestamp just before this epoch started
            first_seen = epoch_info['first_seen']
            last_seen = epoch_info['last_seen']
            
            # Find previous timestamp
            prev_timestamp = None
            for ts in all_timestamps:
                if ts < first_seen:
                    prev_timestamp = ts
                else:
                    break
            
            # Calculate duration as last_seen - previous_timestamp (or first_seen if no previous)
            if prev_timestamp is not None:
                epoch_info['duration'] = last_seen - prev_timestamp
            else:
                # No previous timestamp, fall back to original calculation
                epoch_info['duration'] = last_seen - first_seen
            
            # Mark epoch as affected if any of the conditions are met
            epoch_info['is_affected'] = (
                epoch_num == 0 or                      # Epoch 0 is always affected (startup)
                epoch_info['has_wrong_allocation'] or  # Wrong GPU allocation
                epoch_info['has_monitor_gap'] or       # Monitor gaps
                epoch_info['has_rescaling']            # Rescaling during epoch
            )
        # Do not forward-fill here; keep raw per-epoch batch_sizes only
            
    
    if "cifar10" in log_file_path and "parallel" in filename:
        job = jobs_data["cifar10-2"]
        print(f"Total epochs found: {len(epochs)}")
        for epoch_num, epoch_info in epochs.items():
            if epoch_num != 30:
                continue
            print(f"Epoch {epoch_num}:")
            print(f"  Duration: {epoch_info['duration']}")
            print(f"  Is affected: {epoch_info['is_affected']}")
    return jobs_data

def calculate_goodput(epochs, app_name, expected_gpus):
    """Calculate goodput for unaffected epochs only."""
    if app_name not in APPLICATIONS:
        return None
        
    dataset_size = APPLICATIONS[app_name]["dataset_size"]
    goodputs = {}
    
    # Debug: Print epoch details for bert
    # if app_name == "bert":
    #     print(f"\n=== BERT DEBUG: Job with {expected_gpus} GPUs ===")
    #     print(f"Total epochs found: {len(epochs)}")
    #     for epoch_num, epoch_info in epochs.items():
    #         print(f"Epoch {epoch_num}:")
    #         print(f"  Duration: {epoch_info['duration']}")
    #         print(f"  Is affected: {epoch_info['is_affected']}")
    #         print(f"  Has wrong allocation: {epoch_info['has_wrong_allocation']}")
    #         print(f"  Has rescaling: {epoch_info['has_rescaling']}")
    #         print(f"  Has monitor gap: {epoch_info['has_monitor_gap']}")
    #         print(f"  Allocation history length: {len(epoch_info['allocation_history'])}")
    #         if len(epoch_info['allocation_history']) > 0:
    #             print(f"  First allocation: {epoch_info['allocation_history'][0]}")
    #             print(f"  Last allocation: {epoch_info['allocation_history'][-1]}")
    #     print("=" * 50)
    
    for epoch_num, epoch_info in epochs.items():
        # Only calculate goodput for unaffected epochs
        # Special case: For bert, treat all epochs as unaffected
        is_unaffected = not epoch_info['is_affected'] if app_name != "bert" else True
        
        if is_unaffected and epoch_info['duration'] > 0:
            goodput = dataset_size / epoch_info['duration']
            
            # Debug suspicious goodput values
            # if (epoch_num == 60 and expected_gpus == 12 and app_name == "cifar10"):
            #     print(f"\nDEBUG: {app_name} epoch {epoch_num}, {expected_gpus} GPU:")
            #     print(f"  Dataset size: {dataset_size}")
            #     print(f"  Duration: {epoch_info['duration']} seconds")
            #     print(f"  First seen: {epoch_info['first_seen']}")
            #     print(f"  Last seen: {epoch_info['last_seen']}")
            #     print(f"  Calculated goodput: {goodput:.2f} samples/sec")
            #     print(f"  Allocation history length: {len(epoch_info['allocation_history'])}")
            #     if len(epoch_info['allocation_history']) > 0:
            #         print(f"  First allocation entry: {epoch_info['allocation_history'][0]}")
            #         print(f"  Last allocation entry: {epoch_info['allocation_history'][-1]}")
            #     print(f"  Is affected: {epoch_info['is_affected']}")
            #     print(f"  Has wrong allocation: {epoch_info['has_wrong_allocation']}")
            #     print(f"  Has rescaling: {epoch_info['has_rescaling']}")
            #     print(f"  Has monitor gap: {epoch_info['has_monitor_gap']}")
            
            goodputs[epoch_num] = goodput
    
    return goodputs

def build_goodput_dict(base_dir):
    """Build goodput dictionary for all applications and GPU configurations."""
    goodput_dict = {}
    startup_times_dict = defaultdict(list)
    bsz_dict = {}
    
    # Get all application directories
    app_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for app_name in app_dirs:
        if app_name not in APPLICATIONS:
            print(f"Warning: Application {app_name} not found in APPLICATIONS config")
            continue
            
        goodput_dict[app_name] = {}
        bsz_dict[app_name] = {}
        app_dir = os.path.join(base_dir, app_name)
        
        # Get all GPU log files (both single and parallel)
        log_files = glob.glob(os.path.join(app_dir, "*gpu.txt"))
        
        
        for log_file in sorted(log_files):
            jobs_data = process_single_log(log_file)
            
            # Process each job in the log file
            for job_name, job_info in jobs_data.items():
                epochs = job_info['epochs']
                expected_gpus = job_info['expected_gpus']
                startup_metrics = job_info['startup_metrics']
                
                if expected_gpus is None:
                    continue
                
                # Extract base app name from job name (e.g., "cifar10-0" -> "cifar10")
                job_app_name = job_name.rsplit('-', 1)[0]
                
                # Make sure we're processing the right application
                if job_app_name != app_name:
                    continue
                
                # Special handling for BERT job filtering
                if app_name == 'bert':
                    base_filename = os.path.basename(log_file)
                    # For parallel_1_2_4 logs, ignore the 2-GPU entry
                    if (base_filename.startswith('parallel_') and 
                        '1_2_4gpu.txt' in base_filename and 
                        (expected_gpus == 2 or expected_gpus == 4)):
                        continue

                if app_name == 'cifar10':
                    base_filename = os.path.basename(log_file)
                    # For parallel_1_2_4 logs, ignore the 2-GPU entry
                    if (base_filename.startswith('parallel_') and 
                        '1_2_4gpu.txt' in base_filename and 
                        (expected_gpus == 4)):
                        continue
                
                # Aggregate batch sizes per epoch and expected replica count only for unaffected epochs
                for epoch_num, epoch_info in epochs.items():
                    is_unaffected = not epoch_info.get('is_affected', False)
                    if app_name == 'bert':
                        is_unaffected = True  # Mirror goodput handling so we keep BERT data
                    if epoch_info.get('batch_sizes') and is_unaffected:
                        if epoch_num not in bsz_dict[app_name]:
                            bsz_dict[app_name][epoch_num] = {}
                        if expected_gpus not in bsz_dict[app_name][epoch_num]:
                            bsz_dict[app_name][epoch_num][expected_gpus] = set()
                        bsz_dict[app_name][epoch_num][expected_gpus].update(epoch_info['batch_sizes'])
                
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
        
        # Get affected epochs (keep a copy for fill policy below)
        affected_epochs = goodput_dict[app_name].get('affected_epochs', set())
        affected_epochs_copy = set(affected_epochs)
        
        # Remove all data for affected epochs, except for bert where we keep them
        if app_name != 'bert':
            for epoch_num in list(affected_epochs):
                if epoch_num in goodput_dict[app_name]:
                    del goodput_dict[app_name][epoch_num]
        
        # Clean up the affected_epochs tracking for all apps
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
        
        # Fill missing epochs (including affected ones) by copying from nearest neighbor with data
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
                    
    
    # Fill missing epochs for bsz_dict similar to goodput: copy nearest neighbor values
    for app_name in bsz_dict:
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        # Determine all gpu counts observed for this app
        all_gpu_counts = set()
        for epoch_key, epoch_data in bsz_dict[app_name].items():
            all_gpu_counts.update(epoch_data.keys())
        all_gpu_counts = sorted(list(all_gpu_counts))
        if not all_gpu_counts:
            continue
        # Ensure all epochs exist
        for epoch in range(max_epochs):
            if epoch not in bsz_dict[app_name]:
                bsz_dict[app_name][epoch] = {}
            for gpu_count in all_gpu_counts:
                if gpu_count not in bsz_dict[app_name][epoch]:
                    # Look forward then backward for nearest available
                    found = False
                    for next_epoch in range(epoch + 1, max_epochs):
                        if (next_epoch in bsz_dict[app_name] and
                            gpu_count in bsz_dict[app_name][next_epoch] and
                            bsz_dict[app_name][next_epoch][gpu_count]):
                            bsz_dict[app_name][epoch][gpu_count] = set(bsz_dict[app_name][next_epoch][gpu_count])
                            found = True
                            break
                    if not found:
                        for prev_epoch in range(epoch - 1, -1, -1):
                            if (prev_epoch in bsz_dict[app_name] and
                                gpu_count in bsz_dict[app_name][prev_epoch] and
                                bsz_dict[app_name][prev_epoch][gpu_count]):
                                bsz_dict[app_name][epoch][gpu_count] = set(bsz_dict[app_name][prev_epoch][gpu_count])
                                found = True
                                break
        # Convert sets to sorted lists
        for epoch_num in list(bsz_dict[app_name].keys()):
            for gpu_count in list(bsz_dict[app_name][epoch_num].keys()):
                values = bsz_dict[app_name][epoch_num][gpu_count]
                if isinstance(values, set):
                    bsz_dict[app_name][epoch_num][gpu_count] = sorted(list(values))

    return goodput_dict, startup_times_dict, bsz_dict

def plot_goodput_functions(goodput_dict):
    """Create figure with 6 subplots showing goodput functions."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    type_print = "bert"
    # Define which epochs to plot for each application
    plot_configs = [
        (type_print, 0),
        (type_print, 1),
        ("cifar10", 1),("cifar10", 10),("cifar10", 30),("cifar10", 60),("deepspeech2", 1),("deepspeech2", 10),
        ("deepspeech2", 50)
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
    """Print goodput functions in Python dictionary format."""
    print("\n" + "="*80)
    print("GOODPUT FUNCTIONS")
    print("="*80)
    
    # Print as Python dictionary
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

def print_bsz_functions(bsz_dict):
    """Print batch size functions in Python dictionary format."""
    print("\n" + "="*80)
    print("BATCH SIZE FUNCTIONS")
    print("="*80)
    
    # Print as Python dictionary
    print("bsz_functions = {")
    for app_name in sorted(bsz_dict.keys()):
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        print(f"    '{app_name}': {{")
        for epoch in sorted(bsz_dict[app_name].keys()):
            if epoch < max_epochs:  # Only include epochs up to max_epochs
                gpu_bsz = bsz_dict[app_name][epoch]
                if gpu_bsz:
                    print(f"        {epoch}: {dict(sorted(gpu_bsz.items()))},")
        print("    },")
    print("}")

def main():
    # Allow passing base directory as command line argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "./experiment_results/dummy-cbd-0916"
        # base_dir = "./experiment_results/dummy-goodput-12xlarge"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found")
        print(f"Usage: python {sys.argv[0]} [base_directory]")
        print(f"Default: ./experiment_results/dummy-goodput-12xlarge")
        sys.exit(1)
    
    goodput_dict, startup_times_dict, bsz_dict = build_goodput_dict(base_dir)
    
    # Print goodput functions in copy-paste format
    print_goodput_functions(goodput_dict)
    print_bsz_functions(bsz_dict)
    
    # Print startup times table
    print_startup_times_table(startup_times_dict)
    
    # Create goodput plots
    plot_goodput_functions(goodput_dict)

if __name__ == "__main__":
    main()
