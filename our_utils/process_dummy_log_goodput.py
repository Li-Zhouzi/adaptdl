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
    "bert": {"dataset_size": 97077, "max_epochs": 2, "init_bsz": 12},
    "cifar10": {"dataset_size": 50000, "max_epochs": 100, "init_bsz": 128},
    "ncf": {"dataset_size": 1000000, "max_epochs": 10},
    "imagenet": {"dataset_size": 1281167, "max_epochs": 90},
    "deepspeech2": {"dataset_size": 4074, "max_epochs": 80, "init_bsz": 20},
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
                    },
                    'has_reached_expected': False,
                    'last_epoch': None
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
                    'is_affected': False,
                    'duration': 0,
                    'has_monitor_gap': False,
                    'has_wrong_allocation': False,
                    'has_rescaling': False,
                    # New fields for productive time/goodput calculation
                    'productive_time': 0.0,
                    'progress_under_expected': 0.0,
                    'min_progress': None,
                    'max_progress': None,
                    'recent_expected_entries': [],  # list of (ts, progress)
                    'last_growth_ts_under_expected': None,
                }
            
            # Handle epoch transition: if job moved to a new epoch, add end-of-epoch plateau time
            job_last_epoch = job_info.get('last_epoch')
            if job_last_epoch is not None and job_last_epoch != epoch and job_last_epoch in epochs:
                prev_epoch_info = epochs[job_last_epoch]
                last_growth_ts = prev_epoch_info.get('last_growth_ts_under_expected')
                if last_growth_ts is not None:
                    plateau_duration = prev_epoch_info['last_seen'] - last_growth_ts
                    if plateau_duration > 0:
                        prev_epoch_info['productive_time'] += plateau_duration
            job_info['last_epoch'] = epoch

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
            
            # Update min/max progress for this epoch
            if isinstance(progress, (int, float)):
                if epoch_info['min_progress'] is None:
                    epoch_info['min_progress'] = progress
                else:
                    if progress < epoch_info['min_progress']:
                        raise AssertionError(f"Min progress < {progress} for job {job_name} epoch {epoch}, expected number of GPUs: {job_expected_gpus}")
                if epoch_info['max_progress'] is None or progress > epoch_info['max_progress']:
                    epoch_info['max_progress'] = progress

            # Allocation assertion: once reached expected GPUs, must stay at expected
            is_expected_alloc = (job_expected_gpus and len(allocation) == job_expected_gpus)
            completion_timestamp = log_entry.get('completion_time')
            if is_expected_alloc:
                if not job_info['has_reached_expected']:
                    job_info['has_reached_expected'] = True
            else:
                if job_info['has_reached_expected'] and completion_timestamp is not None:
                    # raise AssertionError(f"Job {job_name} ran with non-expected allocation after reaching expected GPUs, at ")
                    raise AssertionError(f"Job {job_name} ran with non-expected allocation after reaching expected GPUs, at epoch {epoch}, expected number of GPUs: {job_expected_gpus}")

            # Productive time and progress under expected using previous 3 entries logic
            if is_expected_alloc and isinstance(progress, (int, float)):
                recent = epoch_info['recent_expected_entries']  # list of (ts, progress), holds last up to 3 previous lines
                prev_ts = None
                prev_prog = None
                if len(recent) > 0:
                    prev_ts, prev_prog = recent[-1]
                # Only add if progress strictly increased since last line
                if prev_ts is not None and isinstance(prev_prog, (int, float)) and progress > prev_prog:
                    # Among the previous 3 lines, find earliest time that has the same progress as the last line
                    anchor_ts = prev_ts
                    for ts_i, prog_i in recent:
                        if prog_i == prev_prog:
                            anchor_ts = ts_i
                            break
                    interval = timestamp - anchor_ts
                    if interval > 0:
                        epoch_info['productive_time'] += interval
                        epoch_info['progress_under_expected'] += (progress - prev_prog)
                        epoch_info['last_growth_ts_under_expected'] = timestamp
                # Append current entry and keep only the last 3 as the "previous 3 lines" for next iteration
                recent.append((timestamp, progress))
                while len(recent) > 3:
                    recent.pop(0)
                
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
            
            # Compute total progress delta for this epoch
            min_p = epoch_info.get('min_progress')
            max_p = epoch_info.get('max_progress')
            if isinstance(min_p, (int, float)) and isinstance(max_p, (int, float)):
                epoch_info['total_progress_delta'] = max(0.0, max_p - min_p)
            else:
                epoch_info['total_progress_delta'] = 0.0

            # New affected rule: unaffected if progress under expected > 50% total progress
            prog_expected = epoch_info.get('progress_under_expected', 0.0)
            total_prog = epoch_info.get('total_progress_delta', 0.0)
            epoch_info['is_affected'] = not (total_prog > 0 and prog_expected > 0.5 * total_prog)

            # Validate plateau share: end-of-epoch plateau must be <= 10% of total epoch duration
            last_growth_ts = epoch_info.get('last_growth_ts_under_expected')
            if last_growth_ts is not None and isinstance(epoch_info.get('duration'), (int, float)):
                plateau_duration = max(0, epoch_info['last_seen'] - last_growth_ts)
                total_duration = epoch_info['duration']
                if total_duration > 0 and plateau_duration > 0:
                    threshold = 0.2 * total_duration
                    app_name = job_name.rsplit('-', 1)[0]
                    # Skip assertion if epoch overlaps monitor gaps, since we lack samples
                    # if (not epoch_info.get('has_monitor_gap', False) and
                    #     not epoch_info.get('is_affected', False) and
                    #     plateau_duration > threshold and plateau_duration > 10 and
                    #     epoch_num < APPLICATIONS[app_name]["max_epochs"]-1):
                    #     raise AssertionError(
                    #         f"End-of-epoch plateau exceeds 20% (plateau={plateau_duration}s, and is larger than 10s "
                    #         f"threshold={threshold}s) for job {job_name} epoch {epoch_num}, expected number of GPUs: {job_info['expected_gpus']}"
                    #     )
        # Do not forward-fill here; keep raw per-epoch batch_sizes only
            
    return jobs_data

def calculate_goodput(epochs, app_name, expected_gpus):
    """Calculate goodput for unaffected epochs only.

    New definition: goodput = progress_under_expected / productive_time.
    """
    goodputs = {}

    for epoch_num, epoch_info in epochs.items():
        is_unaffected = not epoch_info.get('is_affected', False)
        productive_time = epoch_info.get('productive_time', 0.0) or 0.0
        progress_under_expected = epoch_info.get('progress_under_expected', 0.0) or 0.0
        if is_unaffected and productive_time > 0:
            goodput = progress_under_expected / productive_time * APPLICATIONS[app_name]["init_bsz"]
            goodputs[epoch_num] = goodput
    return goodputs

def build_goodput_dict(base_dir):
    """Build goodput dictionary for all applications and GPU configurations."""
    goodput_dict = {}
    bsz_dict = {}
    processed_gpu_counts_by_app = defaultdict(set)
    affected_map = defaultdict(lambda: defaultdict(dict))  # app -> epoch -> {gpus: is_affected}
    
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
        singles = sorted([f for f in log_files if not os.path.basename(f).startswith('parallel_')])
        parallels = sorted([f for f in log_files if os.path.basename(f).startswith('parallel_')])

        # First pass: process non-parallel logs
        for log_file in singles:
            jobs_data = process_single_log(log_file)
            # Minimal BERT-only summary print per log
            has_printed_header = False
            for job_name, job_info in jobs_data.items():
                base_app = job_name.rsplit('-', 1)[0]
                if base_app != 'bert':
                    continue
                if not has_printed_header:
                    print("\n" + "-" * 80)
                    print(f"BERT summary for log: {os.path.basename(log_file)}")
                    print("-" * 80)
                    print(f"{'job':<16} {'exp_gpus':>8} {'epoch':>6} {'duration_s':>12} {'productive_s':>13} {'prod_progress':>14} {'total_progress':>14}")
                    has_printed_header = True
                expected_gpus = job_info.get('expected_gpus')
                for epoch_num in sorted(job_info.get('epochs', {}).keys()):
                    e = job_info['epochs'][epoch_num]
                    duration = e.get('duration')
                    productive = e.get('productive_time')
                    prod_prog = e.get('progress_under_expected')
                    total_prog = e.get('total_progress_delta')
                    print(f"{job_name:<16} {expected_gpus:>8} {epoch_num:>6} {duration:>12.1f} {productive:>13.1f} {prod_prog:>14.2f} {total_prog:>14.2f}")
            
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

                # Record that we've processed this GPU count from single logs
                processed_gpu_counts_by_app[app_name].add(expected_gpus)
                
                # Aggregate batch sizes per epoch and expected replica count only for unaffected epochs
                for epoch_num, epoch_info in epochs.items():
                    is_unaffected = not epoch_info.get('is_affected', False)
                    if epoch_info.get('batch_sizes') and is_unaffected:
                        if epoch_num not in bsz_dict[app_name]:
                            bsz_dict[app_name][epoch_num] = {}
                        if expected_gpus not in bsz_dict[app_name][epoch_num]:
                            bsz_dict[app_name][epoch_num][expected_gpus] = set()
                        bsz_dict[app_name][epoch_num][expected_gpus].update(epoch_info['batch_sizes'])
                
                goodputs = calculate_goodput(epochs, app_name, expected_gpus)
                
                
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
                    # Aggregate affected flag into affected_map (OR if multiple sources)
                    prev = affected_map[app_name][epoch_num].get(expected_gpus)
                    if prev is None:
                        affected_map[app_name][epoch_num][expected_gpus] = bool(epoch_info['is_affected'])
                    else:
                        affected_map[app_name][epoch_num][expected_gpus] = bool(prev or epoch_info['is_affected'])
                    if epoch_info['is_affected']:
                        # Mark this epoch as having affected jobs for later removal/interp
                        if 'affected_epochs' not in goodput_dict[app_name]:
                            goodput_dict[app_name]['affected_epochs'] = set()
                        goodput_dict[app_name]['affected_epochs'].add(epoch_num)

        # Second pass: process parallel logs, skipping GPU counts already processed in singles
        for log_file in parallels:
            jobs_data = process_single_log(log_file)
            # Minimal BERT-only summary print per log
            has_printed_header = False
            for job_name, job_info in jobs_data.items():
                base_app = job_name.rsplit('-', 1)[0]
                if base_app != 'bert':
                    continue
                if not has_printed_header:
                    print("\n" + "-" * 80)
                    print(f"BERT summary for log: {os.path.basename(log_file)}")
                    print("-" * 80)
                    print(f"{'job':<16} {'exp_gpus':>8} {'epoch':>6} {'duration_s':>12} {'productive_s':>13} {'prod_progress':>14} {'total_progress':>14}")
                    has_printed_header = True
                expected_gpus = job_info.get('expected_gpus')
                for epoch_num in sorted(job_info.get('epochs', {}).keys()):
                    e = job_info['epochs'][epoch_num]
                    duration = e.get('duration')
                    productive = e.get('productive_time')
                    prod_prog = e.get('progress_under_expected')
                    total_prog = e.get('total_progress_delta')
                    print(f"{job_name:<16} {expected_gpus:>8} {epoch_num:>6} {duration:>12.1f} {productive:>13.1f} {prod_prog:>14.2f} {total_prog:>14.2f}")
            
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

                # Skip GPU counts already processed from single logs
                if expected_gpus in processed_gpu_counts_by_app[app_name]:
                    continue
                
                # Aggregate batch sizes per epoch and expected replica count only for unaffected epochs
                for epoch_num, epoch_info in epochs.items():
                    is_unaffected = not epoch_info.get('is_affected', False)
                    if epoch_info.get('batch_sizes') and is_unaffected:
                        if epoch_num not in bsz_dict[app_name]:
                            bsz_dict[app_name][epoch_num] = {}
                        if expected_gpus not in bsz_dict[app_name][epoch_num]:
                            bsz_dict[app_name][epoch_num][expected_gpus] = set()
                        bsz_dict[app_name][epoch_num][expected_gpus].update(epoch_info['batch_sizes'])
                
                goodputs = calculate_goodput(epochs, app_name, expected_gpus)                
                
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
                    # Aggregate affected flag into affected_map (OR if multiple sources)
                    prev = affected_map[app_name][epoch_num].get(expected_gpus)
                    if prev is None:
                        affected_map[app_name][epoch_num][expected_gpus] = bool(epoch_info['is_affected'])
                    else:
                        affected_map[app_name][epoch_num][expected_gpus] = bool(prev or epoch_info['is_affected'])
                    if epoch_info['is_affected']:
                        # Mark this epoch as having affected jobs for later removal/interp
                        if 'affected_epochs' not in goodput_dict[app_name]:
                            goodput_dict[app_name]['affected_epochs'] = set()
                        goodput_dict[app_name]['affected_epochs'].add(epoch_num)
    
    # Handle affected epochs and interpolation
    for app_name in goodput_dict:
        max_epochs = APPLICATIONS[app_name]["max_epochs"]
        
        # Get affected epochs (keep a copy for fill policy below)
        affected_epochs = goodput_dict[app_name].get('affected_epochs', set())
        affected_epochs_copy = set(affected_epochs)
        
        # Remove all data for affected epochs
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

    return goodput_dict, bsz_dict, affected_map

def plot_goodput_functions(goodput_dict):
    """Create figure with 6 subplots showing goodput functions."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define which epochs to plot for each application
    plot_configs = [
        ("bert", 0),
        ("bert", 1),
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

def print_affected_table(affected_map):
    """Print per-app tables: rows=epochs, columns=expected_gpus, entries=is_affected."""
    print("\n" + "="*80)
    print("AFFECTED EPOCHS TABLES")
    print("="*80)
    for app_name in sorted(affected_map.keys()):
        max_epochs = APPLICATIONS.get(app_name, {}).get("max_epochs", 0)
        # Collect all gpu columns present for this app
        gpu_cols = set()
        for epoch_data in affected_map[app_name].values():
            gpu_cols.update(epoch_data.keys())
        gpu_cols = sorted(list(gpu_cols))
        print(f"\n{app_name.upper()}")
        if not gpu_cols:
            print("(no data)")
            continue
        # Header
        header = ["epoch"] + [str(g) for g in gpu_cols]
        print("\t".join(header))
        # Rows
        for epoch in range(max_epochs):
            row = [str(epoch)]
            epoch_map = affected_map[app_name].get(epoch, {})
            for g in gpu_cols:
                val = epoch_map.get(g)
                row.append("True" if val else ("False" if val is not None else ""))
            print("\t".join(row))

def main():
    # Allow passing base directory as command line argument
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "./experiment_results/dummy-cbd-0916"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found")
        print(f"Usage: python {sys.argv[0]} [base_directory]")
        print(f"Default: ./experiment_results/dummy-goodput-12xlarge")
        sys.exit(1)
    
    goodput_dict, bsz_dict, affected_map = build_goodput_dict(base_dir)
    
    # Print goodput functions in copy-paste format
    print_goodput_functions(goodput_dict)
    # print_bsz_functions(bsz_dict)
    
    # Print affected tables
    print_affected_table(affected_map)
    
    # Create goodput plots
    plot_goodput_functions(goodput_dict)

if __name__ == "__main__":
    main()
