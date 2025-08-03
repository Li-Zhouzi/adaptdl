import json
import sys
import os
import glob
from datetime import datetime
from collections import defaultdict
import numpy as np

def process_single_log(log_file_path):
    """Process a single log file to extract metrics for dummy policy experiment."""
    
    # Extract expected GPU count from filename (e.g., "3gpu.txt" -> 3)
    filename = os.path.basename(log_file_path)
    expected_gpus = int(filename.replace('gpu.txt', ''))
    
    print(f"\nProcessing {filename} (expected {expected_gpus} GPUs)")
    print("-" * 60)
    
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
    
    # Process each log entry
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        
        # Track first timestamp
        if startup_metrics['first_timestamp'] is None:
            startup_metrics['first_timestamp'] = timestamp
        
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
                    'max_allocation': 0
                }
            
            epoch_info = epochs[epoch]
            epoch_info['last_seen'] = timestamp
            epoch_info['progress_history'].append((timestamp, progress))
            epoch_info['allocation_history'].append((timestamp, len(allocation)))
            epoch_info['max_allocation'] = max(epoch_info['max_allocation'], len(allocation))
            
            # Check if this epoch ever ran with less than expected GPUs
            if len(allocation) < expected_gpus:
                epoch_info['is_starting'] = True
    
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
    
    # Calculate epoch durations and idle times
    for epoch_num, epoch_info in epochs.items():
        epoch_info['duration'] = epoch_info['last_seen'] - epoch_info['first_seen']
        
        # Find idle time at end of epoch (when progress stops growing)
        if len(epoch_info['progress_history']) > 1:
            # Find last progress change
            last_progress_change_time = epoch_info['first_seen']
            last_progress = epoch_info['progress_history'][0][1]
            
            for timestamp, progress in epoch_info['progress_history']:
                if progress != last_progress:
                    last_progress_change_time = timestamp
                    last_progress = progress
            
            # Idle time is from last progress change to end of epoch
            epoch_info['idle_time_at_end'] = epoch_info['last_seen'] - last_progress_change_time
    
    return epochs, startup_metrics, expected_gpus

def print_single_log_summary(epochs, startup_metrics, expected_gpus):
    """Print summary for a single log file."""
    
    print("\nEpoch Analysis:")
    print("-" * 40)
    print(f"{'Epoch':<10} {'Status':<15} {'Duration':<12} {'Idle Time':<12} {'Max GPUs':<10}")
    print("-" * 60)
    
    for epoch_num in sorted(epochs.keys()):
        epoch_info = epochs[epoch_num]
        status = "Starting" if epoch_info['is_starting'] else "Normal"
        print(f"{epoch_num:<10} {status:<15} {epoch_info['duration']:<12.2f} "
              f"{epoch_info['idle_time_at_end']:<12.2f} {epoch_info['max_allocation']:<10}")
    
    # Calculate summary statistics
    normal_epochs = [e for e in epochs.values() if not e['is_starting']]
    if normal_epochs:
        avg_duration = np.mean([e['duration'] for e in normal_epochs])
        avg_idle = np.mean([e['idle_time_at_end'] for e in normal_epochs])
        
        print("\nSummary Statistics (Normal Epochs Only):")
        print("-" * 40)
        print(f"Average epoch duration: {avg_duration:.2f} seconds")
        print(f"Average idle time at epoch end: {avg_idle:.2f} seconds")
    
    # Return startup metrics for aggregated display
    return startup_metrics

def process_all_logs(directory):
    """Process all log files in the directory."""
    log_files = glob.glob(os.path.join(directory, "*gpu.txt"))
    
    if not log_files:
        print(f"No log files found in {directory}")
        return
    
    print(f"Found {len(log_files)} log files to process")
    
    all_results = {}
    all_startup_metrics = []
    
    for log_file in sorted(log_files):
        epochs, startup_metrics, expected_gpus = process_single_log(log_file)
        metrics = print_single_log_summary(epochs, startup_metrics, expected_gpus)
        
        all_results[log_file] = {
            'epochs': epochs,
            'startup_metrics': startup_metrics,
            'expected_gpus': expected_gpus
        }
        
        all_startup_metrics.append({
            'gpus': expected_gpus,
            'metrics': metrics
        })
    
    # Print aggregated startup timing table
    print("\n" + "="*80)
    print("STARTUP TIMING SUMMARY")
    print("="*80)
    print(f"{'GPUs':<6} {'Node Preparation':<18} {'Job Allocation':<18} {'Image Building':<18} {'Rescaling':<18}")
    print(f"{'':6} {'Time (s)':<18} {'Change Time (s)':<18} {'Time (s)':<18} {'Time (s)':<18}")
    print("-"*80)
    
    for item in sorted(all_startup_metrics, key=lambda x: x['gpus']):
        gpus = item['gpus']
        m = item['metrics']
        
        node_prep = f"{m['node_preparation_time']:.2f}" if m['node_preparation_time'] is not None else "N/A"
        job_alloc = f"{m['job_allocation_change_time']:.2f}" if m['job_allocation_change_time'] is not None else "N/A"
        image_build = f"{m['image_building_time']:.2f}" if m['image_building_time'] is not None else "N/A"
        rescaling = f"{m['rescaling_time']:.2f}" if m['rescaling_time'] is not None else "N/A"
        
        print(f"{gpus:<6} {node_prep:<18} {job_alloc:<18} {image_build:<18} {rescaling:<18}")
    
    return all_results

def main():
    if len(sys.argv) > 2:
        print("Usage: python process_dummy_log.py [<log_file_path>]")
        print("  If no path provided, processes all files in ./experiment_results/dummy/")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Process single file
        log_file_path = sys.argv[1]
        if not os.path.exists(log_file_path):
            print(f"Error: File {log_file_path} not found")
            sys.exit(1)
            
        epochs, startup_metrics, expected_gpus = process_single_log(log_file_path)
        metrics = print_single_log_summary(epochs, startup_metrics, expected_gpus)
        
        # Print startup timing table for single file
        print("\n" + "="*80)
        print("STARTUP TIMING SUMMARY")
        print("="*80)
        print(f"{'GPUs':<6} {'Node Preparation':<18} {'Job Allocation':<18} {'Image Building':<18} {'Rescaling':<18}")
        print(f"{'':6} {'Time (s)':<18} {'Change Time (s)':<18} {'Time (s)':<18} {'Time (s)':<18}")
        print("-"*80)
        
        node_prep = f"{metrics['node_preparation_time']:.2f}" if metrics['node_preparation_time'] is not None else "N/A"
        job_alloc = f"{metrics['job_allocation_change_time']:.2f}" if metrics['job_allocation_change_time'] is not None else "N/A"
        image_build = f"{metrics['image_building_time']:.2f}" if metrics['image_building_time'] is not None else "N/A"
        rescaling = f"{metrics['rescaling_time']:.2f}" if metrics['rescaling_time'] is not None else "N/A"
        
        print(f"{expected_gpus:<6} {node_prep:<18} {job_alloc:<18} {image_build:<18} {rescaling:<18}")
    else:
        # Process all files in default directory
        directory = "./experiment_results/dummy/cifar10"
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found")
            sys.exit(1)
            
        process_all_logs(directory)

if __name__ == "__main__":
    main()