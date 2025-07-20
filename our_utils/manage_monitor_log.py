import json
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np

def process_log_file(log_file_path):
    """Process log file and extract job metrics."""
    jobs = {}  # job_name -> job_info
    node_usage_history = []  # List of (timestamp, total_nodes, effective_nodes)
    previous_jobs = set()  # Track jobs seen in previous log entry
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    total_length = len(lines)
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        cluster_nodes = log_entry.get('cluster_nodes', {})
        total_nodes = cluster_nodes.get('total', 0)
        
        # Calculate effective nodes (actually used nodes)
        effective_nodes = 0
        allocated_nodes = set()
        current_jobs = set()  # Track jobs in current log entry
        
        for job in log_entry['submitted_jobs']:
            job_name = job['name']
            epoch = job['epoch']
            allocation = job.get('allocation', [])
            pod_status = job.get('pod_status', '')
            completion_time = job.get('completion_time')
            
            # Track this job as seen in current log entry
            current_jobs.add(job_name)
            
            # Add allocated nodes to effective count
            for node in allocation:
                allocated_nodes.add(node)
            
            # Initialize job if first time seeing it
            if job_name not in jobs:
                jobs[job_name] = {
                    'first_seen': timestamp,
                    'completion_time': None,
                    'job_type': job_name.split('-')[0],
                    'epochs': {},
                    'wasted_time': 0,
                    'response_time': None,
                    'last_seen_timestamp': timestamp,
                    'last_seen_epoch': epoch,
                    'last_allocation': allocation.copy()
                }
            
            job_info = jobs[job_name]
            
            # Track completion
            if completion_time is not None and job_info['completion_time'] is None:
                job_info['completion_time'] = timestamp
                job_info['response_time'] = timestamp - job_info['first_seen']
            
            # Track epoch information
            if epoch not in job_info['epochs']:
                job_info['epochs'][epoch] = {
                    'first_seen': timestamp,
                    'last_allocation': allocation.copy(),
                    'wasted_time_in_epoch': 0,
                    'epoch_duration': 0
                }
            
            epoch_info = job_info['epochs'][epoch]
            if i < total_length - 1:
                next_log = json.loads(lines[i+1].strip())
                next_timestamp = next_log['timestamp']
                epoch_info['epoch_duration'] = next_timestamp - epoch_info['first_seen']
            
            # Calculate time since last log entry for wasted time calculation
            time_diff = 0 # this is the time interval for monitor, typically 1 second
            if i > 0:
                prev_log = json.loads(lines[i-1].strip())
                time_diff = timestamp - prev_log['timestamp']
            
            # Check for non-normal pod status first
            if pod_status != "pod status normal" and time_diff > 0:
                epoch_info['wasted_time_in_epoch'] += time_diff
            
            # Check for allocation changes within the same epoch (replica changes)
            if set(allocation) != set(epoch_info['last_allocation']):
                # Allocation changed - ALL previous work on this epoch is wasted
                new_wasted = timestamp - epoch_info['first_seen']
                epoch_info['wasted_time_in_epoch'] = new_wasted
                
                # Debug: Check if wasted time exceeds epoch duration
                # epoch_duration = timestamp - epoch_info['first_seen']
                # if new_wasted > epoch_duration + 1:  # +1 for floating point tolerance
                #     print(f"DEBUG: {job_name} epoch {epoch} has impossible wasted time!")
                #     print(f"  Epoch first seen: {epoch_info['first_seen']}")
                #     print(f"  Current timestamp: {timestamp}")
                #     print(f"  Epoch duration: {epoch_duration}")
                #     print(f"  Wasted time: {new_wasted}")
                #     print(f"  Old wasted: {old_wasted}")
            
            # Update tracking info
            epoch_info['last_allocation'] = allocation.copy()
            job_info['last_seen_timestamp'] = timestamp
            job_info['last_seen_epoch'] = epoch
            job_info['last_allocation'] = allocation.copy()
        
        # Check for jobs that disappeared (completed without explicit completion_time)
        # if i > 0:  # Skip first iteration since there's no previous_jobs yet
        #     disappeared_jobs = previous_jobs - current_jobs
        #     for disappeared_job in disappeared_jobs:
        #         if disappeared_job in jobs and jobs[disappeared_job]['completion_time'] is None:
        #             # Job disappeared, mark as completed
        #             jobs[disappeared_job]['completion_time'] = timestamp
        #             jobs[disappeared_job]['response_time'] = timestamp - jobs[disappeared_job]['first_seen']
        #             print(f"Job {disappeared_job} completed (disappeared from log) at {timestamp}")
        # Update previous_jobs for next iteration
        # previous_jobs = current_jobs.copy()
        
        effective_nodes = len(allocated_nodes)
        node_usage_history.append((timestamp, total_nodes, effective_nodes))
    
    # add up wasted time for each job
    for job_name, job_info in jobs.items():
        job_info['wasted_time'] = sum(epoch_info['wasted_time_in_epoch'] for epoch_info in job_info['epochs'].values())
        
        # Debug: Check if total wasted time exceeds response time
        # if job_info['response_time'] and job_info['wasted_time'] > job_info['response_time'] + 1:
        #     print(f"DEBUG: {job_name} has total wasted time > response time!")
        #     print(f"  Response time: {job_info['response_time']}")
        #     print(f"  Total wasted time: {job_info['wasted_time']}")
        #     print(f"  Epochs: {len(job_info['epochs'])}")
        #     for epoch_num, epoch_info in job_info['epochs'].items():
        #         epoch_duration = epoch_info['epoch_duration']
        #         print(f"    Epoch {epoch_num}: wasted={epoch_info['wasted_time_in_epoch']:.2f}, duration: {epoch_duration:.2f}")
    
    return jobs, node_usage_history

def calculate_metrics(jobs, node_usage_history):
    """Calculate summary metrics."""
    response_times = []
    job_types = defaultdict(list)
    
    # Calculate response times
    for job_name, job_info in jobs.items():
        if job_info['response_time'] is not None:
            response_times.append(job_info['response_time'])
            job_types[job_info['job_type']].append(job_info['response_time'])
    
    # Calculate node usage metrics
    total_node_seconds = 0
    total_effective_node_seconds = 0
    total_time = 0
    
    for i in range(1, len(node_usage_history)):
        prev_time, prev_total, prev_effective = node_usage_history[i-1]
        curr_time, curr_total, curr_effective = node_usage_history[i]
        
        time_diff = curr_time - prev_time
        total_node_seconds += prev_total * time_diff
        total_effective_node_seconds += prev_effective * time_diff
        total_time += time_diff
    
    # Calculate metrics
    metrics = {
        'job_response_times': {name: info['response_time'] for name, info in jobs.items()},
        'mean_response_time_by_type': {job_type: np.mean(times) for job_type, times in job_types.items()},
        'mean_response_time_overall': np.mean(response_times) if response_times else 0,
        'percentile_90_response_time': np.percentile(response_times, 90) if response_times else 0,
        'total_node_seconds': total_node_seconds,
        'average_node_seconds': total_node_seconds / total_time if total_time > 0 else 0,
        'average_effective_node_seconds': total_effective_node_seconds / total_time if total_time > 0 else 0,
        'wasted_times': {name: info['wasted_time'] for name, info in jobs.items()},
        'total_time': total_time
    }
    
    return metrics

def print_summary(metrics):
    """Print summary of metrics."""
    print("\n" + "="*70)
    print("LOG PROCESSING SUMMARY")
    print("="*70)
    
    print("\n1. Response Time of All Jobs:")
    print("-" * 40)
    for job_name, response_time in metrics['job_response_times'].items():
        wasted_time = metrics['wasted_times'][job_name]
        if response_time is not None:
            print(f"  {job_name}: {response_time:.2f} seconds (wasted: {wasted_time:.2f} seconds)")
        else:
            print(f"  {job_name}: Not completed (wasted: {wasted_time:.2f} seconds)")
    
    print("\n2. Mean Response Time by Job Type:")
    print("-" * 40)
    for job_type, mean_time in metrics['mean_response_time_by_type'].items():
        print(f"  {job_type}: {mean_time:.2f} seconds")
    
    print(f"\n3. Mean Response Time Overall: {metrics['mean_response_time_overall']:.2f} seconds")
    print(f"4. 90th Percentile Response Time: {metrics['percentile_90_response_time']:.2f} seconds")
    print(f"5. Total Node-Second Usage: {metrics['total_node_seconds']:.2f} node-seconds")
    print(f"6. Average Node-Second Usage: {metrics['average_node_seconds']:.2f} nodes")
    print(f"7. Average Effective Node-Second: {metrics['average_effective_node_seconds']:.2f} nodes")
    
    print(f"Total Experiment Time: {metrics['total_time']:.2f} seconds")

    

def main():
    if len(sys.argv) != 2:
        print("Usage: python manage_monitor_log.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    print(f"Processing log file: {log_file_path}")
    
    jobs, node_usage_history = process_log_file(log_file_path)
    metrics = calculate_metrics(jobs, node_usage_history)
    print_summary(metrics)
    # job_info = jobs['cifar10-21']
    # for epoch, epoch_info in job_info['epochs'].items():
    #     print(epoch, epoch_info['wasted_time_in_epoch'], epoch_info['epoch_duration'])

if __name__ == "__main__":
    main()
