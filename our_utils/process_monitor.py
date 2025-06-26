import json
from datetime import datetime
import sys
from collections import defaultdict
import inspect
import re


def _get_target_gpus(filename):
    """Extract target GPU number from filename (e.g., 4gpus.txt -> 4)"""
    match = re.search(r'(\d+)gpus\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


def _handle_job_completion(job_info, completed_epoch, timestamp):
    """When see a sign that an epoch is completed, update the job_info"""
    """Called when see a job completed"""
    epoch_data = job_info['epoch_data'][completed_epoch]
    last_phase = epoch_data['phases'][-1]
    if timestamp - last_phase['start_time'] < 2: # a very short phase at the end of a job
        last_phase = epoch_data['phases'][-2]
    epoch_data['running_time'] = timestamp - last_phase['start_time']
    epoch_data['rescaling_time'] = (timestamp - epoch_data['start_time']) - epoch_data['running_time']


def _handle_epoch_completion(job_info, new_epoch, timestamp):
    """Called when see a new epoch"""
    """there might be missing epochs"""
    last_epoch = list(job_info['epoch_data'].keys())[-1]
    last_phase = job_info['epoch_data'][last_epoch]['phases'][-1]
    line_no = inspect.currentframe().f_lineno
    assert last_phase['pods_status'].get('Running', 0) == last_phase['num_gpu'], \
        f"Line {line_no}: Last phase, pods should be all running. Found {last_phase['pods_status'].get('Running', 0)} running pods but {last_phase['num_gpu']} GPUs"

    start_time = last_phase['start_time']
    time_interval = timestamp - start_time
    num_total_epochs = new_epoch - last_epoch
    time_per_epoch = time_interval / num_total_epochs
    
    for t_epoch in range(last_epoch, new_epoch):
        if t_epoch > last_epoch:
            # This epoch is missing
            # Initialize missing epoch with same phase as current epoch
            job_info['epoch_data'][t_epoch] = {
                'phases': [],
                'running_time': time_per_epoch,  # the missing epochs are completed quickly, so no rescaling time
                'rescaling_time': 0,  
                'start_time': start_time + (t_epoch - last_epoch) * time_per_epoch
            }
        else:
            job_info['epoch_data'][t_epoch]['running_time'] = time_per_epoch
            job_info['epoch_data'][t_epoch]['rescaling_time'] = start_time - job_info['epoch_data'][t_epoch]['start_time']


def process_log_file(log_file_path):
    """Process log file and return job information."""
    """Format: return a dictionary of jobs: {job_name: {arrival_time, epoch_data, is_completed}} and log metrics {autoscaling_time, gpu_ready_time}"""
    """epoch_data: {epoch: {phases: [phase_dicts], running_time, rescaling_time, start_time}}"""
    """phase_dict: {num_gpu, num_pods, pods_status: {Running: count, Pending: count, ...}, start_time: float}"""
    jobs = {}  # {job_name: {arrival_time, epoch_data, is_completed}}
    target_gpus = _get_target_gpus(log_file_path)
    if target_gpus is None:
        raise ValueError(f"Could not determine target GPU number from filename: {log_file_path}")
    
    # Log level metrics
    log_metrics = {
        'autoscaling_time': None,
        'gpu_ready_time': None,
        '_first_running_time': None,
        '_total_gpu_reached_time': None,
        '_ready_gpu_reached_time': None
    }
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                timestamp = log_entry['timestamp']
                cluster_nodes = log_entry.get('cluster_nodes', {})
                total_gpus = cluster_nodes.get('total', 0)
                ready_gpus = cluster_nodes.get('ready', 0)
                
                # Track autoscaling and GPU ready times at log level
                if log_metrics['_first_running_time'] is None and len(log_entry['submitted_jobs']) > 0:
                    log_metrics['_first_running_time'] = timestamp
                
                if log_metrics['_total_gpu_reached_time'] is None and total_gpus >= target_gpus:
                    log_metrics['_total_gpu_reached_time'] = timestamp
                    if log_metrics['_first_running_time'] is not None:
                        log_metrics['autoscaling_time'] = timestamp - log_metrics['_first_running_time']
                
                if log_metrics['_ready_gpu_reached_time'] is None and ready_gpus >= target_gpus:
                    log_metrics['_ready_gpu_reached_time'] = timestamp
                    if log_metrics['_total_gpu_reached_time'] is not None:
                        log_metrics['gpu_ready_time'] = timestamp - log_metrics['_total_gpu_reached_time']
                
                for job in log_entry['submitted_jobs']:
                    job_name = job['name']
                    epoch = job['epoch']
                    allocation = job['allocation']
                    pod_status = job['pod_status']
                    completion_time = job['completion_time']
                    
                    # Skip if job is already marked as completed
                    if job_name in jobs and jobs[job_name]['is_completed']:
                        continue

                    # Check if job is completed
                    if completion_time is not None:
                        jobs[job_name]['is_completed'] = True
                        # Update the last epoch's timing information
                        last_epoch = list(jobs[job_name]['epoch_data'].keys())[-1]
                        line_no = inspect.currentframe().f_lineno
                        assert last_epoch == epoch, \
                            f"Line {line_no}: Last epoch {last_epoch} does not match completed epoch {epoch}"
                        _handle_job_completion(jobs[job_name], last_epoch, timestamp)
                        continue  # Skip further processing for completed job

                    # Check if epoch is completed
                    if job_name in jobs:
                        last_epoch = list(jobs[job_name]['epoch_data'].keys())[-1]
                        if epoch > last_epoch:
                            # Handle missing epochs
                            _handle_epoch_completion(jobs[job_name], epoch, timestamp)

                    # Now update information for the current epoch
                    # Initialize job if first time seeing it
                    if job_name not in jobs:
                        jobs[job_name] = {
                            'arrival_time': datetime.fromtimestamp(timestamp), # only used for printing
                            'epoch_data': {},
                            'is_completed': False
                        }
                    
                    # Initialize epoch if first time seeing it
                    if epoch not in jobs[job_name]['epoch_data']:
                        jobs[job_name]['epoch_data'][epoch] = {
                            'phases': [],
                            'running_time': None,
                            'rescaling_time': None,
                            'start_time': timestamp
                        }

                    # Update epoch phases information
                    # Count pod statuses
                    pod_status_counts = defaultdict(int)
                    for pod_info in pod_status.values():
                        phase = pod_info['phase']
                        pod_status_counts[phase] += 1
                    
                    total_pods = sum(pod_status_counts.values())
                    
                    # Create current phase info
                    current_phase = {
                        'num_gpu': len(allocation),
                        'num_pods': total_pods,
                        'pods_status': dict(pod_status_counts),
                    }
                    
                    phase_changed = False 
                    # Check if we need to create a new phase
                    if len(jobs[job_name]['epoch_data'][epoch]['phases']) == 0:
                        phase_changed = True
                    else: 
                        last_phase = jobs[job_name]['epoch_data'][epoch]['phases'][-1]      
                        # Check if GPU allocation changed
                        if last_phase['num_gpu'] != current_phase['num_gpu']:
                            phase_changed = True
                        # Check if pod status counts changed
                        elif last_phase['pods_status'] != current_phase['pods_status']:
                            phase_changed = True
                    
                    if phase_changed:                        
                        # Add new phase
                        current_phase['start_time'] = timestamp
                        jobs[job_name]['epoch_data'][epoch]['phases'].append(current_phase)
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {e}")
                print(f"Line content: {line.strip()}")
                raise  # Re-raise the exception to see the full traceback
    
    # Return both jobs and log metrics
    return {
        'jobs': jobs,
        'log_metrics': {
            'autoscaling_time': log_metrics['autoscaling_time'],
            'gpu_ready_time': log_metrics['gpu_ready_time']
        }
    }

def print_results(result):
    """Print job information in a readable format."""
    jobs = result['jobs']
    log_metrics = result['log_metrics']
    
    print("\n" + "="*50)
    print("Log Metrics")
    print("="*50)
    if log_metrics['autoscaling_time'] is not None:
        print(f"Autoscaling Time: {log_metrics['autoscaling_time']:.2f} seconds")
    else:
        print("Autoscaling Time: Not reached")
    if log_metrics['gpu_ready_time'] is not None:
        print(f"GPU Ready Time: {log_metrics['gpu_ready_time']:.2f} seconds")
    else:
        print("GPU Ready Time: Not reached")
    
    print("\n" + "="*50)
    print("Job Processing Results")
    print("="*50)
    
    for job_name, job_info in jobs.items():
        print(f"\nJob: {job_name}")
        print(f"Arrival Time: {job_info['arrival_time']}")
        print(f"Status: {'Completed' if job_info['is_completed'] else 'Running'}")
        print("Epoch Data:")
        for epoch, epoch_info in job_info['epoch_data'].items():
            print(f"  Epoch {epoch}:")
            print(f"   Start Time: {datetime.fromtimestamp(epoch_info['start_time'])}")
            print(f"   Running Time: {epoch_info['running_time']:.2f} seconds")
            print(f"   Rescaling Time: {epoch_info['rescaling_time']:.2f} seconds")
            print("   Phases:")
            for i, phase in enumerate(epoch_info['phases']):
                print(f"    Phase {i+1}:")
                print(f"     Start Time: {datetime.fromtimestamp(phase['start_time'])}")
                print(f"     Number of GPUs: {phase['num_gpu']}")
                print(f"     Number of Pods: {phase['num_pods']}")
                print(f"     Pod Status: {phase['pods_status']}")
    print("\n" + "="*50)

def main():
    if len(sys.argv) != 2:
        print("Error: Missing log file path!")
        sys.exit(1)
    
    log_file = sys.argv[1]
    result = process_log_file(log_file)
    print_results(result)

if __name__ == "__main__":
    main() 