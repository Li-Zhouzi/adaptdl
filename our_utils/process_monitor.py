import json
from datetime import datetime
import sys

def process_log_file(log_file_path):
    """Process log file and return job information."""
    jobs = {}  # {job_name: {arrival_time, epoch_data}}
    job_current_epochs = {}  # Track current epoch for each job
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                timestamp = log_entry['timestamp']
                
                for job in log_entry['submitted_jobs']:
                    job_name = job['name']
                    epoch = job['epoch']
                    allocation = job['allocation']
                    completion_time = job['completion_time']
                    
                    # Initialize job if first time seeing it
                    if job_name not in jobs:
                        jobs[job_name] = {
                            'arrival_time': datetime.fromtimestamp(timestamp),
                            'epoch_data': {}
                        }
                        job_current_epochs[job_name] = epoch
                    
                    # Initialize epoch if first time seeing it
                    if epoch not in jobs[job_name]['epoch_data']:
                        jobs[job_name]['epoch_data'][epoch] = {
                            'num_gpu': [len(allocation)],  # List to store all GPU allocations
                            'completion_time': None,
                            'start_time': timestamp
                        }
                    else:
                        # Add new GPU allocation if different from last one
                        current_gpus = jobs[job_name]['epoch_data'][epoch]['num_gpu']
                        if len(allocation) not in current_gpus:
                            current_gpus.append(len(allocation))
                    
                    # Check if we're moving to a new epoch or job is complete
                    assert job_name in job_current_epochs, f"Job {job_name} not found in job_current_epochs"
                    current_epoch = job_current_epochs[job_name]
                    if epoch > current_epoch or completion_time is not None:
                        # Handle missing epochs

                        # Calculate time interval for missing epochs
                        start_time = jobs[job_name]['epoch_data'][current_epoch]['start_time']
                        time_interval = timestamp - start_time
                        num_total_epochs = epoch - current_epoch
                        time_per_epoch = time_interval / num_total_epochs
                        
                        # Distribute time among missing epochs
                        for t_epoch in range(current_epoch, epoch):
                            if t_epoch == current_epoch:
                                assert t_epoch in jobs[job_name]['epoch_data'], f"Epoch {t_epoch} not found in jobs[job_name]['epoch_data']"
                                jobs[job_name]['epoch_data'][t_epoch]['completion_time'] = time_per_epoch
                            else:
                                assert t_epoch not in jobs[job_name]['epoch_data'], f"Epoch {t_epoch} already exists"
                                # Initialize missing epoch with same GPU allocation as current epoch
                                jobs[job_name]['epoch_data'][t_epoch] = {
                                    'num_gpu': jobs[job_name]['epoch_data'][current_epoch]['num_gpu'],
                                    'completion_time': time_per_epoch,
                                    'start_time': start_time + (t_epoch - current_epoch) * time_per_epoch
                                }
                                

                        
                        # Update current epoch
                        job_current_epochs[job_name] = epoch
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {e}")
    
    return jobs

def print_results(jobs):
    """Print job information in a readable format."""
    print("\n" + "="*50)
    print("Job Processing Results")
    print("="*50)
    
    for job_name, job_info in jobs.items():
        print(f"\nJob: {job_name}")
        print(f"Arrival Time: {job_info['arrival_time']}")
        print("Epoch Data:")
        for epoch, epoch_info in job_info['epoch_data'].items():
            print(f"  Epoch {epoch}:")
            print(f"   Number of GPU: {epoch_info['num_gpu']}")
            if epoch_info['completion_time'] is not None:
                print(f"    Completion Time: {epoch_info['completion_time']:.2f} seconds")
            else:
                print("    Status: Incomplete")
    print("\n" + "="*50)

def main():
    if len(sys.argv) != 2:
        print("Error: Missing log file path!")
        sys.exit(1)
    
    log_file = sys.argv[1]
    jobs = process_log_file(log_file)
    print_results(jobs)

if __name__ == "__main__":
    main() 