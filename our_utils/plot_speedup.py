import matplotlib.pyplot as plt
import sys
from process_monitor import process_log_file
import os

def plot_epoch_completion_times(log_files):
    """
    Plot epoch completion times from multiple log files.
    Each log file should contain one job, and will be represented as one curve.
    
    Args:
        log_files: List of paths to log files
    """
    plt.figure(figsize=(10, 6))
    
    for log_file in log_files:
        # Process the log file
        jobs = process_log_file(log_file)
        
        # Each log should have exactly one job
        assert len(jobs) == 1, f"Expected 1 job, got {len(jobs)}"
        job_name = list(jobs.keys())[0]
        job_info = jobs[job_name]
        
        # Extract epochs and completion times
        epochs = []
        completion_times = []
        
        for epoch, epoch_info in sorted(job_info['epoch_data'].items()):
            if epoch_info['completion_time'] is not None:
                epochs.append(epoch)
                completion_times.append(epoch_info['completion_time'])
        
        # Plot the curve
        label = os.path.basename(log_file)  # Use filename as label
        plt.plot(epochs, completion_times, 'o-', label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Completion Time (seconds)')
    plt.title('Epoch Completion Times Across Different Runs')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('epoch_completion_times.png')
    print("Plot saved as 'epoch_completion_times.png'")

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide at least one log file path!")
        print("Usage: python plot_speedup.py <log_file1> [log_file2 ...]")
        sys.exit(1)
    
    log_files = sys.argv[1:]
    plot_epoch_completion_times(log_files)

if __name__ == "__main__":
    main()
