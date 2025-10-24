#!/usr/bin/env python3
"""
Extract log information for a specific job from JSON log files.
"""

import json
import sys
import matplotlib.pyplot as plt
from datetime import datetime

def extract_job_logs(log_file, job_name):
    """
    Extract all log entries related to a specific job.

    Args:
        log_file: Path to the JSON log file
        job_name: Name of the job to extract (e.g., "cifar10-0")

    Returns:
        Dictionary where keys are timestamps and values are job info
    """
    job_logs = {}

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} - Invalid JSON: {e}", file=sys.stderr)
                continue

            timestamp = log_entry.get('timestamp')
            submitted_jobs = log_entry.get('submitted_jobs', [])

            # Find the specific job in this log entry
            for job in submitted_jobs:
                if job.get('name') == job_name:
                    job_logs[timestamp] = job
                    break

    return job_logs

def print_job_logs(job_logs, job_name):
    """Pretty print the extracted job logs."""
    if not job_logs:
        print(f"No log entries found for job: {job_name}")
        return

    print(f"Found {len(job_logs)} log entries for job: {job_name}\n")
    print("="*80)

    for timestamp, job_info in sorted(job_logs.items()):
        print(f"\nTimestamp: {timestamp}")
        print(f"Job Info:")
        for key, value in job_info.items():
            print(f"  {key}: {value}")
        print("-"*80)

def plot_progress(job_logs, job_name):
    """Plot job progress over time, marking restart points in red."""
    if not job_logs:
        print(f"No log entries to plot for job: {job_name}")
        return

    # Sort by timestamp and extract data
    sorted_logs = sorted(job_logs.items())

    # Extract timestamps, progress values, and detect restarts
    timestamps = []
    progress_values = []
    restart_points = []  # Track indices of restart points

    # Get the first timestamp as reference point
    first_timestamp = sorted_logs[0][0] if sorted_logs else 0
    previous_allocation = None

    for timestamp, job_info in sorted_logs:
        progress = job_info.get('progress')
        if progress is not None:
            # Convert to relative time in seconds from start
            relative_time = timestamp - first_timestamp
            timestamps.append(relative_time)
            progress_values.append(progress)

            # Check if allocation changed (job restarted)
            current_allocation = job_info.get('allocation')
            if previous_allocation is not None and current_allocation != previous_allocation:
                restart_points.append(len(timestamps) - 1)
            previous_allocation = current_allocation

    if not timestamps:
        print(f"No progress data available to plot for job: {job_name}")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot all points
    plt.plot(timestamps, progress_values, linestyle='-', linewidth=2, color='blue', alpha=0.7)

    # Plot normal points in blue
    normal_indices = [i for i in range(len(timestamps)) if i not in restart_points]
    if normal_indices:
        plt.scatter([timestamps[i] for i in normal_indices],
                   [progress_values[i] for i in normal_indices],
                   color='blue', s=60, zorder=3, label='Normal')

    # Plot restart points in red
    if restart_points:
        plt.scatter([timestamps[i] for i in restart_points],
                   [progress_values[i] for i in restart_points],
                   color='red', s=100, zorder=4, marker='o', label='Restart')
        print(f"\nDetected {len(restart_points)} restart(s) at:")
        for idx in restart_points:
            print(f"  Time: {timestamps[idx]:.2f}s, Progress: {progress_values[idx]}")

    plt.xlabel('Time (seconds from start)', fontsize=12)
    plt.ylabel('Progress', fontsize=12)
    plt.title(f'Job Progress Over Time: {job_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_file = f"{job_name}_progress.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nProgress plot saved to: {output_file}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <log_file> <job_name>")
        print("Example: python test.py scheduler.log cifar10-0")
        sys.exit(1)

    log_file = sys.argv[1]
    job_name = sys.argv[2]

    # Extract job logs
    job_logs = extract_job_logs(log_file, job_name)

    # Print results
    # print_job_logs(job_logs, job_name)


    # Plot progress over time
    plot_progress(job_logs, job_name)

    # for timestamp, job_info in job_logs.items():
    #     progress = job_info.get('progress')
    #     if progress is not None and progress > 10295:
    #         print("timestamp: ", timestamp)

