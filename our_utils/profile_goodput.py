import json
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
from plot_config import apply_plot_style

# Import goodput_functions from goodput-fix-profile.py
script_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("goodput_fix_profile",
                                               os.path.join(script_dir, "goodput-fix-profile.py"))
goodput_fix_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(goodput_fix_profile)
goodput_functions = goodput_fix_profile.goodput_functions

# Dataset sizes and atomic batch sizes
DATASET_SIZES = {
    'bert': 97077,
    'cifar10': 50000,
    'deepspeech2': 4074
}

ATOMIC_BATCH_SIZES = {
    'bert': 12,
    'cifar10': 128,
    'deepspeech2': 20
}

# Hardcoded list of log files to process

LOG_FILES = [
    # "experiment_results/0103-FW-b56/monitor_log.txt",
    # "experiment_results/0102-FW-b68/monitor_log.txt",
    # "experiment_results/0104-FW-b32/monitor_log.txt",
    # "experiment_results/1221-FW-b68/monitor_log.txt",
    # "experiment_results/1220-FW-b48-2/monitor_log.txt",
]

job_epoch_list = [
      ('cifar10', 0), ('cifar10', 10), ('cifar10', 40),
      ('bert', 0), ('bert', 1), ('cifar10', 80),
      ('deepspeech2', 0), ('deepspeech2', 5), ('deepspeech2', 20),
  ]


def process_log_file(log_file_path):
    """Process a single log file and extract goodput data."""
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Track job epochs: {job_name: {epoch: {timestamps, progress_values, allocations}}}
    job_epochs = defaultdict(lambda: defaultdict(lambda: {
        'timestamps': [],
        'progress': [],
        'allocations': []
    }))

    # Read through log and collect progress data
    for line in lines:
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']

        for job in log_entry.get('submitted_jobs', []):
            job_name = job['name']
            epoch = job.get('epoch', 0)
            progress = job.get('progress', None)
            allocation = job.get('allocation', [])
            num_gpus = len(allocation) if allocation else 0

            if progress is not None:
                epoch_data = job_epochs[job_name][epoch]
                epoch_data['timestamps'].append(timestamp)
                epoch_data['progress'].append(float(progress))
                epoch_data['allocations'].append(num_gpus)

    return job_epochs

def calculate_goodput_for_epoch(timestamps, progress_values, allocations, job_type, next_epoch_first_timestamp=None, next_epoch_first_progress=None, job_name=None, epoch=None):
    """Calculate goodput for a single epoch.

    Args:
        timestamps: List of timestamps for this epoch
        progress_values: List of progress values for this epoch
        allocations: List of GPU allocations for this epoch
        job_type: Type of job (bert, cifar10, deepspeech2)
        next_epoch_first_timestamp: First timestamp of next epoch (if not last epoch)
        next_epoch_first_progress: First progress value of next epoch (if not last epoch)
        job_name: Name of the job (for error messages)
        epoch: Epoch number (for error messages)

    Returns: (num_gpus, goodput) where goodput is progress per second during running periods.
    """
    # Determine GPU count and starting point
    gpu_counts = [g for g in allocations if g > 0]
    if not gpu_counts:
        return None, None

    unique_gpu_counts = list(set(gpu_counts))

    # Handle multiple GPU counts
    if len(unique_gpu_counts) > 1:
        # Special case for bert with exactly 2 different GPU counts
        if job_type == 'bert' and len(unique_gpu_counts) == 2:
            # Use the second (final) GPU count
            # Sort to get consistent ordering
            unique_gpu_counts.sort()
            num_gpus = unique_gpu_counts[1]  # The larger/later one

            # Find first index where we use this final GPU count
            start_idx = None
            for idx, g in enumerate(gpu_counts):
                if g == num_gpus:
                    start_idx = idx
                    break

            if start_idx is None:
                return None, None
        else:
            print(f"Warning: Job {job_name} epoch {epoch} has multiple GPU counts: {set(gpu_counts)}; Skipping this epoch.")
            return None, None
    else:
        num_gpus = unique_gpu_counts[0]
        start_idx = 0

    # Identify rescaling periods: continuous stagnation >20 seconds
    # Calculate running time (excluding rescaling periods)
    # Start from start_idx
    running_time = 0.0
    first_progress = progress_values[start_idx] if start_idx < len(progress_values) else None

    i = start_idx
    while i < len(timestamps) - 1:
        current_time = timestamps[i]
        current_progress = progress_values[i]
        next_time = timestamps[i + 1]
        next_progress = progress_values[i + 1]
        time_diff = next_time - current_time

        if next_progress > current_progress:
            # Progress is growing in this interval - this is running time
            running_time += time_diff
            i += 1
        else:
            # Progress is stagnant. Check how long it stays stagnant
            stagnant_start = current_time
            stagnant_progress = current_progress
            j = i + 1

            # Look ahead to find when progress starts growing again
            while j < len(timestamps):
                if progress_values[j] > stagnant_progress:
                    # Progress starts growing again
                    break
                j += 1

            if j < len(timestamps):
                stagnant_duration = timestamps[j] - stagnant_start
                if stagnant_duration <= 20:
                    # Short stagnation (≤20s), still count as running time
                    running_time += stagnant_duration
                else:
                    # Long stagnation (>20s), this is rescaling - don't count this time
                    pass
                # Move to where progress resumes
                i = j
            else:
                # Stagnant until end
                break

    # Add the interval between last timestamp of this epoch and first timestamp of next epoch
    # (This is the time between epochs, should always be counted as running time)
    if next_epoch_first_timestamp is not None and len(timestamps) > 0:
        inter_epoch_interval = next_epoch_first_timestamp - timestamps[-1]
        running_time += inter_epoch_interval

    # Calculate actual progress made (from start_idx to next epoch or end of current epoch)
    atomic_batch_size = ATOMIC_BATCH_SIZES.get(job_type)
    if atomic_batch_size is None:
        return None, None

    if first_progress is None:
        return None, None

    # Determine the ending progress
    if next_epoch_first_progress is not None:
        end_progress = next_epoch_first_progress
    elif len(progress_values) > 0:
        end_progress = progress_values[-1]
    else:
        return None, None

    actual_progress_made = end_progress - first_progress

    # Sanity check: verify progress is reasonable (allow 50% tolerance for partial epochs with GPU changes)
    dataset_size = DATASET_SIZES.get(job_type)
    if dataset_size is not None:
        expected_progress = dataset_size / atomic_batch_size
        if actual_progress_made > expected_progress * 1.2 or actual_progress_made < 0:
            print(f"Warning: Job {job_name} epoch {epoch} progress {actual_progress_made:.2f} seems unreasonable (expected ~{expected_progress:.2f})")
            return None, None

    if running_time > 0 and actual_progress_made > 0:
        # Goodput is (actual_progress / running_time) * atomic_batch_size (samples per second)
        goodput_samples_per_sec = (actual_progress_made / running_time) * atomic_batch_size
        return num_gpus, goodput_samples_per_sec

    return None, None


def build_goodput_profile(log_files):
    """Build goodput profile from multiple log files.

    Returns: {job_type: {epoch: {num_gpus: [list of goodput values]}}}
    """
    goodput_profile = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for log_file in log_files:
        print(f"Processing {log_file}...")
        job_epochs = process_log_file(log_file)

        for job_name, epochs in job_epochs.items():
            job_type = job_name.split('-')[0] if '-' in job_name else job_name

            # Sort epochs to process in order
            sorted_epochs = sorted(epochs.keys())

            for idx, epoch in enumerate(sorted_epochs):
                epoch_data = epochs[epoch]

                # Get first timestamp and progress of next epoch (if not last epoch)
                next_epoch_first_timestamp = None
                next_epoch_first_progress = None
                if idx < len(sorted_epochs) - 1:
                    next_epoch = sorted_epochs[idx + 1]
                    next_epoch_timestamps = epochs[next_epoch]['timestamps']
                    next_epoch_progress = epochs[next_epoch]['progress']
                    if len(next_epoch_timestamps) > 0:
                        next_epoch_first_timestamp = next_epoch_timestamps[0]
                    if len(next_epoch_progress) > 0:
                        next_epoch_first_progress = next_epoch_progress[0]

                num_gpus, goodput = calculate_goodput_for_epoch(
                    epoch_data['timestamps'],
                    epoch_data['progress'],
                    epoch_data['allocations'],
                    job_type,
                    next_epoch_first_timestamp,
                    next_epoch_first_progress,
                    job_name,
                    epoch
                )

                if num_gpus is not None and goodput is not None:
                    goodput_profile[job_type][epoch][num_gpus].append(goodput)

    return goodput_profile


def plot_goodput_comparison(goodput_profile, job_epoch_list, goodput_functions, output_file=None):
    """Plot goodput comparison between theoretical functions and actual measurements.

    Args:
        goodput_profile: Dictionary {job_type: {epoch: {num_gpus: [goodput values]}}}
        job_epoch_list: List of 9 (job_type, epoch) tuples to plot
        goodput_functions: Dictionary {job_type: {epoch: {num_gpus: goodput}}} from goodput-fix-profile.py
        output_file: Optional filename to save the figure
    """
    apply_plot_style()

    if len(job_epoch_list) != 9:
        raise ValueError("job_epoch_list must contain exactly 9 (job_type, epoch) tuples")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (job_type, epoch) in enumerate(job_epoch_list):
        ax = axes[idx]

        # Plot theoretical goodput function in red
        if job_type in goodput_functions and epoch in goodput_functions[job_type]:
            theoretical_data = goodput_functions[job_type][epoch]
            gpu_counts = sorted(theoretical_data.keys())
            goodputs = [theoretical_data[g] for g in gpu_counts]
            ax.plot(gpu_counts, goodputs, 'r-', label='Theoretical')

        # Plot actual mean goodput in blue
        if job_type in goodput_profile and epoch in goodput_profile[job_type]:
            actual_data = goodput_profile[job_type][epoch]
            gpu_counts_actual = sorted(actual_data.keys())
            mean_goodputs = []
            for g in gpu_counts_actual:
                values = actual_data[g]
                if values:
                    mean_goodputs.append(sum(values) / len(values))
                else:
                    mean_goodputs.append(0)

            ax.plot(gpu_counts_actual, mean_goodputs, 'bo', label='Measured (mean)')

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Goodput (samples/sec)')
        ax.set_title(f'{job_type} - Epoch {epoch}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()


def plot_speedup_and_gpu_hours(goodput_functions, output_file_prefix=None):
    """Plot speedup curves and total GPU hours for cifar10.

    Creates two figures:
    1. Speedup function: Shows speedup vs number of GPUs for cifar10 at epochs 1, 10, 50
       - Speedup = goodput[k GPUs] / goodput[1 GPU]
       - Includes a perfect speedup reference line (y=x)

    2. Total GPU hours: Shows total GPU hours needed to train cifar10 for all epochs
       - GPU hours per epoch = (dataset_size / goodput) * num_gpus / 3600
       - Summed across all available epochs

    Args:
        goodput_functions: Dictionary {job_type: {epoch: {num_gpus: goodput}}}
        output_file_prefix: Optional prefix for output filenames (e.g., "cifar10_analysis")
    """
    apply_plot_style()

    cifar10_data = goodput_functions.get('cifar10', {})
    if not cifar10_data:
        print("Error: No cifar10 data found in goodput_functions")
        return

    dataset_size = DATASET_SIZES['cifar10']

    # Figure 1: Speedup curves
    print("Creating speedup plot...")
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    # Get epochs of interest: 1, 10, 50
    epochs_to_plot = [1, 10, 50]
    colors = ['blue', 'green', 'red']

    # Collect all GPU counts to determine the range for perfect speedup line
    all_gpu_counts = set()

    for epoch, color in zip(epochs_to_plot, colors):
        if epoch in cifar10_data:
            epoch_data = cifar10_data[epoch]
            gpu_counts = sorted(epoch_data.keys())
            all_gpu_counts.update(gpu_counts)

            # Get baseline (1 GPU) goodput
            if 1 not in epoch_data:
                print(f"Warning: No 1-GPU data for epoch {epoch}, skipping")
                continue

            baseline_goodput = epoch_data[1]

            # Calculate speedup for each GPU count
            speedups = []
            for k in gpu_counts:
                speedup = epoch_data[k] / baseline_goodput
                speedups.append(speedup)

            ax1.plot(gpu_counts, speedups, marker='o', color=color, label=f'Epoch {epoch}')

    # Plot perfect speedup (45-degree line, y=x)
    if all_gpu_counts:
        max_gpu = max(all_gpu_counts)
        perfect_speedup_x = list(range(1, max_gpu + 1))
        perfect_speedup_y = perfect_speedup_x
        ax1.plot(perfect_speedup_x, perfect_speedup_y, 'k--', label='Perfect Speedup')

    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Speedup')
    # ax1.set_ylim(0, 15)
    ax1.set_xticks([1, 2, 4, 8, 12, 16, 24, 32])
    ax1.legend()
    # ax1.grid(True, alpha=0.3)

    if output_file_prefix:
        speedup_file = f"{output_file_prefix}_speedup.png"
        fig1.savefig(speedup_file, dpi=150, bbox_inches='tight')
        print(f"Speedup figure saved to {speedup_file}")

    # Figure 2: Total GPU hours
    print("Creating GPU hours plot...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Get all available epochs for cifar10
    all_epochs = sorted(cifar10_data.keys())
    print(f"Calculating total GPU hours across {len(all_epochs)} epochs (epoch {min(all_epochs)} to {max(all_epochs)})...")

    # Get all GPU counts from the first epoch
    if all_epochs:
        gpu_counts = sorted(cifar10_data[all_epochs[0]].keys())

        # For each GPU count, calculate total GPU hours
        total_gpu_hours = {k: 0.0 for k in gpu_counts}

        for epoch in all_epochs:
            epoch_data = cifar10_data[epoch]
            for k in gpu_counts:
                if k in epoch_data:
                    goodput = epoch_data[k]
                    # GPU hours for this epoch = (dataset_size / goodput) * k / 3600
                    # goodput is in samples/second
                    # dataset_size / goodput gives time in seconds to process all samples
                    # multiply by k to get GPU-seconds (total compute across all GPUs)
                    # divide by 3600 to convert seconds to hours
                    gpu_hours_for_epoch = (dataset_size / goodput) * k / 3600
                    total_gpu_hours[k] += gpu_hours_for_epoch

        # Plot
        gpu_counts_list = sorted(total_gpu_hours.keys())
        gpu_hours_list = [total_gpu_hours[k] for k in gpu_counts_list]

        ax2.plot(gpu_counts_list, gpu_hours_list, marker='o', color='purple')

        # Print some statistics
        print("\nTotal GPU hours for different GPU counts:")
        for k in gpu_counts_list:
            print(f"  {k} GPUs: {total_gpu_hours[k]:.2f} GPU-hours")

    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Total GPU Hours')
    ax2.set_xticks([1, 2, 4, 8, 12, 16, 24, 32])
    # ax2.grid(True, alpha=0.3)

    if output_file_prefix:
        gpu_hours_file = f"{output_file_prefix}_gpu_hours.png"
        fig2.savefig(gpu_hours_file, dpi=150, bbox_inches='tight')
        print(f"GPU hours figure saved to {gpu_hours_file}")

    if not output_file_prefix:
        plt.show()


def main():
    log_files = LOG_FILES
    print(f"Processing {len(log_files)} log file(s)...")
    goodput_profile = build_goodput_profile(log_files)

    plot_goodput_comparison(goodput_profile, job_epoch_list, goodput_functions, output_file="goodput_comparison.png")

    # Generate speedup and GPU hours plots
    print("\n" + "="*60)
    print("Generating speedup and GPU hours analysis...")
    print("="*60)
    plot_speedup_and_gpu_hours(goodput_functions, output_file_prefix="cifar10_analysis")

if __name__ == "__main__":
    main()
