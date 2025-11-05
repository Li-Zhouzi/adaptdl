
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    # Import the existing log processing function
    from our_utils.manage_monitor_log import process_log_file
except Exception as e:
    raise RuntimeError("Failed to import process_log_file from our_utils.manage_monitor_log") from e


def compute_rescaling_times(jobs):
    """Compute per-event rescaling times from jobs, mirroring manage_monitor_log logic.

    For each epoch where rescaling is detected, we attribute the epoch's wasted_time
    to the rescaling events (evenly divided when multiple rescalings occur within
    a single epoch), producing a list of per-event durations in seconds.
    """
    per_event_times = []

    for job_name, job_info in jobs.items():
        epochs = sorted(job_info.get('epochs', {}).items())
        if not epochs:
            continue

        previous_final_alloc = None
        previous_alloc_len = 0

        for idx, (epoch_num, epoch_info) in enumerate(epochs):
            gpu_allocations = epoch_info.get('gpu_allocations', [])
            current_alloc_len = len(gpu_allocations)
            current_final_alloc = gpu_allocations[-1] if gpu_allocations else 0

            rescale_count = max(0, current_alloc_len - 1)

            # Handle alloc length 1 but final alloc changes between epochs
            if (
                previous_final_alloc is not None
                and previous_alloc_len == 1
                and current_alloc_len == 1
                and current_final_alloc != previous_final_alloc
            ):
                rescale_count += 1

            # Ignore special cases consistent with manage_monitor_log
            if idx == len(epochs) - 1 and rescale_count > 0:
                rescale_count = 0  # ignore the last epoch's rescaling to 0
            if idx == 0 and 0 in gpu_allocations and 1 in gpu_allocations:
                rescale_count -= 1  # ignore the first epoch's rescaling to 1

            if rescale_count > 0:
                wasted_time = float(epoch_info.get('wasted_time', 0) or 0)
                if wasted_time <= 0:
                    if idx + 1 < len(epochs):
                        next_wasted = float(epochs[idx + 1][1].get('wasted_time', 0) or 0)
                        if next_wasted > 0:
                            wasted_time = next_wasted
                        else:
                            wasted_time = 0
                    else:
                        wasted_time = 0

                if wasted_time > 0:
                    per_event_time = wasted_time / rescale_count
                    per_event_times.extend([per_event_time] * rescale_count)

            previous_final_alloc = current_final_alloc
            previous_alloc_len = current_alloc_len

    return per_event_times


def compute_rescaling_times_by_type(jobs):
    """Compute per-event rescaling times grouped by job type.

    Returns a dict: { 'cifar10': [...], 'bert': [...], 'deepspeech2': [...] }.
    """
    per_type = {
        'cifar10': [],
        'bert': [],
        'deepspeech2': [],
    }

    for job_name, job_info in jobs.items():
        job_type = job_name.split('-')[0] if '-' in job_name else job_name
        if job_type not in per_type:
            continue

        epochs = sorted(job_info.get('epochs', {}).items())
        if not epochs:
            continue

        previous_final_alloc = None
        previous_alloc_len = 0

        for idx, (epoch_num, epoch_info) in enumerate(epochs):
            gpu_allocations = epoch_info.get('gpu_allocations', [])
            current_alloc_len = len(gpu_allocations)
            current_final_alloc = gpu_allocations[-1] if gpu_allocations else 0

            rescale_count = max(0, current_alloc_len - 1)

            if (
                previous_final_alloc is not None
                and previous_alloc_len == 1
                and current_alloc_len == 1
                and current_final_alloc != previous_final_alloc
            ):
                rescale_count += 1

            if idx == len(epochs) - 1 and rescale_count > 0:
                rescale_count = 0
            if idx == 0 and 0 in gpu_allocations and 1 in gpu_allocations:
                rescale_count -= 1

            if rescale_count > 0:
                wasted_time = float(epoch_info.get('wasted_time', 0) or 0)
                if wasted_time <= 0:
                    if idx + 1 < len(epochs):
                        next_wasted = float(epochs[idx + 1][1].get('wasted_time', 0) or 0)
                        if next_wasted > 0:
                            wasted_time = next_wasted
                        else:
                            wasted_time = 0
                    else:
                        wasted_time = 0

                if wasted_time > 0:
                    per_event_time = wasted_time / rescale_count
                    per_type[job_type].extend([per_event_time] * rescale_count)

            previous_final_alloc = current_final_alloc
            previous_alloc_len = current_alloc_len

    return per_type


def main():
    # Defaults (can be overridden by CLI args)
    default_log1 = "./experiment_results/1024-fw-48/monitor_log.txt"
    default_log2 = "./experiment_results/1026-Pollux-0.5/monitor_log.txt"

    if len(sys.argv) >= 3:
        log1 = sys.argv[1]
        log2 = sys.argv[2]
    else:
        log1 = default_log1
        log2 = default_log2

    # Parse logs into jobs
    jobs1, *_ = process_log_file(log1)
    jobs2, *_ = process_log_file(log2)

    # Compute per-event rescaling times grouped by job type
    times_by_type1 = compute_rescaling_times_by_type(jobs1)
    times_by_type2 = compute_rescaling_times_by_type(jobs2)

    types_order = ['cifar10', 'bert', 'deepspeech2']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, t in zip(axes, types_order):
        data1 = times_by_type1.get(t, [])
        data2 = times_by_type2.get(t, [])
        combined = data1 + data2
        if len(combined) > 0:
            bin_edges = np.histogram_bin_edges(combined, bins='auto')
        else:
            bin_edges = [0, 1]

        ax.hist(data1, bins=bin_edges, color='blue', alpha=0.6, label=os.path.basename(log1), edgecolor='black')
        ax.hist(data2, bins=bin_edges, color='orange', alpha=0.6, label=os.path.basename(log2), edgecolor='black')
        ax.set_title(f"{t} (blue={len(data1)}, orange={len(data2)})")
        ax.set_xlabel('Rescaling time (s)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    axes[0].set_ylabel('Count')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('Rescaling Time Histograms by Job Type (Overlayed per Log)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()
