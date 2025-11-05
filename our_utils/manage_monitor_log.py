import json
import sys
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

NUM_GPU_PER_NODE = 4


def process_log_file(log_file_path):
    """Process log file and extract simplified job metrics."""
    jobs = {}
    total_gpu_hours = 0
    effective_gpu_hours = 0
    # New waste decomposition metrics
    fragmentation_waste_hours = 0
    ready_unused_waste_hours = 0
    wasted_capacity_hours = 0
    last_job_arrival_time = None
    completed_jobs_status = {}  # Track completion_time and pod_status for completed jobs
    decreased_progress_issues = {}  # Track any progress decreases per job
    job_drop_sums = {}  # Sum of progress drops per job
    job_max_progress = {}  # Max progress observed per job
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        
        # Calculate GPU hours for this time step
        if i > 0:
            prev_log = json.loads(lines[i-1].strip())
            time_diff = timestamp - prev_log['timestamp']
            
            # Extract cluster node information
            cluster_nodes = prev_log.get('cluster_nodes', {})
            ready_nodes = cluster_nodes.get('ready', 0)

            # Calculate GPU metrics (use READY GPUs for total/average usage)
            ready_gpus = ready_nodes * NUM_GPU_PER_NODE
            total_gpu_hours += ready_gpus * time_diff / 3600  # Convert to hours using ready capacity

            # Count effective GPUs (only those actually allocated)
            prev_effective_gpus = sum(len(job.get('allocation', [])) for job in prev_log['submitted_jobs'] if job.get('allocation'))
            effective_gpu_hours += prev_effective_gpus * time_diff / 3600

            # New waste decomposition:
            # - Fragmentation waste: 4 * (used_nodes) - used_gpus
            # - Ready-unused nodes waste: 4 * (ready_nodes - used_nodes)
            all_alloc_items = []
            for job in prev_log['submitted_jobs']:
                allocation = job.get('allocation', [])
                if allocation:
                    all_alloc_items.extend(allocation)

            used_gpus = len(all_alloc_items)
            used_nodes = len(set(all_alloc_items))

            fragmentation_waste_gpus = max(0, NUM_GPU_PER_NODE * used_nodes - used_gpus)
            ready_unused_waste_gpus = max(0, NUM_GPU_PER_NODE * (ready_nodes - used_nodes))

            fragmentation_waste_hours += fragmentation_waste_gpus * time_diff / 3600
            ready_unused_waste_hours += ready_unused_waste_gpus * time_diff / 3600

            # Total wasted capacity is the sum of the two components
            wasted_capacity_gpus = fragmentation_waste_gpus + ready_unused_waste_gpus
            wasted_capacity_hours += wasted_capacity_gpus * time_diff / 3600
            
        
        # Build quick lookup for current step jobs by name
        current_jobs_by_name = {j['name']: j for j in log_entry['submitted_jobs']}

        # Attribute queueing/wasted decomposition per time step using previous state
        if i > 0:
            prev_jobs = prev_log.get('submitted_jobs', [])
            for pjob in prev_jobs:
                pname = pjob['name']
                pepoch = pjob['epoch']
                palloc = pjob.get('allocation', []) or []
                pprogress = pjob.get('progress', None)
                ppod_status = pjob.get('pod_status', '')

                # Ensure job/epoch exist in our structure starting from prev timestamp
                if pname not in jobs:
                    jobs[pname] = {'first_seen': prev_log['timestamp'], 'epochs': {}}
                if pepoch not in jobs[pname]['epochs']:
                    jobs[pname]['epochs'][pepoch] = {
                        'first_seen': prev_log['timestamp'],
                        'last_seen': prev_log['timestamp'],
                        'gpu_allocations': [],
                        'progress_history': [],
                        'wasted_time': 0,
                        'queueing_time': 0,
                        'container_creation_time': 0,
                        'rescaling_time': 0,
                        'last_progress': None,
                        'stuck_start_time': None,
                        'queueing_start_time': None,
                        'allocation_pairs': set()
                    }

                einfo = jobs[pname]['epochs'][pepoch]
                # Time step length
                dt = time_diff
                has_alloc = len(palloc) > 0

                # Determine growth across the interval using current job state
                cjob = current_jobs_by_name.get(pname)
                cprogress = cjob.get('progress', None) if cjob is not None else None
                grew = False
                if pprogress is not None and cprogress is not None:
                    try:
                        grew = float(cprogress) > float(pprogress)
                    except Exception:
                        grew = False

                if not has_alloc:
                    einfo['queueing_time'] += dt
                else:
                    # With allocation
                    if not grew:
                        # Wasted this interval
                        einfo['wasted_time'] += dt
                        if ppod_status == 'pod status normal':
                            einfo['rescaling_time'] += dt
                        else:
                            einfo['container_creation_time'] += dt

        for job in log_entry['submitted_jobs']:
            job_name = job['name']
            epoch = job['epoch']
            allocation = job.get('allocation', [])
            progress = job.get('progress', 0)

            # Track per-job max progress
            if isinstance(progress, (int, float)):
                prev_max = job_max_progress.get(job_name)
                if prev_max is None or progress > prev_max:
                    job_max_progress[job_name] = progress

            # Track completed jobs and their pod status
            completion_time = job.get('completion_time', None)
            pod_status = job.get('pod_status', '')
            if completion_time is not None:
                # Store or update the completion info
                completed_jobs_status[job_name] = {
                    'completion_time': completion_time,
                    'pod_status': pod_status,
                    'timestamp': timestamp
                }
            
            # Initialize job if first time seeing it
            if job_name not in jobs:
                jobs[job_name] = {
                    'first_seen': timestamp,
                    'epochs': {}
                }
                # Track the latest job arrival time
                last_job_arrival_time = timestamp
            
            # Initialize epoch if first time seeing it
            if epoch not in jobs[job_name]['epochs']:
                jobs[job_name]['epochs'][epoch] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'gpu_allocations': [],
                    'progress_history': [],
                    'wasted_time': 0,
                    'queueing_time': 0,
                    'container_creation_time': 0,
                    'rescaling_time': 0,
                    'last_progress': None,
                    'stuck_start_time': None,
                    'queueing_start_time': None,
                    'allocation_pairs': set()
                }
            
            epoch_info = jobs[job_name]['epochs'][epoch]
            epoch_info['last_seen'] = timestamp
            
            # Track allocation pairs (num_nodes, num_replicas) observed in this epoch
            if allocation:
                num_replicas = len(allocation)
                num_nodes = len(set(allocation))
                epoch_info['allocation_pairs'].add((num_nodes, num_replicas))

            # Track GPU allocations
            gpu_count = len(allocation)
            if gpu_count not in epoch_info['gpu_allocations']:
                epoch_info['gpu_allocations'].append(gpu_count)
            has_allocation = gpu_count > 0
            
            # Note: time attribution handled per-step above using prev/current states.

            # Detect decreasing progress
            if (
                progress is not None
                and epoch_info['last_progress'] is not None
                and isinstance(progress, (int, float))
                and isinstance(epoch_info['last_progress'], (int, float))
                and progress < epoch_info['last_progress']
            ):
                if job_name not in decreased_progress_issues:
                    decreased_progress_issues[job_name] = []
                decreased_progress_issues[job_name].append({
                    'epoch': epoch,
                    'previous': epoch_info['last_progress'],
                    'current': progress,
                    'timestamp': timestamp
                })
                # Accumulate drop amount
                drop_amount = epoch_info['last_progress'] - progress
                if drop_amount > 0:
                    job_drop_sums[job_name] = job_drop_sums.get(job_name, 0) + drop_amount
            
            # No start/stop accumulation here; per-step attribution already applied
            
            epoch_info['last_progress'] = progress
            epoch_info['progress_history'].append((timestamp, progress))
    
    # Calculate final epoch durations
    for job_name, job_info in jobs.items():
        for epoch_num, epoch_info in job_info['epochs'].items():
            epoch_info['duration'] = epoch_info['last_seen'] - epoch_info['first_seen']
    
    return (
        jobs,
        total_gpu_hours,
        effective_gpu_hours,
        fragmentation_waste_hours,
        ready_unused_waste_hours,
        wasted_capacity_hours,
        last_job_arrival_time,
        completed_jobs_status,
        decreased_progress_issues,
        job_drop_sums,
        job_max_progress,
    )

def is_pod_status_failing(pod_status):
    """
    Determine if a pod status string indicates failure.
    Returns True if the pod status shows signs of failure.
    """
    if not pod_status:
        return False

    # "pod status normal" means everything is OK
    if pod_status == "pod status normal":
        return False

    # Check for failure indicators in the pod status string
    failure_indicators = [
        "Failed",
        "Error",
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "terminated",
        "not ready"
    ]

    pod_status_lower = pod_status.lower()
    for indicator in failure_indicators:
        if indicator.lower() in pod_status_lower:
            return True

    return False

def calculate_metrics(total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours, wasted_capacity_hours, last_job_arrival_time, first_job_time):
    """Calculate simplified metrics based on GPU hours and job arrival time."""
    experiment_duration_hours = (last_job_arrival_time - first_job_time) / 3600
    # average_gpu_usage now reflects READY GPUs average
    average_gpu_usage = total_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    effective_average_gpu_usage = effective_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    fragmentation_waste_average = fragmentation_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    ready_unused_average = ready_unused_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    wasted_capacity_average = wasted_capacity_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0

    
    return {
        'total_gpu_hours': total_gpu_hours,
        'effective_gpu_hours': effective_gpu_hours,
        'fragmentation_waste_hours': fragmentation_waste_hours,
        'ready_unused_waste_hours': ready_unused_waste_hours,
        'wasted_capacity_hours': wasted_capacity_hours,
        'experiment_duration_hours': experiment_duration_hours,
        'average_gpu_usage': average_gpu_usage,
        'effective_average_gpu_usage': effective_average_gpu_usage,
        'fragmentation_waste_average': fragmentation_waste_average,
        'ready_unused_average': ready_unused_average,
        'wasted_capacity_average': wasted_capacity_average,
        'last_job_arrival_time': last_job_arrival_time
    }

def print_summary(metrics):
    """Print summary of simplified metrics."""
    print("\n" + "="*60)
    print("LOG PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total GPU Hours: {metrics['total_gpu_hours']:.2f}")
    print(f"Effective GPU Hours: {metrics['effective_gpu_hours']:.2f}")
    print(f"Experiment Duration: {metrics['experiment_duration_hours']:.2f} hours (From first job arrival to last job arrival)")
    print(f"Average GPU Usage: {metrics['average_gpu_usage']:.2f} GPUs")
    print(f"Effective Average GPU Usage: {metrics['effective_average_gpu_usage']:.2f} GPUs")
    
    print(f"\nDecomposition:")
    print(f"Fragmentation Waste Average: {metrics['fragmentation_waste_average']:.2f} GPUs")
    print(f"Ready-Unused Nodes Average: {metrics['ready_unused_average']:.2f} GPUs")
    print(f"Wasted Capacity Average: {metrics['wasted_capacity_average']:.2f} GPUs")
    
    print(f"\nLast Job Arrival Time: {datetime.fromtimestamp(metrics['last_job_arrival_time'])}")

def print_failed_completed_jobs(completed_jobs_status):
    """Print jobs that completed with failing pod status."""
    failed_jobs = []

    for job_name, job_info in completed_jobs_status.items():
        pod_status = job_info['pod_status']
        if is_pod_status_failing(pod_status):
            failed_jobs.append({
                'job_name': job_name,
                'completion_time': job_info['completion_time'],
                'pod_status': pod_status,
                'timestamp': job_info['timestamp']
            })

    if not failed_jobs:
        print(f"\n" + "="*80)
        print("COMPLETED JOBS WITH FAILING POD STATUS")
        print("="*80)
        print("No completed jobs with failing pod status found.")
        return

    print(f"\n" + "="*80)
    print("COMPLETED JOBS WITH FAILING POD STATUS")
    print("="*80)
    print(f"Found {len(failed_jobs)} job(s) that completed with failing pod status:\n")

    for job in failed_jobs:
        print(f"Job: {job['job_name']}")
        print(f"  Completion Time: {job['completion_time']}")
        print(f"  Timestamp: {datetime.fromtimestamp(job['timestamp'])}")
        print(f"  Pod Status: {job['pod_status']}")
        print()


def print_mean_rescaling_time(jobs):
    """Calculate and print mean rescaling+container-creation time per job type.

    Uses the same rescaling detection logic, and attributes time as the sum of
    epoch-level rescaling_time and container_creation_time (falling back to the
    next epoch if needed as before).
    """
    rescale_stats = {}

    for job_name, job_info in jobs.items():
        job_type = job_name.split('-')[0] if '-' in job_name else job_name
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
                rescale_count = 0 # ignore the last epoch's rescaling to 0.
            # if idx == 0 and 0 in gpu_allocations and 1 in gpu_allocations:
            #     rescale_count -= 1 # ignore the first epoch's rescaling to 1.
            if rescale_count <= 0:
                previous_final_alloc = current_final_alloc
                previous_alloc_len = current_alloc_len
                continue
            # Sum both components for mean: rescaling + container creation
            wasted_time = (
                float(epoch_info.get('rescaling_time', 0) or 0)
                + float(epoch_info.get('container_creation_time', 0) or 0)
            )
            if job_name == 'cifar10-46':
                print(f"Rescaling detected for job {job_name}, epoch {epoch_num}, rescale_count: {rescale_count}")
            if wasted_time <= 0:
                if idx + 1 >= len(epochs):
                    raise AssertionError(
                        f"Rescaling detected but no subsequent wasted time for job {job_name}, epoch {epoch_num}"
                    )
                next_epoch = epochs[idx + 1][1]
                next_wasted = (
                    float(next_epoch.get('rescaling_time', 0) or 0)
                    + float(next_epoch.get('container_creation_time', 0) or 0)
                )
                assert next_wasted > 0, (
                    f"Expected wasted time after rescaling for job {job_name}, epoch {epoch_num}, got 0"
                )
                wasted_time = next_wasted

            stats = rescale_stats.setdefault(job_type, {'num_rescaling': 0, 'total_time': 0.0})
            stats['num_rescaling'] += rescale_count
            stats['total_time'] += wasted_time

            previous_final_alloc = current_final_alloc
            previous_alloc_len = current_alloc_len
    if not rescale_stats:
        print("\nNo rescaling events detected across jobs.")
        return

    print(f"\n" + "=" * 80)
    print("MEAN RESCALE+CREATE TIME BY JOB TYPE")
    print("=" * 80)

    overall_events = 0
    overall_time = 0.0

    for job_type in sorted(rescale_stats.keys()):
        stats = rescale_stats[job_type]
        events = stats['num_rescaling']
        total_time = stats['total_time']
        mean_time = (total_time / events) if events > 0 else 0.0
        print(
            f"{job_type:<15} mean_rescale+create: {mean_time:.1f}s | "
            f"events: {events}, total_time: {total_time:.1f}s"
        )
        overall_events += events
        overall_time += total_time

    if overall_events > 0:
        overall_mean = overall_time / overall_events
        print("-" * 80)
        print(
            f"Overall mean rescaling time: {overall_mean:.1f}s "
            f"across {overall_events} rescaling events"
        )


def plot_rescaling_time_histograms(jobs):
    """Plot histograms of per-rescaling times for CIFAR10, BERT, and DeepSpeech2.

    This mirrors the rescaling detection logic in `print_mean_rescaling_time`,
    but collects a per-event time (wasted_time divided by the number of rescalings
    detected in that epoch) and plots their distributions.
    """
    # Collect per-event rescaling times per job type
    rescale_times_by_type = {
        'cifar10': [],
        'bert': [],
        'deepspeech2': [],
    }

    for job_name, job_info in jobs.items():
        job_type = job_name.split('-')[0] if '-' in job_name else job_name
        # Only track the three requested types
        if job_type not in rescale_times_by_type:
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

            # Handle the case where alloc length is 1 but final alloc changes between epochs
            if (
                previous_final_alloc is not None
                and previous_alloc_len == 1
                and current_alloc_len == 1
                and current_final_alloc != previous_final_alloc
            ):
                rescale_count += 1

            # Ignore special/residual cases consistent with mean function
            if idx == len(epochs) - 1 and rescale_count > 0:
                rescale_count = 0  # ignore the last epoch's rescaling to 0
            # if idx == 0 and 0 in gpu_allocations and 1 in gpu_allocations:
            #     rescale_count -= 1  # ignore the first epoch's rescaling to 1

            if rescale_count > 0:
                wasted_time = float(epoch_info.get('rescaling_time', 0) or 0)
                if wasted_time <= 0:
                    if idx + 1 < len(epochs):
                        next_wasted = float(epochs[idx + 1][1].get('rescaling_time', 0) or 0)
                        if next_wasted > 0:
                            wasted_time = next_wasted
                        else:
                            # If we cannot attribute wasted time, skip adding to histogram
                            wasted_time = 0
                    else:
                        wasted_time = 0

                if wasted_time > 0:
                    per_event_time = wasted_time / rescale_count
                    # Add one entry per rescaling event to build the distribution
                    rescale_times_by_type[job_type].extend([per_event_time] * rescale_count)
                    # Warn if CIFAR10 per-event rescaling time is unusually large
                    if job_type == 'cifar10' and per_event_time > 200:
                        print(
                            f"WARNING: CIFAR10 per-event rescale time {per_event_time:.1f}s > 200s "
                            f"for job {job_name}, epoch {epoch_num} (rescale_count={rescale_count}, total_rescaling={wasted_time:.1f}s)"
                        )

            previous_final_alloc = current_final_alloc
            previous_alloc_len = current_alloc_len

    # Create a single figure with 3 subplots, one per job type
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    types_order = ['cifar10', 'bert', 'deepspeech2']

    for ax, jtype in zip(axes, types_order):
        data = rescale_times_by_type[jtype]
        if len(data) > 0:
            ax.hist(data, bins='auto', color='#1f77b4', alpha=0.8, edgecolor='black')
        else:
            # Draw an empty histogram frame if no data
            ax.hist([], bins=1, color='#1f77b4', alpha=0.8, edgecolor='black')
        ax.set_title(f"{jtype} (n={len(data)})")
        ax.set_xlabel("Rescaling time (s)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    axes[0].set_ylabel("Count")
    fig.suptitle("Rescaling Time Distributions by Job Type")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_rescaling_time_breakdown_histograms(jobs):
    """Plot histograms of per-event rescaling vs container-creation times by job type.

    Figure A: Only rescaling time
    Figure B: Overlay rescaling and container-creation with different colors
    """
    types = ['cifar10', 'bert', 'deepspeech2']
    per_event_rescale = {t: [] for t in types}
    per_event_create = {t: [] for t in types}

    for job_name, job_info in jobs.items():
        job_type = job_name.split('-')[0] if '-' in job_name else job_name
        if job_type not in types:
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
            # if idx == 0 and 0 in gpu_allocations and 1 in gpu_allocations:
            #     rescale_count -= 1
            if rescale_count > 0:
                rescale_time = float(epoch_info.get('rescaling_time', 0) or 0)
                create_time = float(epoch_info.get('container_creation_time', 0) or 0)
                if rescale_time <= 0 and idx + 1 < len(epochs):
                    nxt = float(epochs[idx + 1][1].get('rescaling_time', 0) or 0)
                    if nxt > 0:
                        rescale_time = nxt
                if create_time <= 0 and idx + 1 < len(epochs):
                    nxtc = float(epochs[idx + 1][1].get('container_creation_time', 0) or 0)
                    if nxtc > 0:
                        create_time = nxtc
                if rescale_time > 0:
                    per_event_rescale[job_type].extend([rescale_time / rescale_count] * rescale_count)
                if create_time > 0:
                    per_event_create[job_type].extend([create_time / rescale_count] * rescale_count)
            previous_final_alloc = current_final_alloc
            previous_alloc_len = current_alloc_len

    # Figure A: rescaling only
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, jtype in zip(axes1, types):
        data = per_event_rescale[jtype]
        if len(data) > 0:
            ax.hist(data, bins='auto', color='#1f77b4', alpha=0.8, edgecolor='black')
        else:
            ax.hist([], bins=1)
        ax.set_title(f"{jtype} rescale (n={len(data)})")
        ax.set_xlabel("Per-event rescale (s)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    axes1[0].set_ylabel("Count")
    fig1.suptitle("Per-Event Rescaling Time by Job Type")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Figure B: overlay rescaling and container creation
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, jtype in zip(axes2, types):
        rdata = per_event_rescale[jtype]
        cdata = per_event_create[jtype]
        if len(rdata) > 0:
            ax.hist(rdata, bins='auto', color='#1f77b4', alpha=0.6, edgecolor='black', label='Rescale')
        if len(cdata) > 0:
            ax.hist(cdata, bins='auto', color='#ff7f0e', alpha=0.6, edgecolor='black', label='Container Creation')
        ax.set_title(f"{jtype} (r={len(rdata)}, c={len(cdata)})")
        ax.set_xlabel("Per-event time (s)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    axes2[0].set_ylabel("Count")
    handles, labels = axes2[0].get_legend_handles_labels()
    if handles:
        fig2.legend(handles, labels, loc='upper right')
    fig2.suptitle("Per-Event Rescaling vs Container Creation Time by Job Type")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def print_decreasing_progress_warnings(decreased_progress_issues, job_drop_sums, job_max_progress):
    """Print warning for jobs where progress decreased, including drop stats."""
    if not decreased_progress_issues:
        return

    print(f"\n" + "="*80)
    print("WARNING: JOBS WITH DECREASING PROGRESS DETECTED")
    print("="*80)
    for job_name in sorted(decreased_progress_issues.keys()):
        occurrences = len(decreased_progress_issues[job_name])
        total_progress = job_max_progress.get(job_name, 0) or 0
        total_drop = job_drop_sums.get(job_name, 0) or 0
        fraction = (total_drop / total_progress) if total_progress > 0 else 0.0
        print(f"{job_name} (occurrences: {occurrences}) | sum_drops: {total_drop:.6f}, total_progress: {total_progress:.6f}, fraction: {fraction:.6f}")

def print_all_jobs_summary(jobs):
    """Print response time and queueing/wasted time for all jobs."""
    print(f"\n" + "="*80)
    print("ALL JOBS SUMMARY")
    print("="*80)
    print(f"{'Job Name':<20} {'Response Time(s)':<18} {'Queueing(s)':<12} {'Wasted(s)':<12} {'Idle(s)':<12}")
    print("-" * 80)
    for job_name, job_info in jobs.items():
        # Calculate total response time (from first seen to completion of last epoch)
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            response_time = last_epoch_end - job_info['first_seen']
        else:
            response_time = 0
        # Calculate queueing and wasted times across all epochs
        total_queueing = sum(epoch_info.get('queueing_time', 0) for epoch_info in job_info['epochs'].values())
        total_wasted = sum(epoch_info.get('wasted_time', 0) for epoch_info in job_info['epochs'].values())
        total_idle = total_queueing + total_wasted
        print(f"{job_name:<20} {response_time:<18.1f} {total_queueing:<12.1f} {total_wasted:<12.1f} {total_idle:<12.1f}")

def compute_mean_job_response_time(jobs):
    """Compute mean total response time per job (from first_seen to last epoch end)."""
    response_times = []
    for job_name, job_info in jobs.items():
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            response_times.append(last_epoch_end - job_info['first_seen'])
    return (sum(response_times) / len(response_times)) if response_times else 0.0

def get_theoretical_duration(application, epoch, num_gpus):
    """Calculate theoretical duration using goodput function."""
    goodput_functions = {
        'bert': {
            0: {1: 14.777758035005506, 2: 21.352068051613056, 4: 24.453647524136986, 8: 16.740192512369788, 12: 13.237639734844972, 16: 16.49061741344335, 24: 24.060092988249075, 32: 29.496607606234743},
            1: {1: 16.05313546474798, 2: 31.195719148632165, 4: 56.10326354436123, 8: 103.16557330455697, 12: 127.39506263744747, 16: 146.28101130098835, 24: 162.68530046653478, 32: 187.15654366589814},
        },
        'cifar10': {
            0: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            1: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            2: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            3: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            4: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            5: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            6: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            7: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            8: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            9: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            10: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            11: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            12: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            13: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            14: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            15: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            16: {1: 979.9834454147685, 2: 1558.5698686812405, 4: 2628.822159808343, 8: 2218.5200670484014, 12: 2491.1734275087815, 16: 4065.5595989357344, 24: 5792.5768632633035, 32: 6446.95210074248},
            17: {1: 999.3825197696726, 2: 1007.5445976364409, 4: 2461.1090512079536, 8: 2218.5200670484014, 12: 2778.9277402621015, 16: 4174.077754904613, 24: 5801.170750188324, 32: 7679.676201152359},
            18: {1: 988.294581617859, 2: 1603.6271738718344, 4: 2633.5795177066125, 8: 2218.5200670484014, 12: 3536.156467747355, 16: 4152.119298352597, 24: 5821.531013867037, 32: 6421.6788493651},
            19: {1: 988.294581617859, 2: 1603.6271738718344, 4: 2633.5795177066125, 8: 2218.5200670484014, 12: 3536.156467747355, 16: 4152.119298352597, 24: 5821.531013867037, 32: 6421.6788493651},
            20: {1: 988.294581617859, 2: 1603.6271738718344, 4: 2633.5795177066125, 8: 2218.5200670484014, 12: 3536.156467747355, 16: 4152.119298352597, 24: 5821.531013867037, 32: 6421.6788493651},
            21: {1: 683.3763458833629, 2: 1551.1315686862652, 4: 2632.2272951926348, 8: 2218.5200670484014, 12: 3262.2282381280766, 16: 4146.9063532684295, 24: 5810.479444515204, 32: 7687.7930873237},
            22: {1: 992.0512134934465, 2: 1375.589580582176, 4: 2643.450858791405, 8: 2218.5200670484014, 12: 3530.904642111086, 16: 4160.028609083467, 24: 5832.687158115831, 32: 6446.785824677657},
            23: {1: 1008.1093950826872, 2: 931.0306935685705, 4: 2589.498087923301, 8: 2218.5200670484014, 12: 3507.522295571011, 16: 4256.516824251578, 24: 5850.7328174311415, 32: 7670.266944346536},
            24: {1: 865.342645610416, 2: 1560.2959371491984, 4: 2807.688057485502, 8: 2218.5200670484014, 12: 3536.26218716381, 16: 4604.448875011798, 24: 5477.0207187701735, 32: 6427.596018857206},
            25: {1: 790.327892697667, 2: 1620.820692730599, 4: 2988.9663683418403, 8: 2218.5200670484014, 12: 3458.8092438545355, 16: 4150.67821362216, 24: 5820.424902408096, 32: 7577.616339235232},
            26: {1: 987.1110480112912, 2: 1564.7893738729408, 4: 2566.7343250572208, 8: 2218.5200670484014, 12: 3526.712668348805, 16: 4083.846250223992, 24: 5845.70992088348, 32: 7737.802480212106},
            27: {1: 887.2482459036619, 2: 1605.798747647392, 4: 2802.015653621252, 8: 2218.5200670484014, 12: 3014.227488000845, 16: 4153.856898454097, 24: 5816.978393336095, 32: 7637.358609397888},
            28: {1: 994.0323467292243, 2: 1085.2416319251076, 4: 2613.545361534364, 8: 2218.5200670484014, 12: 2487.0577121953684, 16: 2965.950750966162, 24: 5849.782014482084, 32: 7366.976088852009},
            29: {1: 994.0323467292243, 2: 1085.2416319251076, 4: 2613.545361534364, 8: 2218.5200670484014, 12: 2487.0577121953684, 16: 2965.950750966162, 24: 5849.782014482084, 32: 7366.976088852009},
            30: {1: 993.7170267529701, 2: 1085.2416319251076, 4: 2611.807425680939, 8: 2218.5200670484014, 12: 3522.710750003158, 16: 4151.373073524911, 24: 5827.040035143091, 32: 7648.9594357701935},
            31: {1: 993.7170267529701, 2: 1085.2416319251076, 4: 2611.807425680939, 8: 2218.5200670484014, 12: 3522.710750003158, 16: 4151.373073524911, 24: 5827.040035143091, 32: 7648.9594357701935},
            32: {1: 993.7170267529701, 2: 1085.2416319251076, 4: 2611.807425680939, 8: 2218.5200670484014, 12: 3522.710750003158, 16: 4151.373073524911, 24: 5827.040035143091, 32: 7648.9594357701935},
            33: {1: 993.7170267529701, 2: 1085.2416319251076, 4: 2611.807425680939, 8: 2218.5200670484014, 12: 3522.710750003158, 16: 4151.373073524911, 24: 5827.040035143091, 32: 7648.9594357701935},
            34: {1: 903.8607763259118, 2: 1152.0959515068791, 4: 2708.8757015973624, 8: 1914.8650959355036, 12: 3450.147031088096, 16: 4622.329390384952, 24: 5786.245482806633, 32: 6398.087876142241},
            35: {1: 779.0961996376354, 2: 1635.5030768430383, 4: 2803.0045121626677, 8: 3031.73693473914, 12: 3502.1507608877223, 16: 4088.1930683423625, 24: 5726.995062497414, 32: 7757.532713737458},
            36: {1: 998.8858823825058, 2: 1404.6452666085038, 4: 2975.1088542085977, 8: 3049.8254606833207, 12: 3835.7624704412087, 16: 4604.292791159521, 24: 6786.849159375189, 32: 7732.203606426253},
            37: {1: 992.3214178979362, 2: 2050.539063976132, 4: 2769.232191278409, 8: 3241.325465381266, 12: 3316.0670001278263, 16: 4621.33911852792, 24: 5817.23333517426, 32: 7714.188583195705},
            38: {1: 667.5171625814887, 2: 2143.2461739211312, 4: 2619.873086655285, 8: 3224.4341316840023, 12: 3812.3373091172266, 16: 4056.7835802278846, 24: 5802.238411608903, 32: 7397.578912988872},
            39: {1: 1018.6821014041095, 2: 1400.1145393612871, 4: 2591.4196024373296, 8: 3289.344635310897, 12: 2634.718255459998, 16: 4608.932509921618, 24: 6774.7649541773835, 32: 7754.312667905492},
            40: {1: 986.7528456816384, 2: 1656.8164467650447, 4: 2771.2658924114507, 8: 3051.7395932272125, 12: 2222.0734825326163, 16: 4140.377406052228, 24: 5820.759632069167, 32: 7698.569509758542},
            41: {1: 724.2882392777088, 2: 2100.9415944103334, 4: 2752.2258046392813, 8: 3283.96741281444, 12: 2814.6234388662333, 16: 4611.644144621586, 24: 5821.836617736431, 32: 7736.04249062889},
            42: {1: 954.2997061566049, 2: 2011.085022176287, 4: 2793.93793782677, 8: 3204.8105900906216, 12: 3357.0005797089384, 16: 4601.24803920848, 24: 5823.3474929182175, 32: 7728.658471418272},
            43: {1: 1000.1820752672552, 2: 2133.4053409721005, 4: 2630.6887783051066, 8: 3251.416294594857, 12: 3805.643107432795, 16: 3201.4231424012387, 24: 6789.830290180587, 32: 7537.613812594501},
            44: {1: 987.1735346707569, 2: 2137.6446545148456, 4: 2815.9194341800558, 8: 2122.14391591892, 12: 3486.859010836329, 16: 2971.188474539752, 24: 5832.844516869939, 32: 7746.469950880154},
            45: {1: 703.5341254904547, 2: 1767.2111956723065, 4: 2800.9205372195593, 8: 1915.5297097036514, 12: 3539.2880342813173, 16: 2764.8271501166096, 24: 5840.6471669845605, 32: 7668.465755066625},
            46: {1: 1014.8542602632898, 2: 1378.2138544234804, 4: 2755.095683962115, 8: 2968.6242416224404, 12: 3756.542730818655, 16: 4618.930095896972, 24: 6651.324375533642, 32: 7657.930819316203},
            47: {1: 996.243441599554, 2: 2115.7163104517326, 4: 2853.001475358758, 8: 3273.7617955961027, 12: 3550.304232417728, 16: 4150.812862042859, 24: 5807.455246235978, 32: 7727.811494273543},
            48: {1: 662.5992278640284, 2: 2029.3392937588599, 4: 2464.933340374594, 8: 3266.9621560225423, 12: 3527.0763799716274, 16: 4593.511173075819, 24: 6729.866345941339, 32: 7373.021997084194},
            49: {1: 994.9061980331727, 2: 2142.280225513354, 4: 2978.2871619379594, 8: 3280.7869250483695, 12: 3531.199056421541, 16: 4447.41643131918, 24: 5816.908046231109, 32: 7753.329919207063},
            50: {1: 978.6465193648303, 2: 2102.2776682969584, 4: 2817.4604432773335, 8: 3253.680162187536, 12: 3778.957360503287, 16: 4594.996116118376, 24: 6721.569285814994, 32: 9667.983467056394},
            51: {1: 786.7996311441526, 2: 2046.1738897142645, 4: 2342.299554177773, 8: 3275.088305800926, 12: 3510.4446068472344, 16: 4612.864432219541, 24: 5823.176578876541, 32: 7727.606470555679},
            52: {1: 879.5793276578617, 2: 2154.519640382593, 4: 3870.3347644905894, 8: 3043.3651437960543, 12: 2603.853120350854, 16: 4606.846657493914, 24: 6812.4122705618665, 32: 7560.708049187404},
            53: {1: 1006.6997468772872, 2: 1494.1105462210455, 4: 3799.610355280396, 8: 3262.477976563976, 12: 2352.692838172671, 16: 4601.116298662548, 24: 5793.072738538102, 32: 7747.54569852491},
            54: {1: 1003.7084026344072, 2: 1546.2640790614691, 4: 4010.225736015743, 8: 3210.5715770312854, 12: 2845.7554264737983, 16: 4143.0262611291555, 24: 6776.154975232315, 32: 7740.296265402424},
            55: {1: 758.3001863196905, 2: 2157.2787233752774, 4: 3879.9253650450055, 8: 2097.3678611002642, 12: 3845.102129474805, 16: 4589.668409439745, 24: 5835.73769912455, 32: 7768.025708412414},
            56: {1: 1010.9809886535709, 2: 2119.300898461861, 4: 3762.449720127758, 8: 2125.2809003225107, 12: 3498.2817375288164, 16: 4618.498695462281, 24: 6782.923215079574, 32: 7731.637751134665},
            57: {1: 990.1943865194121, 2: 2136.4145784761736, 4: 3908.9422997311804, 8: 3271.7357109655186, 12: 3515.188789167419, 16: 4552.249766293539, 24: 5835.867289919204, 32: 9150.197214562462},
            58: {1: 903.2450929566081, 2: 2038.2997434640174, 4: 3569.5229699601055, 8: 3041.8164033805433, 12: 3779.787482478296, 16: 4582.233706084975, 24: 6796.450779243944, 32: 7734.971646578651},
            59: {1: 909.6232487361892, 2: 2209.07779846012, 4: 3900.703362835914, 8: 3231.8690416444456, 12: 3484.904329110651, 16: 3364.2270083220164, 24: 5834.321994230922, 32: 7763.22409352419},
            60: {1: 979.1106956803744, 2: 1864.455731205119, 4: 3830.4026484227784, 8: 3379.4387289641963, 12: 3746.146768000958, 16: 2401.319808031545, 24: 6829.353501511044, 32: 7752.374357610591},
            61: {1: 1006.2298723629704, 2: 1438.9243559597473, 4: 3889.9120782215937, 8: 3274.5115768859473, 12: 3530.0727352685653, 16: 2551.882291723059, 24: 5786.894365513162, 32: 7753.375496266024},
            62: {1: 788.0460818952836, 2: 2048.3160287591636, 4: 3918.931060366075, 8: 3275.4356275250907, 12: 3757.743961727903, 16: 4535.22238074046, 24: 6789.571559552238, 32: 9575.966568280779},
            63: {1: 1036.010011824429, 2: 2110.849386100381, 4: 3899.186722441014, 8: 3401.3295298942626, 12: 3517.1664527226126, 16: 4598.330718628847, 24: 6819.185540654586, 32: 7776.236239247175},
            64: {1: 1007.7679146752218, 2: 2044.7323619345307, 4: 3595.3018964657244, 8: 3267.941987933738, 12: 3765.3656676219566, 16: 4543.1008395418785, 24: 5719.9972703107505, 32: 7630.428481724647},
            65: {1: 885.6483294540567, 2: 2144.006956506946, 4: 3910.3885937850287, 8: 3250.2920904984976, 12: 2652.025344190109, 16: 4612.313752016614, 24: 6701.591667646108, 32: 9520.777329426208},
            66: {1: 961.1803498853551, 2: 2119.3749818913698, 4: 3915.2799312797756, 8: 3033.0502440929517, 12: 2488.2528451174276, 16: 4805.83056011711, 24: 6750.15719451999, 32: 7708.255196575298},
            67: {1: 1013.2703525881284, 2: 2148.582820343004, 4: 3804.5273569320657, 8: 2129.0240434771035, 12: 3025.9115987367004, 16: 4495.325564664783, 24: 5822.257663452756, 32: 7731.7876875947095},
            68: {1: 1005.7700012927309, 2: 1261.7276287523937, 4: 3917.2964551615632, 8: 2102.9314444300985, 12: 3474.0152307215862, 16: 4525.2385897754675, 24: 6771.5040373902, 32: 7736.002824450799},
            69: {1: 803.3568766492019, 2: 1766.4528885741381, 4: 3931.307397708122, 8: 3242.2048635546557, 12: 3857.529627984513, 16: 4599.16056161571, 24: 6793.30098850752, 32: 9722.041167287281},
            70: {1: 1000.397234339214, 2: 2043.7059821341004, 4: 3804.8267178536307, 8: 3268.5175764612472, 12: 3453.046572426239, 16: 4615.334065556453, 24: 6740.902723063029, 32: 7675.200282772386},
            71: {1: 1041.0998773969109, 2: 2134.337125735821, 4: 3879.04092272828, 8: 3496.5147010117234, 12: 3843.692445587698, 16: 4573.325284398999, 24: 5777.888814917191, 32: 7585.97100046822},
            72: {1: 940.0765312967615, 2: 2197.791073001733, 4: 3762.640960750605, 8: 3264.10645103024, 12: 3453.4827116809647, 16: 4595.819216512789, 24: 6792.823722125043, 32: 9697.099727696323},
            73: {1: 937.6611544190191, 2: 2145.7331362485065, 4: 3911.7652539440446, 8: 3211.0441378009823, 12: 3845.4575501516665, 16: 4538.538728480618, 24: 6782.550722412196, 32: 7639.491670024108},
            74: {1: 1011.337296568719, 2: 2015.186118305433, 4: 3588.2404925576784, 8: 3227.353515085776, 12: 3477.3081053045585, 16: 5094.7038500779845, 24: 6529.379064317817, 32: 7743.077250999927},
            75: {1: 1007.4382368147591, 2: 1533.2426151333677, 4: 3901.594590969782, 8: 3456.3506094717973, 12: 3845.744345891068, 16: 3115.100395216242, 24: 5839.353297146631, 32: 9644.434950425817},
            76: {1: 791.7772735277621, 2: 1538.7177752971593, 4: 3920.258194129079, 8: 3258.787206471877, 12: 3529.2746587247757, 16: 2757.4585359311436, 24: 6633.128726001485, 32: 7749.355883790257},
            77: {1: 1004.2497419978733, 2: 2139.8049241352305, 4: 3861.0372564563067, 8: 3218.112341177066, 12: 3839.9311184231137, 16: 2962.812475828233, 24: 6793.9574768610755, 32: 7654.5037375871525},
            78: {1: 1005.4269511764055, 2: 2155.0955143196334, 4: 3888.172184634338, 8: 2630.48617619779, 12: 2961.8103577720576, 16: 4597.727861029158, 24: 6763.090315411605, 32: 9686.64295358656},
            79: {1: 921.1702021342824, 2: 2143.188634880227, 4: 3730.6837258806186, 8: 1846.3049259454724, 12: 2481.265319992942, 16: 4576.323994744832, 24: 6773.119337653088, 32: 7749.802906161308},
            80: {1: 891.2200188882454, 2: 2156.572517007049, 4: 3889.173451981925, 8: 2606.019109943581, 12: 2492.5733650411958, 16: 4604.581918566285, 24: 5794.921938360095, 32: 7732.552853164191},
            81: {1: 1024.0124151505215, 2: 2156.8478989000587, 4: 2462.6796905149163, 8: 3274.020765622582, 12: 3845.3490344711195, 16: 4569.691349201092, 24: 6806.4920204777145, 32: 9669.672516494427},
            82: {1: 1003.0065297114365, 2: 1972.4329538263023, 4: 2681.058908016992, 8: 3464.2947212947283, 12: 3813.0381616909913, 16: 4607.50889984485, 24: 6739.313579481373, 32: 7671.350533544545},
            83: {1: 850.9642206882658, 2: 1265.8677195781927, 4: 3278.510460379251, 8: 3237.109197175457, 12: 3514.634080040903, 16: 4636.472616826275, 24: 6804.681034632407, 32: 7749.73818352208},
            84: {1: 1017.0286164514283, 2: 2049.1885493702416, 4: 3869.829478921771, 8: 3268.4549196157072, 12: 3831.550920407208, 16: 4426.629355932428, 24: 5807.001285007042, 32: 9663.807732458097},
            85: {1: 1022.6086281149896, 2: 2251.146445000739, 4: 3771.2777394733007, 8: 3264.80063337212, 12: 3522.2797269737407, 16: 5172.272404261752, 24: 6808.031232476961, 32: 6431.095469209168},
            86: {1: 944.6901356695637, 2: 2121.339341464228, 4: 3830.6233196230583, 8: 3257.74686385755, 12: 3799.969602446532, 16: 4587.485952252699, 24: 6818.730791601818, 32: 12811.16050377627},
            87: {1: 942.84357457516, 2: 2159.0831408110234, 4: 4063.261067553862, 8: 3532.623723283648, 12: 3769.2703007719, 16: 4601.128614317297, 24: 6791.137434045413, 32: 19105.49577992283},
            88: {1: 1005.7282503134784, 2: 2132.7710466795165, 4: 3900.5925774071547, 8: 3232.460291671944, 12: 3518.502011298086, 16: 4608.157257866578, 24: 6545.290753388574, 32: 19406.90252657336},
            89: {1: 1005.2931872719281, 2: 1928.602232208619, 4: 3875.2823119637073, 8: 3272.785928059235, 12: 3780.511650928566, 16: 4590.124077798299, 24: 6810.786581689886, 32: 19313.788557432716},
            90: {1: 822.1114176672322, 2: 1182.0669441160553, 4: 3886.629831919613, 8: 2119.4507198288115, 12: 3840.409145151542, 16: 4115.50330586174, 24: 5110.434902798142, 32: 19221.537413564343},
            91: {1: 1017.1654576847569, 2: 2110.5785084991726, 4: 3873.622876339055, 8: 2107.1487562006546, 12: 2785.7903480861746, 16: 2962.3420666650986, 24: 13536.123371305099, 32: 12941.023508315748},
            92: {1: 998.5964332558832, 2: 2164.722291008218, 4: 4132.432558544266, 8: 2886.016063774543, 12: 2465.3498561636693, 16: 3103.4504241822156, 24: 20593.69974696233, 32: 19368.37983285478},
            93: {1: 957.2216122728661, 2: 2223.8424591060293, 4: 3870.2739089788256, 8: 3557.4685368785294, 12: 2769.549007077712, 16: 3183.244639012221, 24: 13580.266020525114, 32: 19281.53935153261},
            94: {1: 916.2611026947951, 2: 2104.3840102650374, 4: 3910.209744334448, 8: 3185.1083935176976, 12: 3690.085350221168, 16: 4599.691560065885, 24: 13620.322672387696, 32: 19360.4337604506},
            95: {1: 1015.1703378832581, 2: 2101.323042185817, 4: 3875.7839369178255, 8: 3542.712334397804, 12: 3788.078561547742, 16: 3784.1738050253107, 24: 20276.537617840975, 32: 19426.972706850072},
            96: {1: 1001.451526712268, 2: 2145.35369965619, 4: 3818.0500572529454, 8: 3266.5658491944737, 12: 3609.5391018055666, 16: 10416.48671755789, 24: 20319.70451845694, 32: 19370.31727425356},
            97: {1: 866.3446483155919, 2: 1625.6658671507726, 4: 3806.244507367453, 8: 3268.8882688617705, 12: 3489.3703131415486, 16: 13864.83250717835, 24: 13701.60293868608, 32: 12972.059516221167},
            98: {1: 1009.7958846774957, 2: 1506.4199799362564, 4: 4306.746430783321, 8: 3227.457470326126, 12: 3849.3689750132203, 16: 13896.670732876642, 24: 13581.9036614793, 32: 18230.097926865663},
            99: {1: 1009.7958846774957, 2: 1506.4199799362564, 4: 4306.746430783321, 8: 3227.457470326126, 12: 3849.3689750132203, 16: 13896.670732876642, 24: 13581.9036614793, 32: 18230.097926865663},
        },
        'deepspeech2': {
            0: {1: 25.646389406298546, 2: 31.310407673307605, 4: 33.31379863146288, 8: 35.56125712376749, 12: 34.346949775867266, 16: 56.45748407883799, 24: 34.18455590216275, 32: 51.06061185475536},
            1: {1: 25.646389406298546, 2: 31.310407673307605, 4: 33.31379863146288, 8: 35.56125712376749, 12: 34.346949775867266, 16: 56.45748407883799, 24: 34.18455590216275, 32: 51.06061185475536},
            2: {1: 25.729855359139957, 2: 32.89997129944637, 4: 34.58732477490208, 8: 35.92407808324871, 12: 40.810520605760956, 16: 53.83469745160286, 24: 76.46063730164867, 32: 52.28765269658166},
            3: {1: 25.766867103007833, 2: 32.98935763319743, 4: 33.01949976427251, 8: 32.67407887806574, 12: 47.224406935340674, 16: 70.4873119332604, 24: 93.83355324996006, 32: 50.61374396287266},
            4: {1: 26.002745256890766, 2: 30.74412070993897, 4: 33.27628123373354, 8: 34.60505583317427, 12: 53.77992812372419, 16: 65.56819489503343, 24: 93.51160143796763, 32: 54.28141889786498},
            5: {1: 26.46098627927363, 2: 32.11537867926803, 4: 35.33750504878903, 8: 41.40395511798771, 12: 69.10808063871683, 16: 71.11107903194385, 24: 100.00669651086042, 32: 86.60961347353178},
            6: {1: 25.62226168662642, 2: 32.430417668313446, 4: 33.614540871808806, 8: 49.996117794760636, 12: 83.85037834480008, 16: 77.68537557330978, 24: 118.95814319976428, 32: 74.69225603889069},
            7: {1: 25.586588399556135, 2: 31.485629470655965, 4: 37.503176879562744, 8: 78.22860925304344, 12: 85.11467926587744, 16: 74.66626438213413, 24: 121.50154109675974, 32: 123.05245849062958},
            8: {1: 25.36363065976065, 2: 30.635475254952258, 4: 40.13997715363764, 8: 80.8134988040969, 12: 88.09425975409951, 16: 82.88469856567362, 24: 137.3055462568196, 32: 134.08688868654184},
            9: {1: 26.181441724402152, 2: 31.83889998849483, 4: 41.50997904337633, 8: 88.80273335562359, 12: 88.86164412389448, 16: 93.9056630382397, 24: 151.3299318597078, 32: 139.75510718975465},
            10: {1: 25.737349477742022, 2: 32.00707397096866, 4: 52.763036644474326, 8: 78.44754147682792, 12: 88.22892558240696, 16: 95.02928062428778, 24: 174.67879490483966, 32: 113.88768660073505},
            11: {1: 25.96148095367734, 2: 33.2765663651297, 4: 54.52992925831109, 8: 76.85080286753485, 12: 106.02627684869869, 16: 100.55873979113017, 24: 165.24012800495672, 32: 127.37210549842008},
            12: {1: 26.682390187846565, 2: 35.376247035313504, 4: 60.02045477356853, 8: 75.08591196190245, 12: 114.99494183875703, 16: 106.41047398341351, 24: 166.8422808876958, 32: 189.19425698984904},
            13: {1: 26.4483424952181, 2: 36.67066602589624, 4: 64.84479447178937, 8: 77.98576649313509, 12: 113.53520903203058, 16: 108.70153605633072, 24: 176.12492930046872, 32: 188.65287934094636},
            14: {1: 26.232160956548483, 2: 40.64080041900974, 4: 65.55794970350553, 8: 84.51590229563656, 12: 124.09049429805245, 16: 116.28397555520536, 24: 176.13530827672838, 32: 228.93358437311144},
            15: {1: 25.62406584919778, 2: 40.5042256133027, 4: 63.03408966899836, 8: 87.0723580823191, 12: 124.1626406718608, 16: 129.88743275060725, 24: 185.66411391285237, 32: 229.5396721145169},
            16: {1: 26.34944844197741, 2: 45.24346036900386, 4: 68.67211949952527, 8: 77.05384389570537, 12: 126.06454786391177, 16: 129.7722281281919, 24: 177.30452475947106, 32: 227.61802231903084},
            17: {1: 26.335942532278075, 2: 44.87246838647371, 4: 73.61020489103602, 8: 82.91693932339103, 12: 131.06458458842718, 16: 115.1693688795275, 24: 187.14182384877532, 32: 245.83921492099898},
            18: {1: 26.237389675476663, 2: 46.96295846989401, 4: 70.90612192996188, 8: 81.65898453295394, 12: 120.65992614911924, 16: 121.85768560106528, 24: 200.6740861704969, 32: 227.7689472324883},
            19: {1: 25.727747445234503, 2: 45.99018521718986, 4: 78.47558268538464, 8: 88.25041343314722, 12: 128.66480320150444, 16: 129.07353928670508, 24: 198.9101413752923, 32: 212.63192122875498},
            20: {1: 25.755635526095748, 2: 46.901379853608944, 4: 82.15861387205533, 8: 84.20631907039025, 12: 148.6307790491282, 16: 124.98187632748396, 24: 213.63614428268394, 32: 185.87476729749994},
            21: {1: 25.787074618411093, 2: 45.46998767632442, 4: 76.68417175509799, 8: 89.91978071682321, 12: 158.15208321862454, 16: 130.61042527375372, 24: 222.9122130966094, 32: 176.06666309125808},
            22: {1: 26.506576161024544, 2: 47.81776700101317, 4: 77.70941805379238, 8: 96.5237884646794, 12: 160.99775844307854, 16: 130.54125858281589, 24: 198.7823419670217, 32: 200.6519381267175},
            23: {1: 25.76599100069744, 2: 48.14810614237019, 4: 91.2262020170182, 8: 95.76283555777518, 12: 160.592145863383, 16: 135.2899930486466, 24: 199.1327230351661, 32: 265.09581818488147},
            24: {1: 29.005629765121203, 2: 47.08382814064336, 4: 88.08927102686876, 8: 101.40807333174108, 12: 185.30069015626256, 16: 165.00345547915146, 24: 198.86746082786573, 32: 262.0703047881878},
            25: {1: 30.161080151241215, 2: 49.547101535380726, 4: 91.3476326705633, 8: 108.4392831107374, 12: 171.82547832705706, 16: 136.79708190681885, 24: 186.62771611861496, 32: 178.80640227498037},
            26: {1: 29.487728901554753, 2: 47.46451503314195, 4: 85.08414750009642, 8: 118.84234609344053, 12: 161.0319522191178, 16: 142.76080241729315, 24: 227.9387430504746, 32: 200.70308230660154},
            27: {1: 31.508038510367477, 2: 49.495628205322056, 4: 86.35436369218603, 8: 108.05567533851324, 12: 170.10779937682204, 16: 162.17669923748377, 24: 199.01177148555308, 32: 200.76520399846646},
            28: {1: 31.42204300166704, 2: 49.38219650789591, 4: 91.08470549317933, 8: 115.58038589478366, 12: 177.09334704264037, 16: 165.93310313453205, 24: 227.8032991814471, 32: 229.20695002869448},
            29: {1: 31.050598558101584, 2: 54.620067376896756, 4: 86.26087273985017, 8: 120.1728511440786, 12: 160.89594290047128, 16: 157.8246584991145, 24: 186.62273053270565, 32: 268.3179725081238},
            30: {1: 29.274672815645125, 2: 51.264117226537884, 4: 83.99693781424635, 8: 124.42541078835853, 12: 161.42874571734387, 16: 153.60537666095374, 24: 209.2253487129042, 32: 247.59917066799446},
            31: {1: 32.34037140902285, 2: 49.57659413487372, 4: 87.18759471505322, 8: 116.59632813974025, 12: 176.54022756914227, 16: 156.72875206007168, 24: 213.01457299587275, 32: 247.40478694450096},
            32: {1: 31.154005847159326, 2: 48.56198766249561, 4: 84.31473536956969, 8: 121.52671542126765, 12: 183.74954265569747, 16: 149.69543462505158, 24: 227.6940847782439, 32: 292.5058005441066},
            33: {1: 31.39211841123518, 2: 46.777513853835224, 4: 86.1163510210964, 8: 126.80730388517546, 12: 176.2973839325573, 16: 150.14902649796912, 24: 225.51959554170122, 32: 268.86070414571236},
            34: {1: 31.904393229603865, 2: 49.52685312141443, 4: 78.87519391477471, 8: 140.20335199971498, 12: 198.26612916644032, 16: 158.43535969500172, 24: 214.37863786618036, 32: 265.7311834515213},
            35: {1: 30.68441262869093, 2: 45.773590322318306, 4: 85.80349235430711, 8: 136.1066255040557, 12: 187.0040882397016, 16: 151.1511291964065, 24: 188.07706728766445, 32: 246.53973026199407},
            36: {1: 31.877441700124006, 2: 46.947881449653984, 4: 86.59567964672179, 8: 140.42808469569326, 12: 175.73642432188123, 16: 144.7076391805197, 24: 224.52605221655426, 32: 292.7089351848275},
            37: {1: 32.0486522089105, 2: 46.60569954646789, 4: 82.18462822404847, 8: 142.4537875049257, 12: 177.67582256094374, 16: 158.50806031408217, 24: 289.591708640779, 32: 267.30051245859096},
            38: {1: 31.88093953731487, 2: 50.257438561977736, 4: 84.23485364650381, 8: 133.86001199081392, 12: 177.99343241980648, 16: 183.5171651905041, 24: 262.0136925643478, 32: 284.60214074922493},
            39: {1: 31.587517976664394, 2: 49.58370015565351, 4: 90.25611035770014, 8: 138.15760916723363, 12: 153.1020754631051, 16: 199.68090141239028, 24: 291.13943098405656, 32: 292.10034096788945},
            40: {1: 30.676846180425954, 2: 49.36290185087022, 4: 90.28467360214327, 8: 139.7040579236188, 12: 145.34767521134734, 16: 184.7490025122698, 24: 321.624106569876, 32: 321.2586406063976},
            41: {1: 29.848483552576475, 2: 52.306900463244794, 4: 89.91468576610485, 8: 136.76658883907086, 12: 191.67035792664015, 16: 182.23655348030698, 24: 287.83224641571877, 32: 321.3023753304319},
            42: {1: 31.627245059490257, 2: 49.86483689733213, 4: 91.63950802724669, 8: 146.06672248243825, 12: 186.49328828691765, 16: 184.12151213423286, 24: 286.42422707624024, 32: 287.98023729170677},
            43: {1: 31.96686896629971, 2: 50.527879830306276, 4: 90.7752899249776, 8: 152.1481575705722, 12: 194.60263808539622, 16: 194.01598290002826, 24: 289.7122661672538, 32: 266.08348386788555},
            44: {1: 30.62033777987219, 2: 51.69885114820509, 4: 88.38909841709462, 8: 161.45129503631213, 12: 210.95685507716422, 16: 207.94665554930464, 24: 289.83542147680555, 32: 246.74401140975544},
            45: {1: 30.117547208723188, 2: 53.063434409888174, 4: 89.71640566569397, 8: 149.01927195084906, 12: 190.38367059402842, 16: 196.88705849941607, 24: 318.58276703753506, 32: 268.01308947515946},
            46: {1: 30.838679201518055, 2: 51.18671383764489, 4: 87.49933474502072, 8: 161.5176159093863, 12: 179.94022506709376, 16: 172.89740960386553, 24: 321.86169948140383, 32: 291.4463745815071},
            47: {1: 29.473251606363103, 2: 50.34428626947243, 4: 91.80413085197125, 8: 161.34443074727898, 12: 166.21487270297058, 16: 173.60358892486445, 24: 285.9465928850723, 32: 292.1435789145497},
            48: {1: 29.914928988006018, 2: 53.00298123719729, 4: 79.62111126702085, 8: 142.1551264864987, 12: 176.41017561508764, 16: 172.84723308387498, 24: 316.4513219258394, 32: 292.75963127173327},
            49: {1: 28.31896673650106, 2: 52.22790141588286, 4: 91.57765878914151, 8: 174.01605421738955, 12: 191.54668647643592, 16: 194.31392990452156, 24: 306.40181053913113, 32: 244.0732974305105},
            50: {1: 30.43546246414771, 2: 52.07116383306128, 4: 82.93732826607625, 8: 163.81678011992483, 12: 185.45522160547299, 16: 184.65209949491785, 24: 290.7695888835985, 32: 247.02146519731474},
            51: {1: 32.379274660014424, 2: 51.19694973990505, 4: 88.76725126338152, 8: 150.51481548223254, 12: 202.42574891041647, 16: 225.17848395936844, 24: 293.1308905431034, 32: 292.7696833088228},
            52: {1: 32.91144613396595, 2: 53.629450719202815, 4: 95.9916916679987, 8: 140.91108448001404, 12: 207.71893638378813, 16: 230.38901147771148, 24: 288.2564604093939, 32: 264.8307539903361},
            53: {1: 33.207762199339626, 2: 52.05934318572878, 4: 91.63779533538396, 8: 151.04596060739595, 12: 202.2843748505072, 16: 254.53956792908855, 24: 280.32179592892817, 32: 242.00981137402857},
            54: {1: 32.20227088159619, 2: 53.49414797136755, 4: 90.90601375107383, 8: 150.50654560467123, 12: 207.87093983616296, 16: 209.69763769416267, 24: 318.1556058018824, 32: 246.11291017433282},
            55: {1: 31.65221973988071, 2: 53.90317556164827, 4: 90.77807296049573, 8: 132.93548801501112, 12: 206.0884604330978, 16: 202.51330205490459, 24: 284.4499721144851, 32: 229.61141827921185},
            56: {1: 32.594776307917755, 2: 51.96639412168281, 4: 88.52859744386363, 8: 127.58833102057818, 12: 225.14475882699273, 16: 183.92735859417263, 24: 354.9571182182325, 32: 264.93382421468584},
            57: {1: 31.75525932249487, 2: 52.90330590473729, 4: 94.0756408013204, 8: 123.0625018729343, 12: 211.25478797443142, 16: 194.7999768532737, 24: 289.79989215687704, 32: 292.7731796132578},
            58: {1: 32.27779936755244, 2: 55.84092519314889, 4: 91.6044064609272, 8: 139.23290800940177, 12: 186.27252801561482, 16: 164.78820717979815, 24: 260.3445059956989, 32: 290.2962571127711},
            59: {1: 33.924186244819246, 2: 53.990820091550354, 4: 99.23608409997757, 8: 138.44613419858172, 12: 236.46596352998867, 16: 183.7468969232224, 24: 318.14241411179785, 32: 292.0111538330347},
            60: {1: 32.216053583898734, 2: 54.83872752262448, 4: 94.06410267314031, 8: 164.3884105844598, 12: 210.0774159549912, 16: 183.41985259598061, 24: 289.1435117536209, 32: 314.7368432530593},
            61: {1: 31.962410355731514, 2: 53.431555136425864, 4: 108.04764574160453, 8: 159.14893178523795, 12: 238.78097207542018, 16: 183.9252422590155, 24: 288.86328191265704, 32: 292.5047140030112},
            62: {1: 32.290920462266605, 2: 53.71391497457585, 4: 96.92106711402064, 8: 156.96455619836493, 12: 239.38144451385944, 16: 185.03989996952637, 24: 317.4108692174514, 32: 291.32261159529355},
            63: {1: 31.847276696606222, 2: 54.26888950173622, 4: 98.6501011080728, 8: 158.20260042066892, 12: 197.8679602992972, 16: 221.89853238453358, 24: 317.65282187635887, 32: 322.2905491466746},
            64: {1: 30.651073041539913, 2: 52.41495889347948, 4: 101.12211039656485, 8: 176.90476561254994, 12: 221.14984101113018, 16: 184.88173672728774, 24: 309.4738424284363, 32: 320.3718309307782},
            65: {1: 30.367406202234246, 2: 52.719968640500326, 4: 99.00568823425746, 8: 157.3151189870662, 12: 224.12960245052082, 16: 239.27439795603732, 24: 318.9114755537579, 32: 358.7548085947918},
            66: {1: 30.735548108283748, 2: 50.36077551552121, 4: 108.07739238381642, 8: 182.3028187183221, 12: 209.38126925408653, 16: 239.84323716553476, 24: 289.72883016767236, 32: 322.48129153651655},
            67: {1: 31.897930829677982, 2: 53.942548875088384, 4: 99.29591491514194, 8: 161.67838040515986, 12: 217.05931452926868, 16: 263.9621005196105, 24: 317.7771307361826, 32: 319.3184649387181},
            68: {1: 31.1830727519556, 2: 53.14187850511707, 4: 102.39000711174437, 8: 181.08368094761067, 12: 255.94571521382198, 16: 227.65948670920275, 24: 317.0119654635021, 32: 322.08418626189},
            69: {1: 32.34259072309621, 2: 52.32240452875121, 4: 95.97310424722629, 8: 154.96279254874997, 12: 256.922688041634, 16: 258.46553706519757, 24: 290.8275794212751, 32: 399.98255877358616},
            70: {1: 31.34851065200724, 2: 56.27222842324754, 4: 104.76708537851297, 8: 169.92760110879993, 12: 224.64929984349112, 16: 251.35416069481658, 24: 320.2988555211129, 32: 320.8096297741231},
            71: {1: 32.18058055088484, 2: 54.715725260188286, 4: 100.2618046173065, 8: 156.2830938973726, 12: 237.1926464759038, 16: 259.40099288511226, 24: 318.7598146187116, 32: 356.0672088638167},
            72: {1: 30.799412820350017, 2: 54.087752281686534, 4: 104.7005299303383, 8: 157.34076250792984, 12: 210.16635607930573, 16: 209.0499859088549, 24: 316.9598484254913, 32: 398.36500705594653},
            73: {1: 29.454495141523623, 2: 54.34103953968918, 4: 103.66747688127947, 8: 163.13775514544136, 12: 210.71373431985577, 16: 223.99162570236962, 24: 353.6722431113928, 32: 319.75650503235477},
            74: {1: 23.833164525258507, 2: 55.143033075878186, 4: 97.07088524203945, 8: 180.75232425840917, 12: 241.6179501963362, 16: 207.78220328488501, 24: 399.7559413347042, 32: 402.82905714923976},
            75: {1: 24.19811846978972, 2: 54.91516691189092, 4: 105.5655701583911, 8: 148.86049972265275, 12: 224.10079025337774, 16: 195.69981254380818, 24: 319.0093854422489, 32: 292.62303415776853},
            76: {1: 26.203698307343533, 2: 52.18700708575459, 4: 99.61468637943206, 8: 161.87532558785853, 12: 257.3286414719162, 16: 243.30087639516708, 24: 308.6584840964765, 32: 355.8093850075716},
            77: {1: 27.60550689813335, 2: 57.27028342183424, 4: 102.94459429975122, 8: 149.38897467991748, 12: 268.3495167049968, 16: 240.08798961538602, 24: 396.5448652962817, 32: 353.26560737945704},
            78: {1: 28.005408974521362, 2: 55.81611058356354, 4: 99.8650130315889, 8: 154.9413981640405, 12: 200.97013313952178, 16: 289.1343040408696, 24: 352.5402927032764, 32: 320.0625982975405},
            79: {1: 28.005408974521362, 2: 55.81611058356354, 4: 99.8650130315889, 8: 154.9413981640405, 12: 200.97013313952178, 16: 289.1343040408696, 24: 352.5402927032764, 32: 320.0625982975405},
        },
    }

    assert application in goodput_functions, f"Application {application} not found in goodput functions"
    assert epoch in goodput_functions[application], f"Epoch {epoch} not found in goodput functions for application {application}"
    assert num_gpus in goodput_functions[application][epoch], f"GPU count {num_gpus} not found in goodput functions for application {application} at epoch {epoch}"
    
    if application == 'cifar10':
        return 50000 / goodput_functions[application][epoch][num_gpus]
    if application == 'bert':
        return 97077 / goodput_functions[application][epoch][num_gpus]
    if application == 'deepspeech2':
        return 4074 / goodput_functions[application][epoch][num_gpus]
    else:
        raise ValueError(f"Application {application} not supported")


def print_epoch_details(job_name, jobs, goodput_function=None):
    """Print formatted epoch details table."""
    if job_name not in jobs:
        print(f"Job {job_name} not found in results.")
        return
    
    application = job_name.split('-')[0]
    job_epochs = jobs[job_name]['epochs']
    if not job_epochs:
        print(f"No epoch data for job {job_name}.")
        return
    
    print(f"\n" + "="*70)
    print(f"EPOCH DETAILS FOR {job_name}")
    print("="*70)
    print(f"{'Epoch':<8} {'GPUs Used':<15} {'Duration(s)':<12} {'Queue(s)':<10} {'Wasted(s)':<10} {'Idle(s)':<10} {'Theoretical(s)':<15}")
    print("-" * 70)
    
    for epoch in sorted(job_epochs.keys()):
        epoch_info = job_epochs[epoch]
        gpu_list = sorted(epoch_info['gpu_allocations'])

        
        gpu_str = str(gpu_list) if len(gpu_list) > 1 else str(gpu_list[0]) if gpu_list else "0"
        
        duration = epoch_info['duration']
        queueing = epoch_info.get('queueing_time', 0)
        wasted = epoch_info.get('wasted_time', 0)
        idle = queueing + wasted
        
        # Calculate theoretical duration based on final GPU allocation
        if len(gpu_list) > 1:
            theoretical = -1
        else:
            final_gpu_count = gpu_list[0]
            theoretical = get_theoretical_duration(application, epoch, final_gpu_count)
        
        print(f"{epoch:<8} {gpu_str:<15} {duration:<12.1f} {queueing:<10.1f} {wasted:<10.1f} {idle:<10.1f} {theoretical:<15.1f}")

def print_job_breakdown(job_name, jobs):
    """Print per-epoch breakdown for a specific job: actual, wasted, theoretical, rescaling, and allocation."""
    if job_name not in jobs:
        print(f"\nJob '{job_name}' not found for breakdown.")
        return

    application = job_name.split('-')[0]
    job_epochs = jobs[job_name]['epochs']
    if not job_epochs:
        print(f"\nNo epoch data for job {job_name}.")
        return

    # Header
    print(f"\n" + "="*90)
    print(f"PER-EPOCH BREAKDOWN FOR {job_name}")
    print("="*90)
    print(f"{'Epoch':<6} {'Actual(s)':<12} {'Queue(s)':<10} {'Wasted(s)':<12} {'Idle(s)':<12} {'Theoretical(s)':<15} {'Rescale(s)':<12} {'Allocations':<20}")
    print("-" * 90)

    previous_final_gpu_count = None
    sum_actual_durations = 0.0
    sum_theoretical_durations = 0.0
    sum_rescale = 0.0

    for epoch in sorted(job_epochs.keys()):
        epoch_info = job_epochs[epoch]
        duration = epoch_info.get('duration', 0)
        queueing = epoch_info.get('queueing_time', 0)
        wasted = epoch_info.get('wasted_time', 0)
        idle = queueing + wasted

        gpu_allocations = epoch_info.get('gpu_allocations', [])
        # Format allocation string (GPU counts) and pairs if available
        alloc_str = str(gpu_allocations) if len(gpu_allocations) > 1 else (str(gpu_allocations[0]) if gpu_allocations else "0")

        # Theoretical duration: average across observed allocations if multiple
        theoretical = -1
        if gpu_allocations:
            if len(gpu_allocations) == 1:
                gpu_count = gpu_allocations[0]
                try:
                    theoretical = get_theoretical_duration(application, epoch, gpu_count)
                except (AssertionError, ValueError):
                    theoretical = -1
            else:
                durations = []
                for gpu_count in gpu_allocations:
                    try:
                        durations.append(get_theoretical_duration(application, epoch, gpu_count))
                    except (AssertionError, ValueError):
                        continue
                if durations:
                    theoretical = sum(durations) / len(durations)

        # Rescaling overhead if final allocation changed from previous epoch
        if gpu_allocations:
            final_gpu_count = gpu_allocations[-1]
        else:
            final_gpu_count = 0

        rescale = 0
        if previous_final_gpu_count is not None and final_gpu_count != previous_final_gpu_count:
            if application == "cifar10":
                rescale = 120
            elif application == "deepspeech2":
                rescale = 150
            elif application == "bert":
                rescale = 300
            else:
                rescale = 0

        previous_final_gpu_count = final_gpu_count

        sum_actual_durations += float(duration)
        if theoretical != -1:
            sum_theoretical_durations += float(theoretical)
        sum_rescale += float(rescale)

        print(f"{epoch:<6} {duration:<12.1f} {queueing:<10.1f} {wasted:<12.1f} {idle:<12.1f} {theoretical:<15.1f} {rescale:<12.1f} {alloc_str:<20}")

    # Total actual response time for this job (from first_seen to last epoch end)
    last_epoch_end = max(ei['last_seen'] for ei in job_epochs.values())
    total_response_time = last_epoch_end - jobs[job_name]['first_seen']
    total_theoretical_time = calculate_theoretical_response_time(job_name, jobs[job_name])

    print("-" * 90)
    print(f"Total Actual Response Time (first->last) (s): {total_response_time:.1f}")
    print(f"Total Actual (sum of epoch durations) (s): {sum_actual_durations:.1f}")
    print(f"Total Theoretical Response Time (s): {total_theoretical_time:.1f}")
    # Totals for queueing/wasted/idle
    total_queueing = sum(ei.get('queueing_time', 0) for ei in job_epochs.values())
    total_wasted = sum(ei.get('wasted_time', 0) for ei in job_epochs.values())
    print(f"Total Queueing Time (s): {total_queueing:.1f}")
    print(f"Total Wasted Time (s): {total_wasted:.1f}")
    print(f"Total Idle Time (s): {total_queueing + total_wasted:.1f}")

def calculate_theoretical_response_time(job_name, job_info):
    """Calculate theoretical response time for a job including rescaling overhead."""
    application = job_name.split('-')[0]
    job_epochs = job_info['epochs']
    
    if not job_epochs:
        return 0
    
    total_theoretical_time = 0
    previous_gpu_count = None
    
    for epoch in sorted(job_epochs.keys()):
        epoch_info = job_epochs[epoch]
        gpu_allocations = epoch_info['gpu_allocations']
        
        if not gpu_allocations:
            continue
            
        # Calculate theoretical running time for this epoch
        if len(gpu_allocations) == 1:
            # Single GPU allocation - use direct theoretical duration
            gpu_count = gpu_allocations[0]
            try:
                theoretical_duration = get_theoretical_duration(application, epoch, gpu_count)
                total_theoretical_time += theoretical_duration
            except (AssertionError, ValueError):
                # If theoretical duration can't be calculated, skip this epoch
                continue
        else:
            # Multiple GPU allocations - average the theoretical durations
            theoretical_durations = []
            for gpu_count in gpu_allocations:
                try:
                    theoretical_duration = get_theoretical_duration(application, epoch, gpu_count)
                    theoretical_durations.append(theoretical_duration)
                except (AssertionError, ValueError):
                    continue
            
            if theoretical_durations:
                avg_theoretical_duration = sum(theoretical_durations) / len(theoretical_durations)
                total_theoretical_time += avg_theoretical_duration
        
        # Add rescaling overhead if GPU allocation changed
        current_gpu_count = gpu_allocations[-1] if gpu_allocations else 0  # Use final allocation
        if previous_gpu_count is not None and current_gpu_count != previous_gpu_count:
            if application == "cifar10":
                total_theoretical_time += 120  # 120 seconds rescaling overhead
            elif application == "deepspeech2":
                total_theoretical_time += 150  # 150 seconds rescaling overhead
            elif application == "bert":
                total_theoretical_time += 300  # 300 seconds rescaling overhead
            else:
                raise ValueError(f"Application {application} not supported")
        
        previous_gpu_count = current_gpu_count
    
    return total_theoretical_time

def plot_response_time_comparison(jobs, output_filename=None):
    """Plot response time comparison with jobs on x-axis and horizontal dashed lines for theoretical times."""
    job_names = []
    actual_response_times = []
    theoretical_response_times = []
    
    for job_name, job_info in jobs.items():
        # Calculate actual response time
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            actual_response_time = last_epoch_end - job_info['first_seen']
        else:
            actual_response_time = 0
        
        # Calculate theoretical response time
        theoretical_response_time = calculate_theoretical_response_time(job_name, job_info)
        
        if theoretical_response_time > 0:  # Only include jobs with valid theoretical times
            job_names.append(job_name)
            actual_response_times.append(actual_response_time)
            theoretical_response_times.append(theoretical_response_time)
    
    if not job_names:
        print("No jobs with valid theoretical response times found.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual response times as bars
    x = np.arange(len(job_names))
    bars = ax.bar(x, actual_response_times, alpha=0.7, label='Actual Response Time', color='steelblue')
    
    # Add horizontal dashed lines for theoretical response times
    for i, theoretical_time in enumerate(theoretical_response_times):
        ax.axhline(y=theoretical_time, xmin=(i-0.4)/len(job_names), xmax=(i+0.4)/len(job_names), 
                  color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add a legend entry for the theoretical lines
    ax.axhline(y=-1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Theoretical Response Time')
    
    ax.set_xlabel('Jobs')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Response Time Comparison: Actual vs Theoretical')
    ax.set_xticks(x)
    ax.set_xticklabels(job_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Response time comparison plot saved to: {output_filename}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print("RESPONSE TIME COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Job Name':<20} {'Actual(s)':<12} {'Theoretical(s)':<15} {'Ratio':<8}")
    print("-" * 60)
    
    for i, job_name in enumerate(job_names):
        actual = actual_response_times[i]
        theoretical = theoretical_response_times[i]
        ratio = actual / theoretical if theoretical > 0 else float('inf')
        print(f"{job_name:<20} {actual:<12.1f} {theoretical:<15.1f} {ratio:<8.2f}")
    
    # Calculate overall statistics
    total_actual = sum(actual_response_times)
    total_theoretical = sum(theoretical_response_times)
    overall_ratio = total_actual / total_theoretical if total_theoretical > 0 else float('inf')
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_actual:<12.1f} {total_theoretical:<15.1f} {overall_ratio:<8.2f}")

    

def plot_job_response_time_stacked(jobs, output_filename=None):
    """Plot per-job total response time as stacked bars of Productive, Wasted, and Queueing.

    - X axis: job names
    - Y axis: total response time (seconds)
    - Bar segments: Productive (response - (queueing + wasted)), Wasted, Queueing
    """
    job_names = []
    response_times = []
    queue_times = []
    wasted_times = []

    for job_name, job_info in jobs.items():
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            response_time = last_epoch_end - job_info['first_seen']
        else:
            response_time = 0

        total_queueing = sum(epoch_info.get('queueing_time', 0) for epoch_info in job_info['epochs'].values())
        total_wasted = sum(epoch_info.get('wasted_time', 0) for epoch_info in job_info['epochs'].values())

        job_names.append(job_name)
        response_times.append(float(response_time))
        queue_times.append(float(total_queueing))
        wasted_times.append(float(total_wasted))

    if not job_names:
        print("No jobs to plot for stacked response time chart.")
        return

    response_arr = np.array(response_times)
    queue_arr = np.array(queue_times)
    wasted_arr = np.array(wasted_times)
    productive_arr = response_arr - (queue_arr + wasted_arr)
    productive_arr = np.maximum(productive_arr, 0.0)  # guard against small negatives

    x = np.arange(len(job_names))

    fig, ax = plt.subplots(figsize=(12, 8))
    bars_prod = ax.bar(x, productive_arr, color='steelblue', alpha=0.85, label='Productive')
    bars_wst = ax.bar(x, wasted_arr, bottom=productive_arr, color='indianred', alpha=0.85, label='Wasted')
    bars_que = ax.bar(x, queue_arr, bottom=productive_arr + wasted_arr, color='goldenrod', alpha=0.85, label='Queueing')

    ax.set_xlabel('Jobs')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Per-Job Response Time (Stacked: Productive, Wasted, Queueing)')
    ax.set_xticks(x)
    ax.set_xticklabels(job_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Stacked response time plot saved to: {output_filename}")
    else:
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python manage_monitor_log.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    print(f"Processing log file: {log_file_path}")

    jobs, total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours, wasted_capacity_hours, last_job_arrival_time, completed_jobs_status, decreased_progress_issues, job_drop_sums, job_max_progress = process_log_file(log_file_path)
    
    # Find first job time for metrics calculation
    first_job_time = min(job_info['first_seen'] for job_info in jobs.values()) if jobs else 0
    
    # Print response time and wasted time for all jobs
    print_all_jobs_summary(jobs)

    # Print jobs that completed with failing pod status
    print_failed_completed_jobs(completed_jobs_status)

    # Hardcoded specific job breakdown
    specific_job_breakdown = 'deepspeech2-155'
    print_job_breakdown(specific_job_breakdown, jobs)

    print_mean_rescaling_time(jobs)

    # plot_rescaling_time_histograms(jobs)
    # plot_rescaling_time_breakdown_histograms(jobs)

    # List of jobs to show detailed epoch information for
    # Modify this list to see details for different jobs
    detailed_jobs = []  # Add more job names here as needed
    
    # Print detailed epoch information for specified jobs
    for job_name in detailed_jobs:
        if job_name in jobs:
            print_epoch_details(job_name, jobs)
        else:
            available_jobs = list(jobs.keys())
            print(f"\nJob '{job_name}' not found. Available jobs: {available_jobs}")
            # Optionally show details for first available job if target not found
            if available_jobs and job_name == detailed_jobs[0]:  # Only for first job in list
                print_epoch_details(available_jobs[0], jobs)
    
    import os
    base_name = os.path.splitext(log_file_path)[0]  # Remove extension properly
    # Plot response time comparison
    plot_filename = base_name + '_response_time_comparison.png'
    plot_response_time_comparison(jobs, plot_filename)
    # Also save stacked response time plot
    stacked_plot_filename = base_name + '_response_time_stacked.png'
    plot_job_response_time_stacked(jobs, stacked_plot_filename)
    
    metrics = calculate_metrics(total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours, wasted_capacity_hours, last_job_arrival_time, first_job_time)
    print_summary(metrics)

    # Finally, print mean job total response time (placed at the very end)
    mean_rt = compute_mean_job_response_time(jobs)
    print("\n" + "="*60)
    print(f"Mean Job Response Time (s): {mean_rt:.1f}")
    print("="*60)

    # At the very end, warn about any jobs with decreasing progress
    print_decreasing_progress_warnings(decreased_progress_issues, job_drop_sums, job_max_progress)

    print("response_dict={")
    for job_name, job_info in jobs.items():
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            actual_response_time = last_epoch_end - job_info['first_seen']
        
        print(f"    '{job_name}': {actual_response_time:.1f},")

    print("}")

if __name__ == "__main__":
    main()