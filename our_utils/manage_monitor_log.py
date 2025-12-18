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
    # Track total GPU-hours per job (allocated GPUs * time interval, in hours)
    job_gpu_hours = {}
    # New waste decomposition metrics
    fragmentation_waste_hours = 0
    ready_unused_waste_hours = 0
    wasted_capacity_hours = 0
    scheduler_waste_hours = 0
    last_job_arrival_time = None
    completed_jobs_status = {}  # Track completion_time and pod_status for completed jobs
    decreased_progress_issues = {}  # Track any progress decreases per job
    job_drop_sums = {}  # Sum of progress drops per job
    job_max_progress = {}  # Max progress observed per job
    # Track nodes in use at each timestamp
    nodes_in_use_dict = {}
    scheduler_nodes = set()
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']

        # Capture the initial scheduler nodes (first log entry expected to have 2 nodes)
        if i == 0 and not scheduler_nodes:
            ready_node_names = log_entry.get('cluster_nodes', {}).get('ready_node_names', [])
            if len(ready_node_names) > 0:
                assert len(ready_node_names) == 2, f"Expected 2 scheduler nodes, got {len(ready_node_names)}"
            scheduler_nodes = set(ready_node_names)
        
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
            used_node_set = set(all_alloc_items)
            used_nodes = len(used_node_set)

            # Track nodes in use at this timestamp
            nodes_in_use_dict[prev_log['timestamp']] = used_nodes

            fragmentation_waste_gpus = max(0, NUM_GPU_PER_NODE * used_nodes - used_gpus)
            ready_unused_waste_gpus = max(0, NUM_GPU_PER_NODE * (ready_nodes - used_nodes))
            # Scheduler nodes waste: scheduler nodes that are not in use
            scheduler_unused_nodes = [n for n in scheduler_nodes if n not in used_node_set]
            scheduler_waste_gpus = len(scheduler_unused_nodes) * NUM_GPU_PER_NODE

            fragmentation_waste_hours += fragmentation_waste_gpus * time_diff / 3600
            ready_unused_waste_hours += ready_unused_waste_gpus * time_diff / 3600
            scheduler_waste_hours += scheduler_waste_gpus * time_diff / 3600

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
                # Accumulate per-job GPU-hours using previous state allocation
                if has_alloc and dt > 0:
                    job_gpu_hours[pname] = job_gpu_hours.get(pname, 0.0) + (len(palloc) * dt / 3600.0)

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
        scheduler_waste_hours,
        last_job_arrival_time,
        completed_jobs_status,
        decreased_progress_issues,
        job_drop_sums,
        job_max_progress,
        job_gpu_hours,
        nodes_in_use_dict,
        scheduler_nodes,
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

def calculate_metrics(total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours, wasted_capacity_hours, scheduler_waste_hours, last_job_arrival_time, first_job_time):
    """Calculate simplified metrics based on GPU hours and job arrival time."""
    experiment_duration_hours = (last_job_arrival_time - first_job_time) / 3600
    # average_gpu_usage now reflects READY GPUs average
    average_gpu_usage = total_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    effective_average_gpu_usage = effective_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    fragmentation_waste_average = fragmentation_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    ready_unused_average = ready_unused_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    scheduler_waste_average = scheduler_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    wasted_capacity_average = wasted_capacity_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0

    
    return {
        'total_gpu_hours': total_gpu_hours,
        'effective_gpu_hours': effective_gpu_hours,
        'fragmentation_waste_hours': fragmentation_waste_hours,
        'ready_unused_waste_hours': ready_unused_waste_hours,
        'scheduler_waste_hours': scheduler_waste_hours,
        'wasted_capacity_hours': wasted_capacity_hours,
        'experiment_duration_hours': experiment_duration_hours,
        'average_gpu_usage': average_gpu_usage,
        'effective_average_gpu_usage': effective_average_gpu_usage,
        'fragmentation_waste_average': fragmentation_waste_average,
        'ready_unused_average': ready_unused_average,
        'scheduler_waste_average': scheduler_waste_average,
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
    print(f"Scheduler Waste Average: {metrics['scheduler_waste_average']:.2f} GPUs")
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
                # assert next_wasted > 0, (
                #     f"Expected wasted time after rescaling for job {job_name}, epoch {epoch_num}, got 0"
                # )
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

def print_job_gpu_hours(job_gpu_hours):
    """Print total GPU-hours used by each job (allocated GPUs * time, hours)."""
    if not job_gpu_hours:
        print("\nNo per-job GPU-hour data available.")
        return
    print(f"\n" + "="*80)
    print("PER-JOB GPU HOURS")
    print("="*80)
    print(f"{'Job Name':<30} {'GPU Hours':>12}")
    print("-" * 80)
    for job_name in sorted(job_gpu_hours.keys()):
        hours = job_gpu_hours[job_name]
        print(f"{job_name:<30} {hours:>12.2f}")
    # Also print a Python-friendly dictionary for direct processing
    print("\njob_gpu_hours = {")
    for job_name in sorted(job_gpu_hours.keys()):
        hours = job_gpu_hours[job_name]
        print(f"    '{job_name}': {hours:.6f},")
    print("}")

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
        0: {1: 15.96825167355522, 2: 31.188082721094638, 4: 52.72845691453209, 8: 101.37881480030006, 16: 138.73039755109875, 24: 155.27364334267196, 32: 170.14564780156877},
        1: {1: 16.557836447402064, 2: 34.38264935810678, 4: 64.28211551464216, 8: 141.29218880778566, 16: 236.65776820830857, 24: 272.95928352184313, 32: 359.1130743579616},
    },
    'cifar10': {
        0: {1: 1013.5714309987593, 2: 943.0296061303505, 4: 1641.003340549607, 8: 1410.674885922193, 12: 2674.5499511228295, 16: 2418.0631485173863, 24: 1398.5621306833518, 32: 1316.0806400741658},
        1: {1: 1013.5714309987593, 2: 943.0296061303505, 4: 1641.003340549607, 8: 1410.674885922193, 12: 2674.5499511228295, 16: 2418.0631485173863, 24: 1398.5621306833518, 32: 1316.0806400741658},
        2: {1: 1013.5714309987593, 2: 943.0296061303505, 4: 1641.003340549607, 8: 1410.674885922193, 12: 2674.5499511228295, 16: 2418.0631485173863, 24: 1398.5621306833518, 32: 1316.0806400741658},
        3: {1: 994.954526983021, 2: 1664.8563317498808, 4: 1845.873878170769, 8: 1379.1196669360506, 12: 2248.7417617256265, 16: 2979.6040487898595, 24: 1359.4239673816314, 32: 1305.4308181000083},
        4: {1: 747.7024978118161, 2: 1614.6049424312791, 4: 1689.5758738694371, 8: 1362.3791366622854, 12: 1902.108722019539, 16: 3225.4395193861474, 24: 1419.3004987089087, 32: 1312.8990681101152},
        5: {1: 953.6754600220341, 2: 1566.567158633992, 4: 1785.4648158339398, 8: 1390.234988240772, 12: 2486.6020555616515, 16: 3552.9264900663898, 24: 1419.250859684994, 32: 1283.4847396653129},
        6: {1: 977.7789778392585, 2: 1628.8261406326383, 4: 1652.3292807280893, 8: 1358.7356141749895, 12: 3059.170859976754, 16: 3700.5120920853365, 24: 1434.489921709794, 32: 3833.9471482870536},
        7: {1: 1015.8611231604368, 2: 863.2705595397092, 4: 1553.067825963764, 8: 1406.8101210932496, 12: 3428.518942821861, 16: 3821.559300863485, 24: 1363.5659648897365, 32: 4548.340169435377},
        8: {1: 677.9496944699547, 2: 1601.361998176313, 4: 2767.5654883398256, 8: 1387.3628852985435, 12: 2990.764032142958, 16: 3835.3822769408976, 24: 4715.999286128661, 32: 5999.179899103151},
        9: {1: 1021.1003289828699, 2: 1601.708758777395, 4: 2791.250548471091, 8: 1378.4037773365105, 12: 3215.03372492373, 16: 3995.4248262672386, 24: 4573.4308272341905, 32: 6175.275195773314},
        10: {1: 986.8213599492735, 2: 1598.7568218896367, 4: 1633.8281522242276, 8: 1370.6874996796153, 12: 3098.922844021129, 16: 4084.5681662501993, 24: 5293.670677761883, 32: 6421.1741747347005},
        11: {1: 708.1704224348636, 2: 1663.8050496447265, 4: 1874.2134788095045, 8: 1343.5052135898204, 12: 3374.1953021198474, 16: 3783.592117545584, 24: 5344.042328655002, 32: 5811.796996042115},
        12: {1: 993.0988810950823, 2: 908.2245665249824, 4: 2880.9325896204246, 8: 1358.6417752756745, 12: 3431.852061048785, 16: 3663.2266416929874, 24: 6443.375815521897, 32: 7161.859418638551},
        13: {1: 1006.208139313585, 2: 1476.9485269839652, 4: 2712.5628913646924, 8: 1392.8217195094726, 12: 3444.1450911598827, 16: 2414.3242164276185, 24: 5617.03057973154, 32: 7203.918733553826},
        14: {1: 725.7672344692053, 2: 1682.765638632026, 4: 2493.867183770674, 8: 1308.764081409834, 12: 3793.1919964433378, 16: 2407.3309644011974, 24: 5713.020008050588, 32: 6798.678360746351},
        15: {1: 939.7978029323912, 2: 1570.8330758537277, 4: 2703.6938465492203, 8: 1331.926308093815, 12: 2576.5178644876582, 16: 4405.666876887215, 24: 5000.22768752611, 32: 7440.082206037931},
        16: {1: 993.8451439396183, 2: 1610.7874546783103, 4: 2667.3766940209084, 8: 1335.1089703638086, 12: 2579.4093755468834, 16: 4357.5771517444055, 24: 5804.823742888084, 32: 7568.607378910889},
        17: {1: 1014.1225399337055, 2: 1025.3183278441475, 4: 2512.8396979623935, 8: 1330.3778393069524, 12: 2717.025955642181, 16: 4526.804712623574, 24: 5835.805433026283, 32: 7667.342337014903},
        18: {1: 285.0654417455307, 2: 1377.5950887132087, 4: 2721.5397581988086, 8: 1379.25669155049, 12: 3511.426234946565, 16: 4424.664943781437, 24: 5137.5372447666505, 32: 7223.390421920295},
        19: {1: 795.1295662757801, 2: 1609.3853128471476, 4: 2950.2262915435404, 8: 1375.455804723307, 12: 3699.609124916888, 16: 4091.907879857912, 24: 5986.412650667445, 32: 7841.237077258239},
        20: {1: 1001.7966845782105, 2: 1659.175119785301, 4: 2705.194912073531, 8: 1377.4511218645084, 12: 3590.3106853855334, 16: 4554.241368712126, 24: 6118.182928668389, 32: 6343.713694692176},
        21: {1: 675.5084652231473, 2: 1557.6580729332052, 4: 2763.1476883821015, 8: 1425.480001914207, 12: 3293.4020816718294, 16: 4099.734161031572, 24: 6033.4566507004665, 32: 7900.379331717429},
        22: {1: 1006.5372511817263, 2: 1364.8869885560766, 4: 2699.012944545104, 8: 1358.1919100385226, 12: 3603.4722092299294, 16: 4133.690759804403, 24: 6193.152255560644, 32: 7357.34418637291},
        23: {1: 1023.126618655247, 2: 924.0286787471085, 4: 2704.474460146046, 8: 1379.0506708253629, 12: 3552.018783149077, 16: 3779.139284696327, 24: 6120.65095991086, 32: 7937.5270321918015},
        24: {1: 874.9415114423354, 2: 1567.294531228291, 4: 2703.668206612522, 8: 1365.7081346577288, 12: 3621.616700519836, 16: 4136.1171887925, 24: 5832.961367200498, 32: 6686.284438226702},
        25: {1: 791.5106496522474, 2: 1678.4399528834335, 4: 2874.479943886422, 8: 1368.1930056018036, 12: 3544.3373730910234, 16: 4170.76243217278, 24: 6251.418607953919, 32: 8574.609019328129},
        26: {1: 1001.0953699195729, 2: 1573.1235170980704, 4: 2613.7322080937433, 8: 1428.3949875619703, 12: 3629.623690985066, 16: 4117.780993607393, 24: 6204.293383394848, 32: 6302.911330397448},
        27: {1: 892.9809776384444, 2: 1616.7198438858393, 4: 2877.2344096729475, 8: 1366.108902039856, 12: 3085.739193973562, 16: 3991.9544270981364, 24: 6207.866870302107, 32: 6817.033837849558},
        28: {1: 785.0737103701063, 2: 426.1169757052443, 4: 2652.770181920526, 8: 1377.6164265086786, 12: 2731.8711320029934, 16: 2822.321432417803, 24: 6211.962709530692, 32: 6804.001647473258},
        29: {1: 1008.2961356825281, 2: 1147.4830168930328, 4: 2771.960885879868, 8: 1381.007190793898, 12: 2587.4033117512913, 16: 2684.3157816242747, 24: 6288.730564463081, 32: 6791.199391641688},
        30: {1: 1004.2551717953696, 2: 1147.4830168930328, 4: 1860.8795430927016, 8: 1385.3563201285278, 12: 3533.5886946937326, 16: 3358.2518017081484, 24: 6374.998181968677, 32: 6932.534504276681},
        31: {1: 699.2548548546503, 2: 1621.3335078630405, 4: 1785.118009234518, 8: 1386.9992255054683, 12: 3610.517983792907, 16: 4242.015116792734, 24: 6544.172564277265, 32: 7460.271265993014},
        32: {1: 987.6671252547363, 2: 1557.693427357374, 4: 2540.2056112855466, 8: 1355.389153542334, 12: 3628.864288422268, 16: 4256.227759773348, 24: 6342.5949298403675, 32: 6990.286368140109},
        33: {1: 1005.5479581229063, 2: 1100.2749370031108, 4: 2629.9685563411317, 8: 3282.271439248415, 12: 3664.7304490703427, 16: 4221.322736278987, 24: 6371.242379915072, 32: 8688.686357100934},
        34: {1: 910.0279996048971, 2: 1166.7964667583701, 4: 2869.210571504653, 8: 3081.9275091638974, 12: 3589.5743308552746, 16: 4813.452467674206, 24: 6314.581193212387, 32: 6928.697228204336},
        35: {1: 778.0254912242856, 2: 1632.4457312475633, 4: 2649.981923808486, 8: 3106.7194493392135, 12: 3669.5142014178314, 16: 4205.478116230861, 24: 6378.4632292449705, 32: 8068.569125654416},
        36: {1: 1010.4995770750315, 2: 1446.2321094993904, 4: 2970.2705247478493, 8: 3132.2810900400377, 12: 3707.2560307145986, 16: 4835.466719180976, 24: 6574.8923186181055, 32: 8722.43102098433},
        37: {1: 1006.323978558657, 2: 2099.5245543683554, 4: 2815.9786898448183, 8: 3114.522730162314, 12: 3623.1444089138067, 16: 4870.710035788818, 24: 6526.77241280307, 32: 7073.365540341886},
        38: {1: 670.1520872667733, 2: 2098.343586978059, 4: 2697.3905918595765, 8: 3254.7722358195033, 12: 4034.254228026331, 16: 4322.592136926421, 24: 6541.817148750223, 32: 7101.051326576036},
        39: {1: 1034.3307797702687, 2: 1397.052702031397, 4: 2672.163303657763, 8: 3376.5612841614134, 12: 2745.496817953381, 16: 4873.3708354943055, 24: 6833.389452883082, 32: 8847.942158597007},
        40: {1: 1000.7981612550342, 2: 1651.8282603537593, 4: 2816.1739458558377, 8: 3072.992524975514, 12: 2256.7362405930967, 16: 4298.561581823229, 24: 6492.059894357276, 32: 6834.389475160372},
        41: {1: 732.0032116985396, 2: 2166.4588144145664, 4: 2795.8256151947135, 8: 3089.1779467706087, 12: 2901.6414260566057, 16: 4331.447641453162, 24: 5759.992823032956, 32: 7056.201531337926},
        42: {1: 951.879063393686, 2: 2074.093908927295, 4: 2816.334319350361, 8: 3299.80506268624, 12: 3592.128729174138, 16: 4876.104972449855, 24: 6662.535744288076, 32: 9059.936030223484},
        43: {1: 1014.4298015957659, 2: 2210.692650783365, 4: 2666.1173570098745, 8: 3406.6779292059773, 12: 3713.900853348709, 16: 3199.6871701939117, 24: 6930.544987019747, 32: 9081.744691281705},
        44: {1: 998.475749401605, 2: 2221.985975342753, 4: 2868.024885788786, 8: 3099.3917784583105, 12: 3716.8507895427856, 16: 2998.2596807086406, 24: 6554.620273002891, 32: 7579.779841497028},
        45: {1: 697.4312688624694, 2: 1811.1578470940167, 4: 2853.8283531179236, 8: 3413.678719251145, 12: 3741.973060890405, 16: 2838.033787253725, 24: 6613.032355502275, 32: 9007.426791798389},
        46: {1: 1029.6769811763754, 2: 1379.8376554462584, 4: 2790.169966305437, 8: 3424.5489117028974, 12: 3998.998960702332, 16: 4839.158571910974, 24: 7025.734896154904, 32: 7336.094363851473},
        47: {1: 1007.7113057332351, 2: 2189.0003493186027, 4: 2935.4796631939253, 8: 3133.8807606042055, 12: 3755.121186901569, 16: 4389.641625656092, 24: 5944.395416738541, 32: 7262.481606872156},
        48: {1: 670.3061265233638, 2: 2096.233260776668, 4: 2573.031463476053, 8: 3167.5665736445662, 12: 3707.050266625486, 16: 4950.873138217285, 24: 6983.371417536941, 32: 9101.826019575088},
        49: {1: 1004.2325897917549, 2: 2218.965897489875, 4: 3122.3730989542078, 8: 3419.1134462094087, 12: 3744.194503028865, 16: 4761.578827943044, 24: 5871.248719140032, 32: 9019.654657975801},
        50: {1: 987.5142119401304, 2: 2103.9121577686105, 4: 2854.732495793854, 8: 3171.8215335060245, 12: 3766.3674398131725, 16: 4956.248348287967, 24: 7008.041856824317, 32: 9412.549702105818},
        51: {1: 798.1094915724732, 2: 2045.1928716454393, 4: 2427.89814764811, 8: 3167.7451497584434, 12: 3731.8102589875666, 16: 4971.582309770046, 24: 5933.55805336337, 32: 9334.242825298294},
        52: {1: 882.8343152791584, 2: 2157.4317634101876, 4: 3526.7080515111793, 8: 3430.4599011186456, 12: 2624.402877406714, 16: 4967.04429516981, 24: 7139.829575060397, 32: 7450.680673493245},
        53: {1: 1021.1768235310343, 2: 1494.249212494135, 4: 3861.305994132565, 8: 3382.256993628851, 12: 2420.2196153323216, 16: 4515.5931337899565, 24: 5893.1073812580735, 32: 7298.367579827328},
        54: {1: 1018.2674350663691, 2: 1528.3612342954655, 4: 4106.69428714531, 8: 3450.5694878575323, 12: 2870.1784348641268, 16: 4426.087718780391, 24: 7118.872703530787, 32: 9147.728530177534},
        55: {1: 754.8835553363411, 2: 2058.132660536852, 4: 3963.4354818876946, 8: 3188.803287142216, 12: 3792.624801890516, 16: 4971.076153415605, 24: 5976.936412202329, 32: 9417.62698279487},
        56: {1: 1036.0423730267676, 2: 2150.748881992835, 4: 3826.3163917588886, 8: 3423.793557960396, 12: 3737.441222304432, 16: 5013.944511214633, 24: 7174.316038729155, 32: 9460.844004457122},
        57: {1: 1004.2651082042515, 2: 2148.651494932642, 4: 3999.5593705662977, 8: 3313.7170889490817, 12: 3758.536385789662, 16: 4951.87409084337, 24: 6019.35184226317, 32: 9331.920945594604},
        58: {1: 911.687628569463, 2: 2048.2039998787322, 4: 3626.542909058523, 8: 3441.1332944225173, 12: 3793.646283981895, 16: 5014.309054460985, 24: 7184.616155162188, 32: 9701.80302700982},
        59: {1: 924.918745696167, 2: 2120.8992387519415, 4: 3995.241023173986, 8: 3465.5520256194413, 12: 3741.926806545007, 16: 3578.6394632715173, 24: 6016.039660868934, 32: 9419.929202592308},
        60: {1: 998.2569410082876, 2: 1923.6575791114124, 4: 3924.1480558989083, 8: 3130.3590348514686, 12: 3763.765151295122, 16: 2417.6069028625175, 24: 7237.443217363916, 32: 9473.512867519818},
        61: {1: 1020.5110900649956, 2: 1449.8864024016143, 4: 4001.3524340743365, 8: 3391.898678141484, 12: 3788.3381169511003, 16: 2607.488352220477, 24: 5962.058828286311, 32: 7788.286064419148},
        62: {1: 786.9814404338697, 2: 2106.57299929136, 4: 3992.5307769360898, 8: 3474.625013954759, 12: 3431.933555210462, 16: 4948.529589007362, 24: 6014.615078105434, 32: 9735.16428429942},
        63: {1: 1026.8854484800795, 2: 2161.049597214974, 4: 4024.612421308238, 8: 3241.2874897001125, 12: 3784.202451307327, 16: 4994.874144175887, 24: 7268.8278572990685, 32: 9783.692072172644},
        64: {1: 1021.6613409297779, 2: 2061.615981240677, 4: 3678.4355263703796, 8: 3529.208536921033, 12: 3446.0265098807395, 16: 4959.931907298177, 24: 5946.248498845243, 32: 9586.041688784608},
        65: {1: 900.3805060062348, 2: 2060.766104631324, 4: 4050.2525755438023, 8: 3270.901186573893, 12: 2772.6950229777935, 16: 5056.343544514681, 24: 6010.539187187474, 32: 9617.696461097396},
        66: {1: 956.0761441171086, 2: 2146.653057225325, 4: 4046.2306315810024, 8: 3564.6131388887457, 12: 2533.127524170099, 16: 4815.322621783123, 24: 7274.071059867679, 32: 8040.841597975647},
        67: {1: 1027.412964631691, 2: 2177.4897301928386, 4: 3924.9375877839957, 8: 3462.0947117704677, 12: 3195.370456553817, 16: 4929.488520991755, 24: 6093.079510805167, 32: 9539.011856715548},
        68: {1: 1028.8555367417284, 2: 1287.3174934062317, 4: 4007.028104819447, 8: 3284.8436470082015, 12: 3483.4020048509956, 16: 4989.927844961241, 24: 6066.681291486032, 32: 9619.255810426264},
        69: {1: 806.8086530465969, 2: 1762.3437409711175, 4: 4062.9184200831123, 8: 3508.192929430614, 12: 3908.3637490087026, 16: 4602.087572808394, 24: 7353.510619108516, 32: 9694.000266403516},
        70: {1: 1006.4910142970413, 2: 2072.278624473175, 4: 3923.3385641439227, 8: 3558.1882250014514, 12: 3724.6456796401526, 16: 4626.7671213883295, 24: 7280.6925600738, 32: 8092.201455327977},
        71: {1: 1056.8658018005267, 2: 2168.9071833796143, 4: 4010.76399170605, 8: 3499.042913282813, 12: 3907.327562916652, 16: 5057.4310805492105, 24: 6083.488299790239, 32: 9738.429345210969},
        72: {1: 953.6340424393027, 2: 2236.3284864334983, 4: 3925.2861928716707, 8: 3302.276826418432, 12: 3471.5771438168535, 16: 4604.321232559604, 24: 7391.025534956969, 32: 8131.973712142867},
        73: {1: 951.9019077350281, 2: 2180.899556693075, 4: 4014.824730638165, 8: 3483.0160273363026, 12: 3908.2102050295534, 16: 5004.681325349529, 24: 7350.363106129466, 32: 8690.054009525848},
        74: {1: 1025.6502195176543, 2: 2040.8267112416074, 4: 3693.470480856133, 8: 3312.7237705666203, 12: 3505.009402428483, 16: 5279.69066802271, 24: 7040.290127323589, 32: 8165.7491692798985},
        75: {1: 1021.7351761559813, 2: 1481.3919375175556, 4: 4059.28324229957, 8: 3312.8482342287393, 12: 3914.435502266179, 16: 3281.7554394638373, 24: 6180.926355053129, 32: 7861.382838173997},
        76: {1: 778.1532939699398, 2: 1540.9315714100435, 4: 4070.628141616457, 8: 3619.060966513566, 12: 3563.3397932439634, 16: 2879.8267013829973, 24: 6007.771949128884, 32: 8159.725877049033},
        77: {1: 1007.7359506784969, 2: 2176.3990946426024, 4: 4020.7050603485022, 8: 3508.3104453912442, 12: 3563.0164477164235, 16: 2890.4538130231144, 24: 7404.17254444895, 32: 8205.028941557746},
        78: {1: 1019.2810226503369, 2: 2190.0845840291386, 4: 4040.6638627593957, 8: 3641.2454104117965, 12: 3043.4480192959695, 16: 4643.539646379825, 24: 7396.242055990324, 32: 8555.97092881122},
        79: {1: 930.1856591346657, 2: 2181.054102291204, 4: 3867.224796075595, 8: 3626.4256486997247, 12: 2541.7172940256482, 16: 4613.552009266672, 24: 7405.235958304743, 32: 8217.525835367136},
        80: {1: 898.6052681990983, 2: 2197.681845586923, 4: 4054.5475196390225, 8: 3270.5454903469017, 12: 2593.310429017854, 16: 4641.4611050699605, 24: 6164.458436405312, 32: 7793.918973623014},
        81: {1: 1038.517302881615, 2: 2194.012269182055, 4: 2528.845718364165, 8: 3554.8653822663496, 12: 3920.5186649853645, 16: 5072.737177757081, 24: 7458.363728567617, 32: 8302.998373712539},
        82: {1: 1017.2793751740141, 2: 1974.2598314755655, 4: 2711.510776058483, 8: 3446.79594847984, 12: 3884.6346096624047, 16: 4654.697255046088, 24: 7382.354471367564, 32: 8307.30182993873},
        83: {1: 828.8952787326737, 2: 1288.759690111693, 4: 3299.054314939432, 8: 3556.5018072881794, 12: 3549.5915471408725, 16: 5107.930068271206, 24: 7464.001593035566, 32: 7292.8677396720195},
        84: {1: 1031.6501615237325, 2: 2083.773820537221, 4: 4031.312885140385, 8: 3542.463959425248, 12: 3904.7099703554723, 16: 4449.221940475767, 24: 6191.207314479568, 32: 9724.91957271734},
        85: {1: 1037.2398526690279, 2: 2186.765680434741, 4: 4038.3625777682078, 8: 3507.1471974117135, 12: 3560.9252412578694, 16: 4644.191097550631, 24: 6245.472257696964, 32: 6345.464233685221},
        86: {1: 944.7232273500234, 2: 2203.9813186183746, 4: 4257.125364770689, 8: 3512.438409858994, 12: 3837.380873934246, 16: 4642.3137994815825, 24: 7474.909995867334, 32: 23506.975861580617},
        87: {1: 944.7232273500234, 2: 2203.9813186183746, 4: 4257.125364770689, 8: 3512.438409858994, 12: 3837.380873934246, 16: 4642.3137994815825, 24: 7474.909995867334, 32: 23506.975861580617},
        88: {1: 1023.1385997047987, 2: 2172.3965653644373, 4: 4071.600501844063, 8: 3431.18093347834, 12: 3556.164462295258, 16: 4656.086942921516, 24: 7157.882943296952, 32: 11415.07340899738},
        89: {1: 827.6806119795838, 2: 1186.1579231213707, 4: 4061.324598490053, 8: 3509.8477415481184, 12: 3921.206070288756, 16: 4393.419075920226, 24: 5358.452820984432, 32: 17689.33080589964},
        90: {1: 827.6806119795838, 2: 1186.1579231213707, 4: 4061.324598490053, 8: 3509.8477415481184, 12: 3921.206070288756, 16: 4393.419075920226, 24: 5358.452820984432, 32: 17689.33080589964},
        91: {1: 1031.6885357836638, 2: 2147.7236985158884, 4: 4044.965077725922, 8: 3506.7819759340337, 12: 2836.013387321122, 16: 2774.7740106983774, 24: 19196.761211654808, 32: 23030.06866027586},
        92: {1: 1012.0393980272112, 2: 2095.640501668526, 4: 4047.8368466809043, 8: 3450.9426031684175, 12: 2428.587069234277, 16: 2834.8895207029577, 24: 21655.72972301128, 32: 11595.035254147133},
        93: {1: 936.4741755122571, 2: 2155.353534527718, 4: 4042.6486442263713, 8: 3502.191869525565, 12: 2817.670090989608, 16: 3237.294566815765, 24: 10828.532795907955, 32: 23003.43717321322},
        94: {1: 926.9261244818489, 2: 2140.1720424443906, 4: 4090.8508079428916, 8: 3500.9443658656514, 12: 3750.1521021831413, 16: 4660.637252265508, 24: 10883.860096397639, 32: 22868.61239361407},
        95: {1: 1013.280728105032, 2: 1533.342646800167, 4: 4546.8498032813495, 8: 3232.6629734698217, 12: 3575.6930720238474, 16: 15465.14816982006, 24: 10941.357053313433, 32: 23035.717362756364},
        96: {1: 1013.280728105032, 2: 1533.342646800167, 4: 4546.8498032813495, 8: 3232.6629734698217, 12: 3575.6930720238474, 16: 15465.14816982006, 24: 10941.357053313433, 32: 23035.717362756364},
        97: {1: 1013.280728105032, 2: 1533.342646800167, 4: 4546.8498032813495, 8: 3232.6629734698217, 12: 3575.6930720238474, 16: 15465.14816982006, 24: 10941.357053313433, 32: 23035.717362756364},
        98: {1: 1013.280728105032, 2: 1533.342646800167, 4: 4546.8498032813495, 8: 3232.6629734698217, 12: 3575.6930720238474, 16: 15465.14816982006, 24: 10941.357053313433, 32: 23035.717362756364},
        99: {1: 1013.280728105032, 2: 1533.342646800167, 4: 4546.8498032813495, 8: 3232.6629734698217, 12: 3575.6930720238474, 16: 15465.14816982006, 24: 10941.357053313433, 32: 23035.717362756364},
    },
    'deepspeech2': {
        0: {1: 25.616089119901172, 2: 31.33029891888372, 4: 32.99158697687773, 8: 35.49467763687132, 12: 34.29489752955102, 16: 56.66943501834432, 24: 33.78713337921309, 32: 51.53318538988096},
        1: {1: 25.616089119901172, 2: 31.33029891888372, 4: 32.99158697687773, 8: 35.49467763687132, 12: 34.29489752955102, 16: 56.66943501834432, 24: 33.78713337921309, 32: 51.53318538988096},
        2: {1: 25.699110919792506, 2: 32.764817821496756, 4: 34.328955355207015, 8: 36.12242116220577, 12: 40.496923115575576, 16: 53.874713975589515, 24: 77.23087181821438, 32: 52.203498042091795},
        3: {1: 25.60965534267906, 2: 32.85543577500836, 4: 32.877479845135994, 8: 32.612818891410186, 12: 46.790622642208334, 16: 68.98665602146824, 24: 90.92021742639363, 32: 49.32488986125676},
        4: {1: 25.97368099550909, 2: 30.752523486782096, 4: 33.21078927986425, 8: 34.17349828123464, 12: 53.8514102171132, 16: 64.56216898240974, 24: 91.26269096414488, 32: 54.55261206961281},
        5: {1: 26.435277746640185, 2: 32.13650591336682, 4: 35.216671382670654, 8: 40.85829646738041, 12: 67.26855779493494, 16: 69.39879694968057, 24: 97.605127498586, 32: 84.09130606777313},
        6: {1: 25.33782013829937, 2: 32.13022710358339, 4: 33.722481425502565, 8: 49.40994941433321, 12: 83.47092035712869, 16: 76.97895449783022, 24: 117.06994994351815, 32: 73.67325554560432},
        7: {1: 25.430198967395988, 2: 31.505198861586315, 4: 36.855961575408685, 8: 78.62857420776541, 12: 82.54808974868313, 16: 73.81636547458223, 24: 119.52277088550787, 32: 119.51926753013302},
        8: {1: 25.205552841908503, 2: 30.62743623450578, 4: 39.92644450923454, 8: 80.33205263810859, 12: 87.88139446930239, 16: 83.78511120628619, 24: 130.8126458798246, 32: 130.07641157049744},
        9: {1: 26.025081652176773, 2: 31.531384336441395, 4: 41.70352255069523, 8: 83.2632595723968, 12: 86.97703442250501, 16: 91.85517243924743, 24: 150.36558771516215, 32: 133.75973721556915},
        10: {1: 25.711573974255955, 2: 31.911781219638613, 4: 52.020564772637826, 8: 75.54699973129448, 12: 85.76163988420014, 16: 91.61333056643443, 24: 162.7782624147829, 32: 107.64849200564794},
        11: {1: 25.948106195325774, 2: 33.16523550470039, 4: 53.05388720940351, 8: 74.49594808046169, 12: 100.72011302057648, 16: 98.94131501295024, 24: 156.31506842489463, 32: 130.51201888954915},
        12: {1: 26.662116790434073, 2: 35.464059052262535, 4: 58.134778331744116, 8: 74.84345461189554, 12: 110.37358909492184, 16: 105.26229053088716, 24: 160.9113437368433, 32: 184.8279404881673},
        13: {1: 26.29838872084487, 2: 36.42232778848059, 4: 62.60914752331095, 8: 76.64631457727114, 12: 106.09857075503344, 16: 101.8393113549787, 24: 174.02092414388272, 32: 176.41538424133273},
        14: {1: 26.20979467470706, 2: 39.63275779291018, 4: 63.53615295478772, 8: 84.01081579127742, 12: 115.14818338205194, 16: 113.77462250667294, 24: 160.4257131099302, 32: 217.30640129215323},
        15: {1: 25.598560763158815, 2: 40.01588418391039, 4: 61.82426338668385, 8: 84.47415658815905, 12: 119.94321220767696, 16: 122.6355150058143, 24: 180.98728977803833, 32: 207.06205179615353},
        16: {1: 26.06808792800619, 2: 44.19895534474975, 4: 67.7438020279932, 8: 73.90677326616665, 12: 124.70146826209239, 16: 121.51690752881146, 24: 171.6434357542202, 32: 204.14371694259498},
        17: {1: 26.315672592798762, 2: 44.34460104682164, 4: 73.03079096862365, 8: 81.79075549774232, 12: 118.97836379810998, 16: 112.66293418073056, 24: 181.65536055308152, 32: 232.10892815618226},
        18: {1: 26.084850958467204, 2: 46.396029696781866, 4: 68.0542765639959, 8: 79.70451133369123, 12: 119.11029558659537, 16: 117.09008336720402, 24: 185.99832717976412, 32: 202.93190905154336},
        19: {1: 25.573878670549064, 2: 45.243354670931176, 4: 76.90352717545973, 8: 86.80234268853994, 12: 122.88572909516434, 16: 118.52098274442503, 24: 185.78952798985713, 32: 207.09017734515353},
        20: {1: 25.728971213529267, 2: 46.56115687603187, 4: 78.1316347364701, 8: 82.24212469212647, 12: 142.71269699309664, 16: 122.7635316197856, 24: 198.82116192950775, 32: 169.67448321549625},
        21: {1: 25.635560405093287, 2: 44.31312664370468, 4: 74.50166985432861, 8: 85.459237057629, 12: 134.79903195656894, 16: 120.81798903778576, 24: 188.8434682002739, 32: 171.04126186671328},
        22: {1: 26.48640265291746, 2: 47.33354788049138, 4: 75.4125858075869, 8: 93.61543888385178, 12: 143.43400116354053, 16: 124.50946263086605, 24: 184.00007307529353, 32: 198.6549196626821},
        23: {1: 25.61148276016941, 2: 47.36478533204381, 4: 86.77910971163216, 8: 92.89586631605984, 12: 151.33739012364887, 16: 131.66439309978665, 24: 174.24946993969905, 32: 201.93946771560758},
        24: {1: 28.318486004886694, 2: 46.671145622659544, 4: 86.2253299435502, 8: 97.98067005045631, 12: 175.8234004275829, 16: 153.6778991041172, 24: 186.77731286214987, 32: 247.32410220205023},
        25: {1: 29.009624670745712, 2: 49.1538207341356, 4: 87.08462963282267, 8: 103.50357968959885, 12: 148.67439839021444, 16: 130.5182130405864, 24: 183.69019624900235, 32: 153.71805924155424},
        26: {1: 28.905861259230242, 2: 46.998273253228774, 4: 81.91501972121827, 8: 114.34407358373873, 12: 143.2057190634578, 16: 132.02502651736236, 24: 190.1284042523917, 32: 187.30606574037586},
        27: {1: 30.338814625790445, 2: 48.24096037176051, 4: 85.54532603747188, 8: 102.23060371380635, 12: 165.4977515484019, 16: 159.5493768001077, 24: 190.72553718718984, 32: 192.5158354917522},
        28: {1: 30.410868829467734, 2: 48.875517744025586, 4: 85.87364624468046, 8: 108.07531031341396, 12: 154.29137770285482, 16: 146.52984065791964, 24: 192.406193108765, 32: 209.91763114108554},
        29: {1: 29.59546421419262, 2: 53.84077261483495, 4: 83.66190321812778, 8: 113.74149123127468, 12: 143.4844341229148, 16: 151.61270658338478, 24: 174.6722830304989, 32: 257.99030678689167},
        30: {1: 28.764907638048534, 2: 51.21217803085837, 4: 80.66451008630176, 8: 118.06486169797284, 12: 152.962757109358, 16: 139.85735684277023, 24: 197.3671028805377, 32: 210.72323564085866},
        31: {1: 30.76084960059076, 2: 49.2903223834134, 4: 83.22131791201564, 8: 105.4020476391965, 12: 151.0432455044135, 16: 142.80860290874017, 24: 205.62860065598966, 32: 228.7680395599621},
        32: {1: 30.356836215104515, 2: 46.777581755879616, 4: 82.95915244684197, 8: 117.85706433859653, 12: 165.77772642045588, 16: 140.12895067776182, 24: 201.32359566382118, 32: 274.2066612035904},
        33: {1: 30.174549360928175, 2: 45.33049613255878, 4: 77.98457455527702, 8: 118.79304493349659, 12: 160.1874306745629, 16: 144.97959447323498, 24: 202.33125473608052, 32: 225.20504668709412},
        34: {1: 30.553382599048238, 2: 48.0398247670497, 4: 75.10455597666713, 8: 129.7907414314962, 12: 185.62481468463614, 16: 153.03108853423453, 24: 189.4660839824057, 32: 211.058172655174},
        35: {1: 30.065364237547108, 2: 44.87535760094922, 4: 78.7717765900253, 8: 128.0994642542903, 12: 165.55212834531096, 16: 140.40857580928133, 24: 179.5190189612166, 32: 242.5052256064649},
        36: {1: 30.656143109264143, 2: 45.5438472958776, 4: 79.48776873605999, 8: 129.69751907030604, 12: 163.0568928274414, 16: 133.2489299448394, 24: 199.65732126368385, 32: 252.21581011594145},
        37: {1: 31.11123623517929, 2: 45.3669442384101, 4: 76.0634584891385, 8: 129.23232559501216, 12: 166.69986026650645, 16: 147.39652109723446, 24: 202.4740354254824, 32: 256.07564718981723},
        38: {1: 30.625521653871402, 2: 48.13022909526343, 4: 76.63738921883397, 8: 125.8447390928938, 12: 158.31087251404355, 16: 177.59756054518704, 24: 241.31846114562077, 32: 256.1979268132682},
        39: {1: 30.305880254019396, 2: 47.925369461665284, 4: 81.91849162791529, 8: 126.80938201558148, 12: 144.42416363231982, 16: 159.29235607807797, 24: 228.27461065350707, 32: 245.51240847659034},
        40: {1: 30.116012483426672, 2: 48.85975036578504, 4: 84.37345891423931, 8: 130.3530210867954, 12: 140.93590494459556, 16: 167.89519282133745, 24: 282.477790179757, 32: 298.48385992442667},
        41: {1: 28.627284879882524, 2: 49.72121422312662, 4: 84.48366444626237, 8: 126.39874165918772, 12: 164.03573988882712, 16: 172.1127926362646, 24: 240.12192838883317, 32: 256.1968340497292},
        42: {1: 30.59274964403845, 2: 48.42489523410698, 4: 81.61451813927704, 8: 141.2732605383663, 12: 172.11637475739371, 16: 167.5079853329143, 24: 241.61174557621155, 32: 275.44875682424856},
        43: {1: 30.543451450002177, 2: 49.056875001997625, 4: 83.94989061005506, 8: 140.64175152714552, 12: 174.53709827634472, 16: 190.24620226146322, 24: 230.86232294712153, 32: 219.15744102788187},
        44: {1: 29.674859656003257, 2: 50.59486223462073, 4: 81.63290543198572, 8: 144.73153410503406, 12: 182.13126967684087, 16: 165.29127020243155, 24: 244.4718496671228, 32: 214.20434693994977},
        45: {1: 29.393141005662045, 2: 51.053910284413284, 4: 83.90276065611513, 8: 145.2124701540269, 12: 157.5282029141214, 16: 177.43552590349276, 24: 263.0335244677357, 32: 249.47183639273615},
        46: {1: 29.738441813093782, 2: 49.30845986078954, 4: 80.38447970077249, 8: 147.85293771068996, 12: 169.77026167368274, 16: 163.71675555540003, 24: 262.0951436727437, 32: 268.03200971147686},
        47: {1: 28.402563682132083, 2: 49.57021264303254, 4: 80.53643749396832, 8: 136.5591726158986, 12: 144.2298496640802, 16: 146.35143636033848, 24: 230.0763060939157, 32: 244.59186480133465},
        48: {1: 28.750764178083582, 2: 50.9006054285103, 4: 76.24137640098401, 8: 129.14655938694617, 12: 161.57171565572008, 16: 154.69832601881785, 24: 213.91659123737725, 32: 265.01209054718976},
        49: {1: 27.66765288281732, 2: 50.52069527937249, 4: 81.5316992031655, 8: 152.05933690282131, 12: 155.87845210564385, 16: 167.5715076805928, 24: 249.91676584778673, 32: 215.49837742922824},
        50: {1: 29.54319674544053, 2: 50.67623873221707, 4: 79.33749441907186, 8: 158.0226980541035, 12: 169.16514384825038, 16: 179.94185402537005, 24: 239.44258096134521, 32: 214.3631422802926},
        51: {1: 30.861036259429643, 2: 49.03592399855705, 4: 81.67726501652075, 8: 136.49668292060687, 12: 184.0765294441204, 16: 194.50625488388224, 24: 271.7108205871183, 32: 221.62612481300408},
        52: {1: 31.894184544012617, 2: 52.100630709216034, 4: 86.408482768272, 8: 118.38127274064635, 12: 178.3519535149173, 16: 200.63751129635693, 24: 231.88334931632184, 32: 257.8551724985854},
        53: {1: 31.764485757144843, 2: 50.22059458996608, 4: 85.67658487346614, 8: 128.60235544849718, 12: 179.2706712855062, 16: 227.80731149286646, 24: 240.36434211803385, 32: 214.87080080203998},
        54: {1: 30.968819681643637, 2: 52.493378227226415, 4: 81.82863470406488, 8: 124.66405573732115, 12: 171.9710936561885, 16: 180.02329730554322, 24: 239.64970491373452, 32: 207.81927884378288},
        55: {1: 30.96209735154714, 2: 51.60167808191693, 4: 83.75650693853913, 8: 118.26362687704913, 12: 177.80277376732823, 16: 160.98773898985772, 24: 250.2043527860618, 32: 211.5097596359123},
        56: {1: 31.155557074699797, 2: 50.09976949753933, 4: 80.38537800283008, 8: 110.71397106836952, 12: 199.38519735315637, 16: 165.68554394693368, 24: 261.1915810282331, 32: 216.61521649690553},
        57: {1: 30.48251403173576, 2: 51.674832690456974, 4: 84.20335079474003, 8: 107.93032704442086, 12: 174.31209752550285, 16: 171.28289698118405, 24: 234.48939108825115, 32: 263.76913693221206},
        58: {1: 31.55028322158738, 2: 52.985682928164536, 4: 84.60479819469202, 8: 116.98843224305904, 12: 164.19716789741855, 16: 154.98934413245692, 24: 244.36067666210027, 32: 240.34876524598974},
        59: {1: 32.30330273627753, 2: 53.1012315292685, 4: 88.46114939492857, 8: 131.20979199305992, 12: 195.31815321465498, 16: 156.95730717443396, 24: 260.07993203542514, 32: 235.34517033311533},
        60: {1: 31.035135522429616, 2: 52.338165158648536, 4: 90.66886825837658, 8: 135.4122871389892, 12: 178.89127472871323, 16: 148.7182649738108, 24: 239.55801097173622, 32: 262.41144718288865},
        61: {1: 31.251891254246758, 2: 51.91268838076837, 4: 92.18777620096245, 8: 134.01973629604254, 12: 205.59107447112171, 16: 174.295882761523, 24: 249.62979400240326, 32: 252.94884155369522},
        62: {1: 31.034023807209184, 2: 51.62533206320498, 4: 89.2512238782119, 8: 134.13983846205954, 12: 202.0007011236225, 16: 153.81810139064177, 24: 229.66296698117642, 32: 267.1082051387041},
        63: {1: 30.7405620812191, 2: 51.70962135305278, 4: 91.5162005269271, 8: 133.66323189513707, 12: 177.3660039630012, 16: 191.21408557703853, 24: 279.5871045595008, 32: 272.7846900853121},
        64: {1: 29.52374421529409, 2: 50.8141399053123, 4: 92.83829446413827, 8: 134.0034753287453, 12: 184.26101258720675, 16: 180.14922539791013, 24: 217.96217399795935, 32: 255.11568466070423},
        65: {1: 29.22136891694146, 2: 51.07088894724926, 4: 90.62664618215422, 8: 138.44600508395817, 12: 195.86662849260466, 16: 189.43410474843057, 24: 227.64153853610398, 32: 298.6036789920115},
        66: {1: 30.149685005527935, 2: 49.73102470242668, 4: 92.97980488186926, 8: 141.41153480746928, 12: 185.84480368252258, 16: 225.94721487825652, 24: 256.60004519555883, 32: 271.0572206501929},
        67: {1: 30.758767861555736, 2: 52.17997577748824, 4: 92.42691265300581, 8: 145.27631678953492, 12: 192.4686122088241, 16: 218.37520206520392, 24: 232.74710005904453, 32: 306.11681466432407},
        68: {1: 29.924563087642486, 2: 50.83850012013887, 4: 89.4035359454198, 8: 143.1293331104277, 12: 205.27606156258764, 16: 175.98047512842288, 24: 270.7865836649352, 32: 270.7588284746718},
        69: {1: 30.946991086429914, 2: 51.32625416951537, 4: 89.7020693751258, 8: 135.0131348984232, 12: 207.75376601325144, 16: 240.9794887561138, 24: 254.84150390948355, 32: 295.73842377021197},
        70: {1: 30.662512176776012, 2: 53.482823483116476, 4: 92.08515032802711, 8: 128.81436659574396, 12: 195.33704059500315, 16: 219.64828739842673, 24: 239.15995029378595, 32: 277.3046396555061},
        71: {1: 30.714101479900805, 2: 53.079324894938004, 4: 94.9831160013523, 8: 130.31257436046243, 12: 204.32754203606004, 16: 189.1086353131271, 24: 296.40787421089635, 32: 268.2043149894316},
        72: {1: 30.284376110994145, 2: 52.71933887575558, 4: 92.64848929090176, 8: 135.2592014792898, 12: 185.09343458455263, 16: 190.5177780981218, 24: 250.528105273078, 32: 305.64978827743164},
        73: {1: 28.940519667367223, 2: 52.542127132205884, 4: 91.61849952904383, 8: 143.61002534240438, 12: 195.28024169839574, 16: 198.54215386642795, 24: 288.16212636873905, 32: 277.5503542933074},
        74: {1: 30.113152033588552, 2: 52.22175039693367, 4: 93.41833785632656, 8: 138.28898369859428, 12: 170.33572772848575, 16: 167.98331419128453, 24: 270.6284929599736, 32: 289.7499922268551},
        75: {1: 31.45318601477171, 2: 53.10807194168568, 4: 92.08684954560357, 8: 135.79713810841824, 12: 188.90776989823704, 16: 170.20924777512909, 24: 265.30170357569534, 32: 273.70385973613276},
        76: {1: 33.49405426478241, 2: 51.49974889820734, 4: 92.97274174327003, 8: 128.30681216926098, 12: 189.71855580456779, 16: 187.08702048940182, 24: 255.65049746375152, 32: 303.18054202633033},
        77: {1: 35.5502820452136, 2: 54.39281487605567, 4: 89.91596608387233, 8: 130.76621407778927, 12: 204.163556718111, 16: 230.53964105532316, 24: 271.9074226678959, 32: 298.9079722861932},
        78: {1: 35.213484184846, 2: 55.173876458025646, 4: 93.79393498773126, 8: 128.182077073902, 12: 179.6532293796778, 16: 211.93543125265592, 24: 297.4483092269991, 32: 268.37249671791164},
        79: {1: 39.329196262352674, 2: 60.575863695143255, 4: 114.15759121721187, 8: 175.8578579818739, 12: 304.19059801375954, 16: 273.0274959583255, 24: 398.1095028025857, 32: 405.8776288023541},
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
        allocation_pairs = epoch_info.get('allocation_pairs', set())

        # Format allocation string with (num_nodes, num_replicas) pairs if available
        if allocation_pairs:
            # Convert set to sorted list for consistent display
            pairs_list = sorted(list(allocation_pairs))
            alloc_str = str(pairs_list) if len(pairs_list) > 1 else str(pairs_list[0])
        else:
            # Fallback to GPU counts only
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
                total_theoretical_time += 50  # 120 seconds rescaling overhead
            elif application == "deepspeech2":
                total_theoretical_time += 87  # 150 seconds rescaling overhead
            elif application == "bert":
                total_theoretical_time += 380  # 300 seconds rescaling overhead
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

def analyze_idle_waste_decomposition(log_file_path, first_job_time, last_job_arrival_time):
    """
    Analyze and decompose idle waste into categories:
    - scheduler_waste: scheduler nodes that are not being used
    - starting_waste: time from node appearance until first use
    - middle_waste: time between usage periods (for nodes used multiple times)
    - ending_waste: time from when node becomes unused until disappearance

    Also tracks nodes with multiple usage periods and their entry timestamps.
    Calculates average waste metrics using experiment_duration = (last_job_arrival_time - first_job_time).

    Raises error if:
    - cluster_nodes doesn't have ready_node_names field
    """
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Track node lifecycle: {node_name: [(timestamp, is_used), ...]}
    node_timeline = {}
    scheduler_nodes = set()

    # First pass: build node timeline
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']

        # Check for ready_node_names
        cluster_nodes = log_entry.get('cluster_nodes', {})
        if 'ready_node_names' not in cluster_nodes:
            raise ValueError(f"Line {i+1}: cluster_nodes missing 'ready_node_names' field. "
                           "This function requires logs with ready_node_names.")

        ready_node_names = cluster_nodes['ready_node_names']

        # Capture scheduler nodes from first entry
        if i == 0:
            scheduler_nodes = set(ready_node_names)

        # Get all allocated nodes
        allocated_nodes = set()
        for job in log_entry.get('submitted_jobs', []):
            allocation = job.get('allocation', [])
            if allocation:
                allocated_nodes.update(allocation)

        # Track each ready node's status
        for node in ready_node_names:
            if node not in node_timeline:
                node_timeline[node] = []
            is_used = node in allocated_nodes
            node_timeline[node].append((timestamp, is_used))

    # Second pass: analyze each node and decompose waste
    scheduler_waste_hours = 0
    starting_waste_hours = 0
    middle_waste_hours = 0
    ending_waste_hours = 0
    nodes_with_multiple_periods = {}  # {node_name: [entry_timestamps]}
    starting_waste_durations = []  # List of starting waste durations in seconds
    ending_waste_durations = []  # List of ending waste durations in seconds

    for node, timeline in node_timeline.items():
        is_scheduler = node in scheduler_nodes

        # Check usage pattern and detect multiple usage periods
        usage_periods = []
        in_use = False
        use_start = None

        for timestamp, used in timeline:
            if used and not in_use:
                # Start of usage period
                use_start = timestamp
                in_use = True
            elif not used and in_use:
                # End of usage period
                usage_periods.append((use_start, timestamp))
                in_use = False
                use_start = None

        # If still in use at the end
        if in_use:
            usage_periods.append((use_start, timeline[-1][0]))

        # Track nodes with multiple usage periods
        if len(usage_periods) > 1:
            entry_timestamps = [period[0] for period in usage_periods]
            nodes_with_multiple_periods[node] = entry_timestamps

        # Calculate waste hours
        if is_scheduler:
            # Scheduler node - all idle time is scheduler waste
            for j in range(len(timeline) - 1):
                timestamp, is_used = timeline[j]
                next_timestamp = timeline[j+1][0]
                if not is_used:
                    time_diff = next_timestamp - timestamp
                    scheduler_waste_hours += NUM_GPU_PER_NODE * time_diff / 3600
        else:
            # Worker node - calculate starting, middle, and ending waste
            if len(usage_periods) == 0:
                # Node never used - all time is starting waste
                node_start_time = timeline[0][0]
                node_end_time = timeline[-1][0]
                starting_duration = node_end_time - node_start_time
                starting_waste_durations.append(starting_duration)

                for j in range(len(timeline) - 1):
                    time_diff = timeline[j+1][0] - timeline[j][0]
                    starting_waste_hours += NUM_GPU_PER_NODE * time_diff / 3600
            else:
                # Node used one or more times
                first_use_start = usage_periods[0][0]
                last_use_end = usage_periods[-1][1]
                node_start_time = timeline[0][0]
                node_end_time = timeline[-1][0]

                # Track starting waste duration (appearance to first use)
                starting_duration = first_use_start - node_start_time
                if starting_duration > 0:
                    starting_waste_durations.append(starting_duration)

                # Track ending waste duration (last use to disappearance)
                ending_duration = node_end_time - last_use_end
                if ending_duration > 0:
                    ending_waste_durations.append(ending_duration)

                # Starting waste: from appearance to first use
                for j in range(len(timeline) - 1):
                    timestamp = timeline[j][0]
                    next_timestamp = timeline[j+1][0]

                    if timestamp < first_use_start:
                        # This period is before first use
                        if next_timestamp <= first_use_start:
                            # Entire period is starting waste
                            time_diff = next_timestamp - timestamp
                        else:
                            # Partial period (up to first use)
                            time_diff = first_use_start - timestamp
                        starting_waste_hours += NUM_GPU_PER_NODE * time_diff / 3600

                # Middle waste: gaps between usage periods (if multiple periods)
                if len(usage_periods) > 1:
                    for k in range(len(usage_periods) - 1):
                        gap_start = usage_periods[k][1]  # End of period k
                        gap_end = usage_periods[k+1][0]  # Start of period k+1

                        for j in range(len(timeline) - 1):
                            timestamp = timeline[j][0]
                            next_timestamp = timeline[j+1][0]

                            # Check if this time interval overlaps with the gap
                            if timestamp >= gap_start and timestamp < gap_end:
                                if next_timestamp <= gap_end:
                                    time_diff = next_timestamp - timestamp
                                else:
                                    time_diff = gap_end - timestamp
                                middle_waste_hours += NUM_GPU_PER_NODE * time_diff / 3600

                # Ending waste: from last use to disappearance
                for j in range(len(timeline) - 1):
                    timestamp = timeline[j][0]
                    next_timestamp = timeline[j+1][0]

                    if timestamp >= last_use_end:
                        # This period is after last use
                        time_diff = next_timestamp - timestamp
                        ending_waste_hours += NUM_GPU_PER_NODE * time_diff / 3600

    # Calculate experiment duration and averages
    experiment_duration_hours = (last_job_arrival_time - first_job_time) / 3600 if (last_job_arrival_time and first_job_time) else 0

    scheduler_waste_avg = scheduler_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    starting_waste_avg = starting_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    middle_waste_avg = middle_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    ending_waste_avg = ending_waste_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    total_waste = scheduler_waste_hours + starting_waste_hours + middle_waste_hours + ending_waste_hours
    total_waste_avg = total_waste / experiment_duration_hours if experiment_duration_hours > 0 else 0

    # Print results
    print("\n" + "="*60)
    print("IDLE WASTE DECOMPOSITION ANALYSIS")
    print("="*60)
    print(f"Experiment Duration: {experiment_duration_hours:.2f} hours")
    # print(f"\nGPU-Hours:")
    # print(f"  Scheduler Waste: {scheduler_waste_hours:.2f}")
    # print(f"  Starting Waste: {starting_waste_hours:.2f}")
    # print(f"  Middle Waste: {middle_waste_hours:.2f}")
    # print(f"  Ending Waste: {ending_waste_hours:.2f}")
    # print(f"  Total Idle Waste: {total_waste:.2f}")
    print(f"\nAverage GPUs:")
    print(f"  Scheduler Waste Average: {scheduler_waste_avg:.2f} GPUs")
    print(f"  Starting Waste Average: {starting_waste_avg:.2f} GPUs")
    print(f"  Middle Waste Average: {middle_waste_avg:.2f} GPUs")
    print(f"  Ending Waste Average: {ending_waste_avg:.2f} GPUs")
    print(f"  Total Idle Waste Average: {total_waste_avg:.2f} GPUs")
    print("="*60)

    # Print nodes with multiple usage periods
    if nodes_with_multiple_periods:
        print("\nNodes with Multiple Usage Periods:")
        print("-" * 60)
        for node_name, entry_times in nodes_with_multiple_periods.items():
            print(f"  {node_name}:")
            print(f"    Number of usage periods: {len(entry_times)}")
            print(f"    Entry timestamps: {entry_times}")
        print("="*60)

    # Plot histograms for starting and ending waste durations
    if starting_waste_durations or ending_waste_durations:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Starting waste histogram
        if starting_waste_durations:
            starting_waste_seconds = starting_waste_durations
            axes[0].hist(starting_waste_seconds, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Duration (seconds)', fontsize=12)
            axes[0].set_ylabel('Number of Nodes', fontsize=12)
            axes[0].set_title('Starting Waste Period Length\n(Node Appearance to First Use)', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')

            # Add statistics
            mean_start = np.mean(starting_waste_seconds)
            median_start = np.median(starting_waste_seconds)
            axes[0].axvline(mean_start, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_start:.1f}s')
            axes[0].axvline(median_start, color='green', linestyle='--', linewidth=2, label=f'Median: {median_start:.1f}s')
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'No starting waste data', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Starting Waste Period Length', fontsize=13, fontweight='bold')

        # Ending waste histogram
        if ending_waste_durations:
            ending_waste_seconds = ending_waste_durations
            axes[1].hist(ending_waste_seconds, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Duration (seconds)', fontsize=12)
            axes[1].set_ylabel('Number of Nodes', fontsize=12)
            axes[1].set_title('Ending Waste Period Length\n(Last Use to Disappearance)', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')

            # Add statistics
            mean_end = np.mean(ending_waste_seconds)
            median_end = np.median(ending_waste_seconds)
            axes[1].axvline(mean_end, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_end:.1f}s')
            axes[1].axvline(median_end, color='green', linestyle='--', linewidth=2, label=f'Median: {median_end:.1f}s')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No ending waste data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Ending Waste Period Length', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.show()

    return {
        'scheduler_waste_hours': scheduler_waste_hours,
        'starting_waste_hours': starting_waste_hours,
        'middle_waste_hours': middle_waste_hours,
        'ending_waste_hours': ending_waste_hours,
        'total_idle_waste_hours': total_waste,
        'scheduler_waste_avg': scheduler_waste_avg,
        'starting_waste_avg': starting_waste_avg,
        'middle_waste_avg': middle_waste_avg,
        'ending_waste_avg': ending_waste_avg,
        'total_idle_waste_avg': total_waste_avg,
        'experiment_duration_hours': experiment_duration_hours,
        'nodes_with_multiple_periods': nodes_with_multiple_periods,
        'starting_waste_durations': starting_waste_durations,
        'ending_waste_durations': ending_waste_durations
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python manage_monitor_log.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    print(f"Processing log file: {log_file_path}")

    jobs, total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours,\
    wasted_capacity_hours, scheduler_waste_hours, last_job_arrival_time, completed_jobs_status, \
    decreased_progress_issues, job_drop_sums, job_max_progress, job_gpu_hours, nodes_in_use_dict, scheduler_nodes = process_log_file(log_file_path)
    
    # Find first job time for metrics calculation
    first_job_time = min(job_info['first_seen'] for job_info in jobs.values()) if jobs else 0
    
    # Print response time and wasted time for all jobs
    print_all_jobs_summary(jobs)
    # Print per-job GPU-hours
    # print_job_gpu_hours(job_gpu_hours)

    # Print jobs that completed with failing pod status
    print_failed_completed_jobs(completed_jobs_status)

    # Hardcoded specific job breakdown
    specific_job_breakdown = 'cifar10-63'
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
    # plot_filename = base_name + '_response_time_comparison.png'
    # plot_response_time_comparison(jobs, plot_filename)
    # Also save stacked response time plot
    # stacked_plot_filename = base_name + '_response_time_stacked.png'
    # plot_job_response_time_stacked(jobs, stacked_plot_filename)
    
    metrics = calculate_metrics(total_gpu_hours, effective_gpu_hours, fragmentation_waste_hours, ready_unused_waste_hours, wasted_capacity_hours, scheduler_waste_hours, last_job_arrival_time, first_job_time)
    print_summary(metrics)

    # Calculate and print time-average used number of nodes (with grace period)
    Grace_period_length = 60
    if nodes_in_use_dict and last_job_arrival_time:
        # First, process the dict to apply grace period for nodes
        # Track when each node last became inactive
        timestamps = sorted(nodes_in_use_dict.keys())

        # For each timestamp, track which nodes are actually in use
        # We need to identify individual nodes from the allocations
        # Since nodes_in_use_dict only stores counts, we need to go back to the log
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        # Build a dict of timestamp -> set of active nodes (with grace period)
        active_nodes_with_grace = {}
        node_last_active = {}  # node_id -> last timestamp it was actively used

        for i, line in enumerate(lines):

            log_entry = json.loads(line.strip())
            timestamp = log_entry['timestamp']

            # Get nodes currently in use
            all_alloc_items = []
            for job in log_entry['submitted_jobs']:
                allocation = job.get('allocation', [])
                if allocation:
                    all_alloc_items.extend(allocation)

            currently_used_nodes = set(all_alloc_items)

            # Update last active time for currently used nodes
            for node in currently_used_nodes:
                node_last_active[node] = timestamp

            # Determine which nodes are active with grace period
            # A node is active if: currently used OR last used within 30 seconds
            active_with_grace = set(currently_used_nodes)
            for node, last_time in node_last_active.items():
                if timestamp - last_time <= Grace_period_length:  # 265-second grace period
                    active_with_grace.add(node)
            
            for node in scheduler_nodes:
                active_with_grace.add(node)

            active_nodes_with_grace[timestamp] = len(active_with_grace)
            assert len(active_with_grace) >= len(currently_used_nodes), f"Active nodes with grace {active_with_grace} is greater than currently used nodes {currently_used_nodes}"

        # Now calculate time-weighted average
        total_node_time = 0
        grace_timestamps = sorted(active_nodes_with_grace.keys())

        for i in range(len(grace_timestamps) - 1):
            current_timestamp = grace_timestamps[i]
            next_timestamp = grace_timestamps[i + 1]
            time_diff = next_timestamp - current_timestamp
            nodes_used = active_nodes_with_grace[current_timestamp]
            total_node_time += nodes_used * time_diff

        # Calculate average: total_node_time / (last_job_arrival_time - first_job_time)
        experiment_duration = last_job_arrival_time - first_job_time
        time_average_nodes = total_node_time / experiment_duration if experiment_duration > 0 else 0

        print("="*60)
        print(f"Time-Average Used Number of GPUs (with grace): {NUM_GPU_PER_NODE * time_average_nodes:.2f}")
        print("="*60)

    # Finally, print mean job total response time (placed at the very end)
    mean_rt = compute_mean_job_response_time(jobs)
    print("\n" + "="*60)
    print(f"Mean Job Response Time (s): {mean_rt:.1f}")
    print("="*60)

    # At the very end, warn about any jobs with decreasing progress
    print_decreasing_progress_warnings(decreased_progress_issues, job_drop_sums, job_max_progress)

    # analyze_idle_waste_decomposition(log_file_path, first_job_time, last_job_arrival_time)

    # print("response_dict={")
    # for job_name, job_info in jobs.items():
    #     if job_info['epochs']:
    #         last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
    #         actual_response_time = last_epoch_end - job_info['first_seen']
        
    #     print(f"    '{job_name}': {actual_response_time:.1f},")

    # print("}")

if __name__ == "__main__":
    main()
