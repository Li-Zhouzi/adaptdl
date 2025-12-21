#!/usr/bin/env python3
"""
Health check script for AdaptDL experiments.
Monitors job status every 5 minutes and handles failures by:
1. Collecting logs from failing jobs, scheduler pods, nodes, and pods
2. Scaling down the autoscaling group to 0
3. Terminating run_workload and run_monitor processes

Additionally, fetches allocator logs every 1 hour for monitoring purposes.
"""

import subprocess
import time
import os
import sys
import json
from datetime import datetime
import threading

# Global shutdown event used to coordinate graceful termination across threads
shutdown_event = threading.Event()

# Configuration
CHECK_INTERVAL = 300  # 5 minutes in seconds
ALLOCATOR_LOG_INTERVAL = 3600  # 1 hour in seconds
COMPLETION_CHECK_INTERVAL = 600  # 10 minutes in seconds
EXP_DIR = "./experiment_results/1220-FW-b48-2"
MONITOR_LOG_PATH = os.path.join(EXP_DIR, "monitor_log.txt")
ERROR_LOG_DIR = os.path.join(EXP_DIR, "errors")
PERIODIC_LOG_DIR = os.path.join(EXP_DIR, "periodic_logs")
AUTOSCALING_GROUP_NAME = "eks-12xlarge-cbd-1007-c8ccdfb1-0ed8-acd8-ee69-7bdc2245b084"
SCHEDULER_NAMESPACE = "adaptdl"
SCHEDULER_POD_PREFIX = "adaptdl-sched"


def check_job_health():
    """Check if any AdaptDL jobs are in Failed state."""
    print(f"[{datetime.now()}] Checking job health...")
    result = subprocess.run(
        "kubectl get adaptdljobs",
        shell=True,
        capture_output=True,
        text=True
    )

    stdout = result.stdout
    print(stdout)

    # Parse output to find failed jobs
    lines = stdout.strip().split('\n')
    if len(lines) <= 1:
        return False, None

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 1:
            job_name = parts[0]
            if "Failed" in line or "Error" in line:
                print(f"[ALERT] Found failing job: {job_name}")
                return True, job_name

    return False, None


def check_scheduler_restarts():
    """Check if scheduler pod has restarted more than 10 times."""
    print(f"[{datetime.now()}] Checking scheduler restarts...")
    result = subprocess.run(
        f"kubectl get pods -n {SCHEDULER_NAMESPACE}",
        shell=True,
        capture_output=True,
        text=True
    )

    stdout = result.stdout
    print(stdout)

    # Parse output to find scheduler pod restarts
    lines = stdout.strip().split('\n')
    if len(lines) <= 1:
        return False

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 4 and SCHEDULER_POD_PREFIX in parts[0]:
            pod_name = parts[0]
            restarts = parts[3]  # RESTARTS column
            try:
                restart_count = int(restarts)
                print(f"[INFO] Scheduler pod {pod_name} has {restart_count} restarts")
                if restart_count > 10:
                    print(f"[ALERT] Scheduler pod has excessive restarts: {restart_count}")
                    return True
            except ValueError:
                # If RESTARTS column is not a number, skip
                pass

    return False


def get_failing_job_logs(job_name):
    """Get logs from the failing job's pods."""
    print(f"[INFO] Fetching logs for failing job {job_name}...")

    os.makedirs(ERROR_LOG_DIR, exist_ok=True)

    # Get all pods in the namespace
    result = subprocess.run(
        f"kubectl get pods -n {SCHEDULER_NAMESPACE} -o wide",
        shell=True,
        capture_output=True,
        text=True
    )

    # Find pods that belong to the failing job
    lines = result.stdout.strip().split('\n')
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 1:
            pod_name = parts[0]
            if job_name in pod_name:
                print(f"[INFO] Found pod: {pod_name}")

                # Get pod logs
                log_file = os.path.join(ERROR_LOG_DIR, f"{job_name}_error.txt")
                result = subprocess.run(
                    f"kubectl logs -n {SCHEDULER_NAMESPACE} {pod_name} --all-containers=true --tail=10000",
                    shell=True,
                    capture_output=True,
                    text=True
                )

                with open(log_file, 'w') as f:
                    f.write(f"=== Logs for pod {pod_name} ===\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Job: {job_name}\n\n")
                    f.write(result.stdout)

                print(f"[INFO] Saved job logs to {log_file}")


def get_scheduler_logs():
    """Get logs from scheduler pod containers (allocator, width-calculator, sched-pod)."""
    print(f"[INFO] Fetching scheduler logs...")

    os.makedirs(ERROR_LOG_DIR, exist_ok=True)

    # Find scheduler pod
    result = subprocess.run(
        f"kubectl get pods -n {SCHEDULER_NAMESPACE} -o wide",
        shell=True,
        capture_output=True,
        text=True
    )

    lines = result.stdout.strip().split('\n')
    scheduler_pod = None
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 1 and SCHEDULER_POD_PREFIX in parts[0]:
            scheduler_pod = parts[0]
            break

    if not scheduler_pod:
        print(f"[WARNING] Could not find scheduler pod")
        return

    print(f"[INFO] Found scheduler pod: {scheduler_pod}")

    # Get logs for each container
    containers = ["allocator", "width-calculator", "supervisor"]
    for container in containers:
        log_file = os.path.join(ERROR_LOG_DIR, f"{container}.txt")
        print(f"[INFO] Fetching logs for container: {container}")

        result = subprocess.run(
            f"kubectl logs -n {SCHEDULER_NAMESPACE} {scheduler_pod} -c {container}",
            shell=True,
            capture_output=True,
            text=True
        )

        with open(log_file, 'w') as f:
            f.write(f"=== Logs for scheduler container {container} ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Pod: {scheduler_pod}\n\n")
            f.write(result.stdout)

        print(f"[INFO] Saved {container} logs to {log_file}")

        log_file = os.path.join(ERROR_LOG_DIR, f"{container}_previous.txt")
        print(f"[INFO] Fetching previous logs for container: {container}")
        result = subprocess.run(
            f"kubectl logs -n {SCHEDULER_NAMESPACE} {scheduler_pod} -c {container} --previous",
            shell=True,
            capture_output=True,
            text=True
        )
        with open(log_file, 'a') as f:
            f.write(f"=== Logs for scheduler container {container} (previous) ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(result.stdout)


def get_nodes_info():
    """Save current node status."""
    print(f"[INFO] Fetching nodes information...")

    os.makedirs(ERROR_LOG_DIR, exist_ok=True)

    result = subprocess.run(
        "kubectl get nodes -o wide",
        shell=True,
        capture_output=True,
        text=True
    )

    log_file = os.path.join(ERROR_LOG_DIR, "nodes.txt")
    with open(log_file, 'w') as f:
        f.write(f"=== Cluster Nodes Status ===\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(result.stdout)

    print(f"[INFO] Saved nodes info to {log_file}")


def get_pods_info():
    """Save current pod status."""
    print(f"[INFO] Fetching pods information...")

    os.makedirs(ERROR_LOG_DIR, exist_ok=True)

    result = subprocess.run(
        "kubectl get pods -A -o wide",
        shell=True,
        capture_output=True,
        text=True
    )

    log_file = os.path.join(ERROR_LOG_DIR, "pods.txt")
    with open(log_file, 'w') as f:
        f.write(f"=== All Pods Status ===\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(result.stdout)

    print(f"[INFO] Saved pods info to {log_file}")


def filter_supervisor_logs_for_job(job_name):
    """Filter supervisor logs to show only entries related to the failing job."""
    print(f"[INFO] Filtering supervisor logs for job {job_name}...")

    supervisor_log_file = os.path.join(ERROR_LOG_DIR, "supervisor.txt")

    # Check if supervisor log exists
    if not os.path.exists(supervisor_log_file):
        print(f"[WARNING] Supervisor log file not found: {supervisor_log_file}")
        return

    try:
        with open(supervisor_log_file, 'r') as f:
            all_lines = f.readlines()

        # Filter for job-specific and DISCOVER logs
        job_lines = [line for line in all_lines
                     if job_name in line or "[DISCOVER" in line]

        if job_lines:
            filtered_log_file = os.path.join(ERROR_LOG_DIR, f"supervisor_{job_name}.txt")
            with open(filtered_log_file, 'w') as f:
                f.write(f"=== Filtered Supervisor Logs for {job_name} ===\n")
                f.write(f"Timestamp: {datetime.now()}\n\n")
                f.writelines(job_lines)

            print(f"[INFO] Saved filtered supervisor logs to {filtered_log_file}")
            print(f"[INFO] Filtered {len(job_lines)} lines related to {job_name}")
        else:
            print(f"[INFO] No supervisor logs found for job {job_name}")

    except Exception as e:
        print(f"[WARNING] Error filtering supervisor logs: {e}")


def scale_down_autoscaling_group():
    """Set autoscaling group desired capacity to 0."""
    print(f"[INFO] Scaling down autoscaling group: {AUTOSCALING_GROUP_NAME}")

    subprocess.run(
        f"aws autoscaling update-auto-scaling-group "
        f"--auto-scaling-group-name {AUTOSCALING_GROUP_NAME} "
        f"--desired-capacity 0",
        shell=True
    )

    print(f"[SUCCESS] Autoscaling group scaled down to 0")


def terminate_processes():
    """Terminate run_workload and run_monitor processes."""
    print(f"[INFO] Terminating run_workload and run_monitor processes...")

    subprocess.run("pkill -f run_workload", shell=True)
    subprocess.run("pkill -f run_monitor", shell=True)

    print(f"[SUCCESS] Sent termination signals to run_workload and run_monitor processes")


def fetch_allocator_log_periodically():
    """Periodically fetch allocator logs every hour and save to separate files."""
    os.makedirs(PERIODIC_LOG_DIR, exist_ok=True)

    iteration = 0
    while not shutdown_event.is_set():
        # Wait for interval or shutdown signal, whichever comes first
        if shutdown_event.wait(ALLOCATOR_LOG_INTERVAL):
            break
        iteration += 1

        print(f"\n[PERIODIC] Fetching allocator log (iteration {iteration})...")

        # Find scheduler pod
        result = subprocess.run(
            f"kubectl get pods -n {SCHEDULER_NAMESPACE} -o wide",
            shell=True,
            capture_output=True,
            text=True
        )

        lines = result.stdout.strip().split('\n')
        scheduler_pod = None
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 1 and SCHEDULER_POD_PREFIX in parts[0]:
                scheduler_pod = parts[0]
                break

        if not scheduler_pod:
            print(f"[WARNING] Could not find scheduler pod for periodic log fetch")
            continue

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(PERIODIC_LOG_DIR, f"allocator_{timestamp}.txt")

        result = subprocess.run(
            f"kubectl logs -n {SCHEDULER_NAMESPACE} {scheduler_pod} -c allocator --tail=50000",
            shell=True,
            capture_output=True,
            text=True
        )

        with open(log_file, 'w') as f:
            f.write(f"=== Periodic Allocator Log ===\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Pod: {scheduler_pod}\n")
            f.write(f"Iteration: {iteration}\n\n")
            f.write(result.stdout)

        print(f"[PERIODIC] Saved allocator log to {log_file}")


def check_monitor_log_empty():
    """
    Check if the last line of monitor_log.txt has empty submitted_jobs.
    Returns True if no jobs in last line, False otherwise.
    Reads only the last few KB to handle large log files efficiently.
    """
    try:
        # Read last 8KB of file (should contain several complete JSON lines)
        with open(MONITOR_LOG_PATH, 'rb') as f:
            # Seek to end of file
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            if file_size == 0:
                print(f"[WARNING] Monitor log file is empty")
                return False

            # Read last 8KB or entire file if smaller
            read_size = min(8192, file_size)
            f.seek(max(0, file_size - read_size))
            tail_bytes = f.read()

        # Decode and get last complete line
        tail_text = tail_bytes.decode('utf-8', errors='ignore')
        lines = tail_text.strip().split('\n')

        # Try to parse lines from the end until we find a valid JSON
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                submitted_jobs = record.get("submitted_jobs", [])

                # Check if submitted_jobs is empty
                is_empty = len(submitted_jobs) == 0
                print(f"[INFO] Monitor log check: {'no jobs' if is_empty else f'{len(submitted_jobs)} jobs'} in last record")
                return is_empty

            except json.JSONDecodeError:
                # This line might be incomplete, try previous line
                continue

        print(f"[WARNING] Could not find valid JSON in monitor log tail")
        return False

    except FileNotFoundError:
        print(f"[WARNING] Monitor log file not found: {MONITOR_LOG_PATH}")
        return False
    except Exception as e:
        print(f"[WARNING] Error reading monitor log: {e}")
        return False


def check_workload_running():
    """
    Check if run_workload process is still running.
    Returns True if running, False if not running.
    """
    result = subprocess.run(
        "pgrep -f run_workload",
        shell=True,
        capture_output=True,
        text=True
    )

    is_running = result.returncode == 0
    print(f"[INFO] run_workload process: {'running' if is_running else 'not running'}")
    return is_running


def check_experiment_completion():
    """
    Periodically check for experiment completion every 10 minutes.
    Experiment is considered complete when BOTH conditions are met for 2 consecutive checks:
    1. Monitor log's last line shows no jobs (submitted_jobs is empty)
    2. run_workload process has finished

    When experiment completes, terminate monitor process and scale down cluster.
    """
    print(f"[INFO] Started experiment completion checker (interval: {COMPLETION_CHECK_INTERVAL/60} minutes)")

    consecutive_completion_checks = 0
    required_consecutive_checks = 2

    while not shutdown_event.is_set():
        # Wait for interval or shutdown signal, whichever comes first
        if shutdown_event.wait(COMPLETION_CHECK_INTERVAL):
            break

        print(f"\n[COMPLETION CHECK] Running experiment completion check...")

        # Check both conditions
        log_empty = check_monitor_log_empty()
        workload_finished = not check_workload_running()

        if log_empty and workload_finished:
            consecutive_completion_checks += 1
            print(f"[COMPLETION CHECK] Both conditions met ({consecutive_completion_checks}/{required_consecutive_checks} checks)")

            if consecutive_completion_checks >= required_consecutive_checks:
                print(f"\n{'='*80}")
                print(f"[SUCCESS] EXPERIMENT COMPLETED")
                print(f"[SUCCESS] Conditions met for {required_consecutive_checks} consecutive checks:")
                print(f"[SUCCESS] - Monitor log shows no jobs")
                print(f"[SUCCESS] - run_workload process has finished")
                print(f"[SUCCESS] Starting cleanup procedure...")
                print(f"{'='*80}\n")

                # Terminate monitor process
                print(f"[INFO] Terminating run_monitor process...")
                subprocess.run("pkill -f run_monitor", shell=True)

                # Scale down cluster
                print(f"[INFO] Scaling down cluster...")
                scale_down_autoscaling_group()

                print(f"\n{'='*80}")
                print(f"[SUCCESS] Experiment completion cleanup finished")
                print(f"[INFO] Exiting health monitor")
                print(f"{'='*80}\n")

                # Signal shutdown to main loop and other threads, then return
                shutdown_event.set()
                return
        else:
            # Reset counter if conditions not met
            if consecutive_completion_checks > 0:
                print(f"[COMPLETION CHECK] Conditions not met, resetting counter (was {consecutive_completion_checks})")
            consecutive_completion_checks = 0
            print(f"[COMPLETION CHECK] Experiment still running (log_empty={log_empty}, workload_finished={workload_finished})")


def cleanup_on_failure(job_name=None):
    """Execute cleanup procedure when a job failure is detected."""
    print(f"\n{'='*80}")
    print(f"[ALERT] EXPERIMENT FAILURE DETECTED")
    if job_name:
        print(f"[ALERT] Failed Job: {job_name}")
    print(f"[ALERT] Starting cleanup procedure...")
    print(f"{'='*80}\n")

    # Step 1: Collect logs
    print("\n--- STEP 1: Collecting logs ---")
    if job_name:
        get_failing_job_logs(job_name)
    get_scheduler_logs()
    if job_name:
        filter_supervisor_logs_for_job(job_name)
    get_nodes_info()
    get_pods_info()

    # Step 2: Scale down autoscaling group
    print("\n--- STEP 2: Scaling down cluster ---")
    scale_down_autoscaling_group()

    # Step 3: Terminate processes
    print("\n--- STEP 3: Terminating processes ---")
    terminate_processes()

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Cleanup procedure completed")
    print(f"[INFO] Logs saved to: {ERROR_LOG_DIR}")
    print(f"{'='*80}\n")
    # Ensure other threads exit promptly
    shutdown_event.set()


def main():
    """Main monitoring loop."""
    print(f"{'='*80}")
    print(f"AdaptDL Experiment Health Monitor")
    print(f"Health check interval: {CHECK_INTERVAL} seconds ({CHECK_INTERVAL/60} minutes)")
    print(f"Allocator log interval: {ALLOCATOR_LOG_INTERVAL} seconds ({ALLOCATOR_LOG_INTERVAL/3600} hour)")
    print(f"Completion check interval: {COMPLETION_CHECK_INTERVAL} seconds ({COMPLETION_CHECK_INTERVAL/60} minutes)")
    print(f"{'='*80}\n")

    # Start periodic allocator log fetching in a background thread
    log_thread = threading.Thread(target=fetch_allocator_log_periodically, daemon=True)
    log_thread.start()
    print(f"[INFO] Started periodic allocator log fetching thread\n")

    # Start experiment completion checker in a background thread
    completion_thread = threading.Thread(target=check_experiment_completion, daemon=True)
    completion_thread.start()
    print(f"[INFO] Started experiment completion checker thread\n")

    while not shutdown_event.is_set():
        has_failure, job_name = check_job_health()

        if has_failure and job_name:
            cleanup_on_failure(job_name)
            print("[INFO] Exiting health monitor after cleanup")
            sys.exit(1)

        # Check scheduler restarts
        excessive_restarts = check_scheduler_restarts()
        if excessive_restarts:
            cleanup_on_failure()
            print("[INFO] Exiting health monitor after cleanup")
            sys.exit(1)

        print(f"[{datetime.now()}] All jobs healthy. Next check in {CHECK_INTERVAL} seconds.\n")

        # Wait for interval or shutdown signal
        if shutdown_event.wait(CHECK_INTERVAL):
            break

    print("[INFO] Shutdown signal received, exiting health monitor")
    sys.exit(0)


if __name__ == "__main__":
    main()
