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
from datetime import datetime
import threading


# Configuration
CHECK_INTERVAL = 300  # 5 minutes in seconds
ALLOCATOR_LOG_INTERVAL = 3600  # 1 hour in seconds
ERROR_LOG_DIR = "./experiment_results/1127-FW-b40/errors"
PERIODIC_LOG_DIR = "./experiment_results/periodic_logs"
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
    containers = ["allocator", "width-calculator"]
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
    while True:
        time.sleep(ALLOCATOR_LOG_INTERVAL)
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


def main():
    """Main monitoring loop."""
    print(f"{'='*80}")
    print(f"AdaptDL Experiment Health Monitor")
    print(f"Health check interval: {CHECK_INTERVAL} seconds ({CHECK_INTERVAL/60} minutes)")
    print(f"Allocator log interval: {ALLOCATOR_LOG_INTERVAL} seconds ({ALLOCATOR_LOG_INTERVAL/3600} hour)")
    print(f"{'='*80}\n")

    # Start periodic allocator log fetching in a background thread
    log_thread = threading.Thread(target=fetch_allocator_log_periodically, daemon=True)
    log_thread.start()
    print(f"[INFO] Started periodic allocator log fetching thread\n")

    while True:
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

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
