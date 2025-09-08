#!/usr/bin/env python3
import os
import subprocess
import time
import re
import signal
import sys
from datetime import datetime


"""Should make sure that 1. the schedulers are running 2. Docker login is done 3. make sure the workload-test3 consists of only one job called {JOB_TYPE}-0 4. experiment_results/dummy directory exists"""
"""If PARALLEL is True, make sure workload-test3 consists of len(NUM_GPU_LIST) jobs"""
# Configuration - list of GPU counts to test
NUM_GPU_LIST = [12, 16]
JOB_TYPE = "cifar10"  # Job type (e.g., "cifar10", "imagenet", "bert", etc.)
DIRECTORY_NAME = "dummy-goodput-12xlarge"  # Directory name under experiment_results/
PARALLEL = False  # If True, use the entire list as input to DummyPolicy; if False, run one by one

# Setup logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")

def log(message):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

def verify_and_update_policy(num_gpu):
    """Verify policy is 'dummy' and update the GPU count in allocator.py
    num_gpu can be either an int or a list of ints"""
    allocator_path = "./sched/adaptdl_sched/allocator.py"
    
    # Read the file
    with open(allocator_path, 'r') as f:
        content = f.read()
    
    # Verify policy is dummy
    policy_pattern = r'SELECTED_POLICY = "dummy"  # <--- CHANGE THIS VALUE TO SWITCH POLICY'
    if not re.search(policy_pattern, content):
        log("ERROR: Policy is not set to 'dummy'")
        sys.exit(1)
    
    log("✓ Verified policy is 'dummy'")
    
    # Update the GPU number - handle both int and list
    old_pattern = r'self\._policy = DummyPolicy\(num_gpus_per_job=(?:\d+|\[[\d, ]+\])\) # Configure dummy as needed'
    new_line = f'self._policy = DummyPolicy(num_gpus_per_job={num_gpu}) # Configure dummy as needed'
    
    content = re.sub(old_pattern, new_line, content)
    
    # Write back
    with open(allocator_path, 'w') as f:
        f.write(content)
    
    log(f"✓ Updated DummyPolicy to use {num_gpu} GPU(s)")

def run_command(cmd, shell=False, capture_output=True):
    """Run a command and return the result"""
    log(f"Running: {cmd}")
    
    if capture_output:
        # Capture output for logging
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        
        # Log stdout if present
        if result.stdout:
            log("Command output:")
            for line in result.stdout.strip().split('\n'):
                log(f"  {line}")
        
        # Log stderr and error info if command failed
        if result.returncode != 0:
            log(f"Error running command: {cmd}")
            if result.stderr:
                log("Error output:")
                for line in result.stderr.strip().split('\n'):
                    log(f"  {line}")
    else:
        # Run command with real-time output (for interactive commands like docker build)
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n--- Real-time output for: {cmd} ---\n")
            # Run command and show output in real-time
            process = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            # Read output line by line and display/log it
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')  # Print to console in real-time
                    log_file.write(line)  # Write to log file
                    log_file.flush()  # Ensure it's written immediately
            
            process.wait()
            result = subprocess.CompletedProcess(cmd, process.returncode, '', '')
            log_file.write(f"--- Command completed with return code: {process.returncode} ---\n")
            
            if process.returncode != 0:
                log(f"Command failed with return code: {process.returncode}")
    
    return result

def check_job_completion(log_file):
    """Check if job is completed by looking for completion time in the log file"""
    if not os.path.exists(log_file):
        return False
    
    try:
        # Read file from end to beginning for efficiency
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Check from tail to head
        for line in reversed(lines):
            # Look for completion time pattern
            if "completion_time" in line and "null" not in line:
                return True
    except:
        return False
    
    return False

def get_last_timestamp(log_file):
    """Get the last timestamp from the log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for timestamp pattern from the end
        for line in reversed(lines):
            # Look for timestamp which can be numeric or string
            if '"timestamp"' in line:
                # Try to extract numeric timestamp first
                match = re.search(r'"timestamp":\s*(\d+\.?\d*)', line)
                if match:
                    return float(match.group(1))
                # Fall back to string timestamp
                match = re.search(r'"timestamp":\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    except:
        return None
    
    return None

def check_if_job_exists(log_file):
    """Check if any job exists in the log file"""
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Check from tail to head for any submitted_jobs that are not empty
        for line in reversed(lines[-20:]):  # Check last 20 lines for efficiency
            if 'submitted_jobs' in line:
                # Check if submitted_jobs is not empty
                import json
                try:
                    entry = json.loads(line.strip())
                    if entry.get('submitted_jobs', []):
                        return True
                except:
                    pass
    except:
        return False
    
    return False

def run_single_experiment(num_gpu):
    """Run a single experiment with the specified number of GPUs"""
    log(f"\n{'='*60}")
    log(f"Starting experiment with NUM_GPU={num_gpu}")
    log(f"{'='*60}\n")
    
    # Step 1: Docker login to ECR
    log("\n✓ Logging into Docker ECR...")
    run_command([
        "aws", "ecr", "get-login-password", "--region", "us-east-1", "|", 
        "docker", "login", "--username", "AWS", "--password-stdin", 
        "399790253372.dkr.ecr.us-east-1.amazonaws.com"
    ], shell=True)
    
    # Step 2: Verify and update policy
    verify_and_update_policy(num_gpu)
    
    # Step 3: Scale up the cluster
    # log(f"\n✓ Scaling up cluster to {num_gpu} nodes...")
    # if num_gpu == 4 or num_gpu == 6:
    #     log(f"Skipping scaling up for {num_gpu} GPUs")
    # else:
    #     run_command([
    #         "aws", "autoscaling", "update-auto-scaling-group",
    #         "--auto-scaling-group-name", "eksctl-adaptdl-eks-cluster-nodegroup-ng-1-NodeGroup-Ld2yZvkxjom7",
    #         "--desired-capacity", str(num_gpu)
    #     ])
    
    # Step 4: Run helm update
    log("\n✓ Running helm update...")
    run_command(["./helm/update_adaptdl.sh"], shell=True, capture_output=False)
    time.sleep(60)  # Give it time to update
    
    # Step 5: Delete existing job
    log(f"\n✓ Deleting existing job {JOB_TYPE}-0...")
    run_command(["kubectl", "delete", "adaptdljob", "--all"])
    time.sleep(10)
    
    # Step 6: Run workload
    log("\n✓ Running workload...")
    run_command(["./benchmark/run_workload.sh"], shell=True, capture_output=False)
    time.sleep(1)  # Give it time to start
    
    # Step 7: Start monitor in subprocess
    if isinstance(num_gpu, list):
        # For PARALLEL mode, use a descriptive filename
        gpu_str = "_".join(map(str, num_gpu))
        log_file = f"./experiment_results/{DIRECTORY_NAME}/{JOB_TYPE}/parallel_{gpu_str}gpu.txt"
    else:
        log_file = f"./experiment_results/{DIRECTORY_NAME}/{JOB_TYPE}/{num_gpu}gpu.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log(f"\n✓ Starting monitor, output to: {log_file}")
    monitor_proc = subprocess.Popen(
        ["python", "benchmark/run_monitor.py", log_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Step 8: Check for completion every 60 seconds
    print("\n✓ Monitoring job completion...")
    last_timestamp = None
    stuck_count = 0
    no_job_count = 0
    
    try:
        while True:
            time.sleep(60)
            
            if check_job_completion(log_file):
                log("\n✅ Job completed!")
                
                # Stop monitor
                print("Stopping monitor...")
                monitor_proc.terminate()
                time.sleep(2)
                
                if monitor_proc.poll() is None:
                    monitor_proc.kill()
                
                print("Monitor stopped.")
                log(f"\n=== Experiment with {num_gpu} GPU(s) Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                break
            else:
                # Check if job exists in the log
                if not check_if_job_exists(log_file):
                    no_job_count += 1
                    log(f"Warning: No job found in log for {no_job_count} check(s)")
                    
                    if no_job_count >= 10:
                        log("\n❌ No job appeared in log for 10 minutes. Stopping experiment.")
                        
                        # Stop monitor
                        monitor_proc.terminate()
                        time.sleep(2)
                        if monitor_proc.poll() is None:
                            monitor_proc.kill()
                        
                        # Delete any stuck job
                        log(f"Deleting job {JOB_TYPE}-0 if exists...")
                        run_command(["kubectl", "delete", "adaptdljob", f"{JOB_TYPE}-0", "-n", "adaptdl", "--ignore-not-found=true"])
                        
                        # Mark this as a failed experiment
                        log(f"\n=== Experiment with {num_gpu} GPU(s) FAILED (no job) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                        return "no_job"  # Return status to indicate no job
                else:
                    # Job exists, reset no_job counter
                    no_job_count = 0
                    
                    # Check if timestamp is stuck
                    current_timestamp = get_last_timestamp(log_file)
                    
                    if current_timestamp:
                        if last_timestamp == current_timestamp:
                            stuck_count += 1
                            log(f"Warning: Timestamp hasn't changed for {stuck_count} check(s). Current timestamp: {current_timestamp}")
                            
                            if stuck_count >= 10:
                                log("\n❌ Job appears to be stuck (timestamp unchanged for 10 minutes). Stopping experiment.")
                                
                                # Stop monitor
                                monitor_proc.terminate()
                                time.sleep(2)
                                if monitor_proc.poll() is None:
                                    monitor_proc.kill()
                                
                                # Delete the stuck job
                                log(f"Deleting stuck job {JOB_TYPE}-0...")
                                run_command(["kubectl", "delete", "adaptdljob", f"{JOB_TYPE}-0", "-n", "adaptdl", "--ignore-not-found=true"])
                                
                                # Mark this as a failed experiment
                                log(f"\n=== Experiment with {num_gpu} GPU(s) FAILED (stuck) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                                return "stuck"  # Return status to indicate stuck job
                        else:
                            # Timestamp changed, reset counter
                            stuck_count = 0
                            last_timestamp = current_timestamp
                
                print(f"Job still running... (checked at {time.strftime('%Y-%m-%d %H:%M:%S')})")
                
    except KeyboardInterrupt:
        log("\n\nInterrupted by user during single experiment. Cleaning up monitor...")
        monitor_proc.terminate()
        time.sleep(2)
        if monitor_proc.poll() is None:
            monitor_proc.kill()
        raise  # Re-raise to be caught by main function
    
    return "completed"  # Return status to indicate successful completion

def main():
    # Create the experiment results directory
    experiment_dir = f"./experiment_results/{DIRECTORY_NAME}"
    os.makedirs(experiment_dir, exist_ok=True)
    log(f"Created/verified experiment directory: {experiment_dir}")
    
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"=== Experiment Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    if PARALLEL:
        log(f"PARALLEL mode enabled. Running single experiment with GPU list: {NUM_GPU_LIST}")
        log(f"Starting experiment for job type '{JOB_TYPE}'")
    else:
        log(f"PARALLEL mode disabled. Running sequential experiments.")
        log(f"Starting experiments for job type '{JOB_TYPE}' with GPU configurations: {NUM_GPU_LIST}")
    
    interrupted = False
    failed = False
    failure_reason = ""
    try:
        if PARALLEL:
            # Run single experiment with the entire list
            status = run_single_experiment(NUM_GPU_LIST)
            if status == "stuck":
                failed = True
                failure_reason = "stuck job"
                log("\n\n❌ Experiment failed due to stuck job.")
            elif status == "no_job":
                failed = True
                failure_reason = "no job appeared"
                log("\n\n❌ Experiment failed because no job appeared.")
            elif status == "completed":
                log("\n\n✅ Experiment completed successfully!")
        else:
            # Run experiments for each GPU configuration
            for num_gpu in NUM_GPU_LIST:
                status = run_single_experiment(num_gpu)
                if status == "stuck":
                    failed = True
                    failure_reason = "stuck job"
                    log("\n\n❌ Experiment failed due to stuck job. Stopping all experiments.")
                    break
                elif status == "no_job":
                    failed = True
                    failure_reason = "no job appeared"
                    log("\n\n❌ Experiment failed because no job appeared. Stopping all experiments.")
                    break
            
            if not failed:
                log("\n\n✅ All experiments completed successfully!")
        
    except KeyboardInterrupt:
        log("\n\nInterrupted by user. Exiting without scaling down...")
        interrupted = True
    
    finally:
        if not interrupted:
            # Scale down if we completed normally OR if a failure was detected
            if failed:
                log(f"\n✓ Scaling down cluster due to {failure_reason}...")
            else:
                log("\n✓ Scaling down cluster to 0 nodes...")
            
            run_command([
                "aws", "autoscaling", "update-auto-scaling-group",
                "--auto-scaling-group-name", "eks-12xlargeonlycifar-7ecc86b8-d4ee-d536-05ac-7e5b51bfcc15",
                "--desired-capacity", "0"
            ])

            # run_command([
            #     "aws", "autoscaling", "update-auto-scaling-group",
            #     "--auto-scaling-group-name", "eks-12xlarge-aacbfd83-cecd-921a-024a-1029e0120fe7",
            #     "--desired-capacity", "0"
            # ]) # this is the 12xlarge cluster
        
        log(f"\n=== All Experiments Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()