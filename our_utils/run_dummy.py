#!/usr/bin/env python3
import os
import subprocess
import time
import re
import signal
import sys
from datetime import datetime

# Configuration constant
NUM_GPU = 1

# Setup logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")

def log(message):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

def verify_and_update_policy():
    """Verify policy is 'dummy' and update the GPU count in allocator.py"""
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
    
    # Update the GPU number
    old_pattern = r'self\._policy = DummyPolicy\(num_gpus_per_job=\d+\) # Configure dummy as needed'
    new_line = f'self._policy = DummyPolicy(num_gpus_per_job={NUM_GPU}) # Configure dummy as needed'
    
    content = re.sub(old_pattern, new_line, content)
    
    # Write back
    with open(allocator_path, 'w') as f:
        f.write(content)
    
    log(f"✓ Updated DummyPolicy to use {NUM_GPU} GPU(s)")

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

def main():
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"=== Experiment Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    log(f"Starting experiment with NUM_GPU={NUM_GPU}")
    
    # Step 1: Verify and update policy
    verify_and_update_policy()
    
    # Step 2: Run helm update
    log("\n✓ Running helm update...")
    run_command(["./helm/update_adaptdl.sh"], shell=True, capture_output=False)
    time.sleep(420)  # Give it time to update
    
    # Step 3: Delete existing job
    log("\n✓ Deleting existing job...")
    run_command(["kubectl", "delete", "adaptdljob", "cifar10-0", "-n", "adaptdl", "--ignore-not-found=true"])
    time.sleep(10)
    
    # Step 4: Run workload
    log("\n✓ Running workload...")
    run_command(["./benchmark/run_workload.sh"], shell=True, capture_output=False)
    time.sleep(5)  # Give it time to start
    
    # Step 5: Start monitor in subprocess
    log_file = f"./experiment_results/dummy/{NUM_GPU}gpu.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log(f"\n✓ Starting monitor, output to: {log_file}")
    monitor_proc = subprocess.Popen(
        ["python", "benchmark/run_monitor.py", log_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Step 6: Check for completion every 60 seconds
    print("\n✓ Monitoring job completion...")
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
                log(f"\n=== Experiment Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                break
            else:
                print(f"Job still running... (checked at {time.strftime('%Y-%m-%d %H:%M:%S')})")
                
    except KeyboardInterrupt:
        log("\n\nInterrupted by user. Cleaning up...")
        monitor_proc.terminate()
        time.sleep(2)
        if monitor_proc.poll() is None:
            monitor_proc.kill()
        sys.exit(0)

if __name__ == "__main__":
    main()