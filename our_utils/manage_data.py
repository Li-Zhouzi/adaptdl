import csv
import os
from collections import defaultdict

def analyze_workload(workload_file):
    """
    Analyze the workload file to calculate arrival rates for each application type.
    
    Args:
        workload_file (str): Path to the workload CSV file
        
    Returns:
        dict: Dictionary with application types as keys and arrival rates as values
    """
    if not os.path.exists(workload_file):
        print(f"Workload file {workload_file} does not exist!")
        return {}
    
    # Count jobs by application type
    job_counts = defaultdict(int)
    last_time = 0
    
    print(f"Analyzing workload file: {workload_file}")
    
    with open(workload_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            time = float(row['time'])
            
            # Extract application type from job name (e.g., "cifar10-1" -> "cifar10")
            app_type = name.split('-')[0]
            job_counts[app_type] += 1
            
            # Track the last time
            if time > last_time:
                last_time = time
    
    print(f"Job counts by application type:")
    for app_type, count in job_counts.items():
        print(f"  {app_type}: {count} jobs")
    
    print(f"Last time in workload: {last_time}")
    
    # Calculate arrival rates (jobs per time unit)
    arrival_rates = {}
    for app_type, count in job_counts.items():
        if last_time > 0:
            arrival_rate = count / last_time
            arrival_rates[app_type] = arrival_rate
            print(f"  {app_type}: {arrival_rate:.8f} jobs/time_unit")
        else:
            arrival_rates[app_type] = 0.0
    
    return arrival_rates

def update_configs_file(arrival_rates, configs_file):
    """
    Update the ARRIVAL_RATE dictionary in the _configs.py file.
    
    Args:
        arrival_rates (dict): Dictionary with application types and their arrival rates
        configs_file (str): Path to the _configs.py file
    """
    if not os.path.exists(configs_file):
        print(f"Config file {configs_file} does not exist!")
        return
    
    print(f"Updating ARRIVAL_RATE in {configs_file}")
    
    # Read the current file
    with open(configs_file, 'r') as f:
        lines = f.readlines()
    
    # Find the ARRIVAL_RATE section and update it
    updated_lines = []
    in_arrival_rate = False
    
    for i, line in enumerate(lines):
        if 'ARRIVAL_RATE = {' in line:
            in_arrival_rate = True
            updated_lines.append(line)
        elif in_arrival_rate and line.strip() == '}':
            # Insert the updated arrival rates
            for app_type, rate in arrival_rates.items():
                updated_lines.append(f'    "{app_type}": {rate},\n')
            # Add any missing applications with rate 0.0
            existing_apps = set(arrival_rates.keys())
            default_apps = {"bert", "cifar10", "ncf", "imagenet", "deepspeech2", "yolov3"}
            for app in default_apps:
                if app not in existing_apps:
                    updated_lines.append(f'    "{app}": 0.0,\n')
            updated_lines.append(line)
            in_arrival_rate = False
        elif in_arrival_rate:
            # Skip the old arrival rate entries
            continue
        else:
            updated_lines.append(line)
    
    # Write the updated file
    with open(configs_file, 'w') as f:
        f.writelines(updated_lines)
    
    print("Successfully updated ARRIVAL_RATE in _configs.py")

def main():
    """
    Main function to analyze workload and update configs.
    """
    # Default paths
    workload_file = "./benchmark/workloads/workload-test.csv"
    configs_file = "./sched/adaptdl_sched/_configs.py"
    
    # Analyze the workload file
    arrival_rates = analyze_workload(workload_file)
    
    if not arrival_rates:
        print("No arrival rates calculated. Exiting.")
        return
    
    # Update the configs file
    update_configs_file(arrival_rates, configs_file)
    
    print("\nProcessing complete!")
    print("Updated arrival rates:")
    for app_type, rate in arrival_rates.items():
        print(f"  {app_type}: {rate:.8f}")

if __name__ == "__main__":
    main()
