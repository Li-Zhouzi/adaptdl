import os
import csv
import sys
from collections import defaultdict

# Add adaptdl to path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptdl'))

from adaptdl.global_profile_state import GlobalProfileState

def decode_placement(placement_str):
    """
    Decode placement string to (num_nodes, num_replicas)
    Example: "1112" -> (2, 4) means 2 nodes with 4 total replicas
    Each digit represents the node ID where a replica is placed.
    """
    placement_str = str(placement_str)
    num_nodes = len(set(placement_str))  # Number of unique digits (unique nodes)
    num_replicas = len(placement_str)    # Total number of replicas (length of string)
    return num_nodes, num_replicas

def process_profiles_directory(profiles_dir):
    """
    Process all profile directories and create global_profiles and global_grad_params dictionaries.
    
    Args:
        profiles_dir (str): Path to the profiles directory
        
    Returns:
        tuple: (global_profiles, global_grad_params)
    """
    global_profiles = {}
    global_grad_params = {}
    
    # Check if profiles directory exists
    if not os.path.exists(profiles_dir):
        print(f"Profiles directory {profiles_dir} does not exist!")
        return global_profiles, global_grad_params
    
    # Iterate through all directories in the profiles directory
    for model_name in os.listdir(profiles_dir):
        model_path = os.path.join(profiles_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"Processing model: {model_name}")
        
        # Initialize dictionaries for this model
        global_profiles[model_name] = {}
        global_grad_params[model_name] = {}
        
        # Process placement files
        placement_file = None
        for filename in os.listdir(model_path):
            if filename.startswith('placements-') and filename.endswith('.csv'):
                placement_file = os.path.join(model_path, filename)
                break
        
        if placement_file and os.path.exists(placement_file):
            print(f"  Processing placement file: {os.path.basename(placement_file)}")
            
            # Dictionary to collect values for averaging (in case of duplicates)
            placement_data = defaultdict(list)
            
            with open(placement_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    placement = row['placement']
                    local_bsz = int(row['local_bsz'])
                    step_time = float(row['step_time'])
                    sync_time = float(row['sync_time'])
                    
                    num_nodes, num_replicas = decode_placement(placement)
                    key = (num_nodes, num_replicas, local_bsz)
                    
                    placement_data[key].append({
                        'step_time': step_time,
                        'sync_time': sync_time
                    })
            
            # Average the values for duplicate keys and format for GlobalProfileState
            for key, values in placement_data.items():
                avg_step_time = sum(v['step_time'] for v in values) / len(values)
                avg_sync_time = sum(v['sync_time'] for v in values) / len(values)
                # Format according to GlobalProfileState expected structure
                global_profiles[model_name][key] = {
                    "accum_step_time": 0.0,
                    "accum_count": 0,
                    "optim_step_time": avg_step_time,
                    "optim_sync_time": avg_sync_time,
                    "optim_count": 1  # Set to 1 since we have averaged data
                }
            
            print(f"    Processed {len(global_profiles[model_name])} unique configurations")
        else:
            print(f"  No placement file found for {model_name}")
        
        # Process validation files
        validation_file = None
        for filename in os.listdir(model_path):
            if filename.startswith('validation-') and filename.endswith('.csv'):
                validation_file = os.path.join(model_path, filename)
                break
        
        if validation_file and os.path.exists(validation_file):
            print(f"  Processing validation file: {os.path.basename(validation_file)}")
            
            with open(validation_file, 'r') as f:
                reader = csv.DictReader(f)
                for epoch, row in enumerate(reader):
                    # Format as tuple for GlobalProfileState
                    global_grad_params[model_name][epoch] = (
                        float(row['grad_sqr']),
                        float(row['grad_var'])
                    )
            
            print(f"    Processed {len(global_grad_params[model_name])} validation entries (epochs 0-{len(global_grad_params[model_name])-1})")
        else:
            print(f"  No validation file found for {model_name}")
    
    return global_profiles, global_grad_params

def save_as_global_profile_state(global_profiles, global_grad_params, profiles_dir):
    """
    Create a GlobalProfileState instance and save it directly to the profile directory.
    """
    # Create GlobalProfileState instance
    global_state = GlobalProfileState()
    
    # Populate the global state with our processed data
    global_state.global_profiles = global_profiles
    global_state.global_grad_params = global_grad_params
    
    # Fit performance parameters for each application
    print("Fitting performance parameters for each application...")
    for application in global_profiles.keys():
        print(f"  Fitting perf_params for {application}...")
        try:
            global_state.fit_perf_params_for_application(application)
            print(f"    Successfully fitted perf_params for {application}")
        except Exception as e:
            print(f"    Failed to fit perf_params for {application}: {e}")
    
    # Save the GlobalProfileState directly to the profile directory
    checkpoint_path = os.path.join(profiles_dir, "global-profile-state")
    with open(checkpoint_path, 'wb') as f:
        global_state.save(f)
    print(f"Saved GlobalProfileState to: {checkpoint_path}")
    
    return global_state

# Main execution
if __name__ == "__main__":
    # Default profiles directory
    profiles_dir = "./sched/adaptdl_sched/policy/profiles"
    
    # Process the profiles
    global_profiles, global_grad_params = process_profiles_directory(profiles_dir)
    
    # Create and save GlobalProfileState with fitted perf_params
    global_state = save_as_global_profile_state(global_profiles, global_grad_params, profiles_dir)
    
    print(f"\nProcessing complete!")
    print(f"- global_profiles: {len(global_profiles)} models")
    print(f"- global_grad_params: {len(global_grad_params)} models")
    print(f"- global_perf_params: {len(global_state.global_perf_params)} models fitted")
