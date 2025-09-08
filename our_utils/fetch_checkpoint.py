import os
import sys
import subprocess
import json

# Add adaptdl to path for importing checkpoint functionality
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptdl'))

from adaptdl.global_profile_state import GlobalProfileState


def delete_local_checkpoint():
    """Delete the existing local checkpoint file if it exists."""
    local_path = "./checkpoint/global-profile-state"
    
    if os.path.exists(local_path):
        print(f"Deleting existing local checkpoint file: {local_path}")
        try:
            os.remove(local_path)
            print("Successfully deleted existing local checkpoint file")
            return True
        except Exception as e:
            print(f"Failed to delete local checkpoint file: {e}")
            return False
    else:
        print("No existing local checkpoint file found")
        return True


def find_scheduler_pod():
    """Find the scheduler pod with 'sched' in its name."""
    print("Looking for scheduler pod...")
    
    # Get all pods and find the scheduler
    cmd = ["kubectl", "get", "pods", "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to get pods: {result.stderr}")
        return None
    
    pods_data = json.loads(result.stdout)
    
    for pod in pods_data["items"]:
        # Check if pod is running and has 'sched' in name
        if pod["status"]["phase"] == "Running" and "sched" in pod["metadata"]["name"]:
            pod_name = pod["metadata"]["name"]
            namespace = pod["metadata"]["namespace"]
            print(f"Found scheduler pod {pod_name} in namespace {namespace}")
            return namespace, pod_name
    
    return None


def delete_checkpoint_file():
    """Delete the checkpoint file from the scheduler pod."""
    pod_info = find_scheduler_pod()
    
    if not pod_info:
        print("No running scheduler pod found")
        return False
    
    namespace, pod_name = pod_info
    container_name = "global-profiler"
    remote_path = "/pollux/checkpoint/global-profile-state"
    
    print(f"Deleting checkpoint file from pod {pod_name} container {container_name}...")
    
    # Use kubectl exec to delete the file
    cmd = [
        "kubectl", "exec", f"{pod_name}", "-n", namespace, "-c", container_name,
        "--", "rm", "-f", remote_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully deleted checkpoint file: {remote_path}")
        return True
    else:
        print(f"Failed to delete checkpoint file: {result.stderr}")
        return False


def download_checkpoint_via_kubectl():
    """Download checkpoint from scheduler pod using kubectl cp."""
    pod_info = find_scheduler_pod()
    
    if not pod_info:
        print("No running scheduler pod found")
        return None
    
    namespace, pod_name = pod_info
    container_name = "global-profiler"
    remote_path = "/pollux/checkpoint/global-profile-state"
    local_path = "./checkpoint/global-profile-state"
    
    # Create local directory
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Downloading checkpoint from pod {pod_name} container {container_name}...")
    
    # Use kubectl cp to get the file
    cmd = [
        "kubectl", "cp",
        f"{namespace}/{pod_name}:{remote_path}",
        local_path,
        "-c", container_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully downloaded checkpoint to {local_path}")
        
        # Check local file size
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"Downloaded file size: {file_size} bytes")
            if file_size == 0:
                print("WARNING: Downloaded file is empty!")
                return None
            return local_path
        else:
            print("ERROR: Downloaded file does not exist locally")
            return None
    else:
        print(f"Failed to download checkpoint: {result.stderr}")
        return None


# Uncomment the line below if you want to delete the checkpoint file
# delete_checkpoint_file()

# First, delete any existing local checkpoint
print("Step 1: Deleting existing local checkpoint...")
delete_local_checkpoint()

# Download checkpoint from scheduler pod
print("\nStep 2: Downloading checkpoint from scheduler pod...")
downloaded_file = download_checkpoint_via_kubectl()
app_wanted = ["cifar10"]
profiles = {}

if downloaded_file and os.path.exists(downloaded_file):
    print(f"Loading checkpoint from {downloaded_file}...")

    with open(downloaded_file, "rb") as f:
        global_state = GlobalProfileState()
        global_state.load(f)
        print("Successfully loaded checkpoint:")
        
        # Print some useful information about the loaded state
        if hasattr(global_state, 'global_profiles'):
            for app in app_wanted:
                if app in global_state.global_profiles:
                    profiles[app] = global_state.global_profiles[app]
            print(f"\nGlobal profiles: {profiles}")
        if hasattr(global_state, 'global_perf_params'):
            print(f"Global perf params: {list(global_state.global_perf_params.keys())}")
        if hasattr(global_state, 'global_goodput_profile'):
            print(f"Global goodput profile: {global_state.global_goodput_profile}")
else:
    print("Unable to download checkpoint from scheduler pod.")
    print("Make sure:")
    print("1. You have kubectl configured and access to the cluster")
    print("2. The scheduler pod is running")
    print("3. The checkpoint file exists at /pollux/checkpoint/global-profile-state")


for app in app_wanted:
    profile = profiles[app]
    print(f"App: {app}")
    for profile_name, profile_value in profile.items():
        print(f"{profile_name}: {profile_value}")
    print("-"*50)





