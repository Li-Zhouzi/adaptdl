# Plan: Secure Supervisor Fix + Enhanced Logging

## Problem Summary
- Pod cifar10-54 group 11 rank 1 queried supervisor for master IP
- Received IP 192.168.72.194 (from node 72-145)
- But group 11 was allocated to nodes [66-149, 66-70] - never 72-145
- Unclear how a group 11 pod on 72-145 could exist

## Defensive Fixes (Graceful, Non-Breaking)

### 1. Filter Terminating Pods
**Action**: Skip pods with `deletion_timestamp != None`
**On mismatch**: Skip this pod, continue looking for valid pods
**Rationale**: Terminating pods may have stale data

### 2. Filter by Event Type
**Action**: Only process ADDED/MODIFIED events, skip DELETED
**On DELETED event**: Skip this event, continue watching
**Rationale**: Deleted pods shouldn't be in discovery results

### 3. Verify Node Matches Expected Allocation (Soft Check)
**Action**:
- Fetch job's `status.allocation` from AdaptDLJob
- For each pod with rank=R, check if `pod.spec.node_name` matches `allocation[R]`
**On mismatch**:
- Log warning with details
- **Skip this pod** (don't add to ip_list)
- Continue looking for valid pods
**Rationale**: Prevents returning IPs from pods on wrong nodes

### 4. Prevent Array Overwriting
**Action**: Only update `pod_ip_list[rank]` if it's currently `None`
**On already filled**: Log warning, skip this pod
**Rationale**: First valid pod wins, prevents corruption from late events

## Enhanced Logging

### Key Events to Log

```python
# Request start
LOG.info("[DISCOVER_START] job=%s group=%s namespace=%s ts=%.3f", ...)

# Each event
LOG.info("[DISCOVER_EVENT] type=%s pod=%s group=%s rank=%s/%s node=%s ip=%s del_ts=%s ts=%.3f",
         event_type, pod_name, pod_group, rank, replicas, node, ip, del_ts, time.time())

# When skipping a pod (with reason)
LOG.warning("[DISCOVER_SKIP] pod=%s reason=%s group=%s rank=%s node=%s ip=%s ts=%.3f",
            pod_name, reason, pod_group, rank, node, ip, time.time())

# When accepting a pod
LOG.info("[DISCOVER_ACCEPT] pod=%s group=%s rank=%s node=%s ip=%s current_list=%s ts=%.3f",
         pod_name, pod_group, rank, node, ip, pod_ip_list, time.time())

# Allocation fetched (for node verification)
LOG.info("[DISCOVER_ALLOC] job=%s group=%s allocation=%s ts=%.3f",
         name, group, allocation, time.time())

# Success return
LOG.info("[DISCOVER_SUCCESS] job=%s group=%s ips=%s duration=%.3f ts=%.3f",
         name, group, pod_ip_list, duration, time.time())

# Timeout
LOG.warning("[DISCOVER_TIMEOUT] job=%s group=%s partial_ips=%s ts=%.3f",
            name, group, pod_ip_list, time.time())
```

### Skip Reasons to Log
- `"deletion_timestamp_set"` - Pod is terminating
- `"event_type_deleted"` - DELETED event
- `"node_mismatch"` - Pod on wrong node vs allocation
- `"rank_already_filled"` - pod_ip_list[rank] already has an IP
- `"allocation_fetch_failed"` - Couldn't get job allocation (proceed anyway)

## Update check_running_health.py

### Add Supervisor Log Extraction

```python
# After extracting allocator logs, add:

# Extract supervisor logs
try:
    supervisor_pods = v1.list_namespaced_pod(
        namespace="adaptdl",
        label_selector="app.kubernetes.io/name=adaptdl-sched"
    ).items

    # Find supervisor container in the sched pod
    for pod in supervisor_pods:
        if pod.metadata.name.startswith("adaptdl-adaptdl-sched"):
            try:
                # Get supervisor container logs
                supervisor_log = v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace="adaptdl",
                    container="supervisor",  # supervisor container name
                    tail_lines=50000  # Last 50k lines
                )
                with open(f"{error_dir}/supervisor.txt", "w") as f:
                    f.write(supervisor_log)
                print(f"Saved supervisor logs to {error_dir}/supervisor.txt")

                # Also save previous logs if pod restarted
                try:
                    supervisor_log_prev = v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace="adaptdl",
                        container="supervisor",
                        previous=True,
                        tail_lines=50000
                    )
                    with open(f"{error_dir}/supervisor_previous.txt", "w") as f:
                        f.write(supervisor_log_prev)
                except:
                    pass  # No previous logs available

            except Exception as e:
                print(f"Could not get supervisor logs: {e}")
            break
except Exception as e:
    print(f"Could not find supervisor pod: {e}")

# Also filter supervisor logs for the failing job
try:
    if os.path.exists(f"{error_dir}/supervisor.txt"):
        with open(f"{error_dir}/supervisor.txt", "r") as f:
            all_lines = f.readlines()

        # Filter for job-specific and DISCOVER logs
        job_lines = [line for line in all_lines
                     if job_name in line or "[DISCOVER" in line]

        if job_lines:
            with open(f"{error_dir}/supervisor_{job_name}.txt", "w") as f:
                f.writelines(job_lines)
except Exception as e:
    print(f"Could not filter supervisor logs: {e}")
```

## Implementation Steps

### Step 1: Add Logging to supervisor.py
- Add all LOG statements
- No behavior changes yet
- Just observe what's happening

### Step 2: Update check_running_health.py
- Add supervisor log extraction
- Verify logs are captured when errors occur

### Step 3: Add Defensive Checks to supervisor.py
- Add deletion_timestamp check
- Add event type check
- Add node verification (graceful)
- Add overwrite protection
- All checks are "skip and continue", never crash

## Files to Modify

1. `sched/adaptdl_sched/supervisor.py` - Add logging and defensive checks
2. `our_utils/check_running_health.py` - Extract supervisor logs

No Dockerfile changes needed - supervisor already logs to stdout.

## Expected Outcomes

### If Error Happens Again
We'll see in logs:
- Exact sequence of events supervisor received
- Which pods were accepted/rejected and why
- What allocation supervisor saw from job status
- Final IP list and how it was constructed
- Timing of all events

### Improved Reliability
- Skip terminating pods → fewer stale IPs
- Skip DELETED events → no corruption from deletion events
- Verify node placement → catch scheduling errors
- No overwrites → first valid pod wins
- **All graceful** → experiments continue even with partial data
