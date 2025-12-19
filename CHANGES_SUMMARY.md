# Summary of Changes - Supervisor Fix & Enhanced Logging

## Implementation Date
December 18, 2025

## Problem Addressed
Pod cifar10-54 group 11 rank 1 received incorrect master IP (192.168.72.194 from node 72-145) when it should have been allocated to nodes [66-149, 66-70]. The exact cause remains unclear, but we've implemented defensive fixes and comprehensive logging to:
1. Prevent similar issues in the future
2. Capture detailed diagnostics if it happens again

---

## Changes Made

### 1. File: `sched/adaptdl_sched/supervisor.py`

#### Added Import
```python
import time  # For timestamp logging
```

#### Enhanced `_handle_discover()` Function

**Defensive Checks Added:**

1. **Filter Terminating Pods**
   - Skip pods with `deletion_timestamp != None`
   - Prevents using IPs from pods being deleted

2. **Filter DELETED Events**
   - Only process ADDED/MODIFIED events
   - Prevents corruption from deletion events

3. **Verify Node Placement**
   - Fetch job's `status.allocation` from AdaptDLJob
   - Verify `pod.spec.node_name` matches `allocation[rank]`
   - Skip pods on wrong nodes (gracefully, with warning)

4. **Prevent Array Overwriting**
   - Only update `pod_ip_list[rank]` if currently `None`
   - First valid pod wins, prevents late events from corrupting results

**Comprehensive Logging Added:**

- `[DISCOVER_START]` - Request initiated (job, group, namespace, timestamp)
- `[DISCOVER_ALLOC]` - Job allocation fetched successfully
- `[DISCOVER_ALLOC_FAILED]` - Failed to fetch allocation (continues without check)
- `[DISCOVER_EVENT]` - Every event received (type, pod, group, rank, node, IP, deletion_ts)
- `[DISCOVER_SKIP]` - Pod rejected with reason (deletion_timestamp_set, event_type_deleted, node_mismatch, rank_already_filled)
- `[DISCOVER_ACCEPT]` - Pod accepted (rank, node, IP, current IP list state)
- `[DISCOVER_SUCCESS]` - Complete IP list returned (IPs, duration)
- `[DISCOVER_TIMEOUT]` - Timeout occurred (partial IP list if any)

**Key Features:**
- All checks are graceful (skip bad pods, continue looking for valid ones)
- No assertions or crashes - experiments continue even with partial data
- Detailed logging shows exact event sequence and decision-making

---

### 2. File: `our_utils/check_running_health.py`

#### Modified `get_scheduler_logs()` Function

**Before:**
```python
containers = ["allocator", "width-calculator"]
```

**After:**
```python
containers = ["allocator", "width-calculator", "supervisor"]
```

Now extracts supervisor logs along with allocator and width-calculator logs.

#### Added `filter_supervisor_logs_for_job()` Function

New function that:
- Reads full supervisor logs from `supervisor.txt`
- Filters for lines containing:
  - The failing job name
  - `[DISCOVER` prefix (all discovery-related logs)
- Saves filtered logs to `supervisor_{job_name}.txt`
- Makes it easy to find relevant supervisor activity for the failing job

#### Modified `cleanup_on_failure()` Function

Added call to filter supervisor logs after extracting scheduler logs:

```python
get_scheduler_logs()
if job_name:
    filter_supervisor_logs_for_job(job_name)
get_nodes_info()
get_pods_info()
```

---

## What Gets Logged Now

### When a Pod Queries the Supervisor

For job `cifar10-54` group `11`:

```
[DISCOVER_START] job=cifar10-54 group=11 namespace=adaptdl ts=1765917754.422
[DISCOVER_ALLOC] job=cifar10-54 group=11 allocation=['ip-192-168-66-149.ec2.internal', 'ip-192-168-66-70.ec2.internal'] ts=1765917754.425
[DISCOVER_EVENT] type=ADDED pod=cifar10-54-...-11-0 group=11 rank=0/2 node=ip-192-168-66-149.ec2.internal ip=192.168.94.150 del_ts=None ts=1765917754.430
[DISCOVER_ACCEPT] pod=cifar10-54-...-11-0 group=11 rank=0 node=ip-192-168-66-149.ec2.internal ip=192.168.94.150 current_list=['192.168.94.150', None] ts=1765917754.430
[DISCOVER_EVENT] type=ADDED pod=cifar10-54-...-11-1 group=11 rank=1/2 node=ip-192-168-66-70.ec2.internal ip=192.168.64.88 del_ts=None ts=1765917754.440
[DISCOVER_ACCEPT] pod=cifar10-54-...-11-1 group=11 rank=1 node=ip-192-168-66-70.ec2.internal ip=192.168.64.88 current_list=['192.168.94.150', '192.168.64.88'] ts=1765917754.440
[DISCOVER_SUCCESS] job=cifar10-54 group=11 ips=['192.168.94.150', '192.168.64.88'] duration=0.018 ts=1765917754.440
```

### When Bad Pods Are Skipped

```
[DISCOVER_EVENT] type=MODIFIED pod=cifar10-54-...-10-0 group=10 rank=0/2 node=ip-192-168-72-145.ec2.internal ip=192.168.72.194 del_ts=2025-12-16T20:42:25Z ts=1765917754.435
[DISCOVER_SKIP] pod=cifar10-54-...-10-0 reason=deletion_timestamp_set group=10 rank=0 node=ip-192-168-72-145.ec2.internal ip=192.168.72.194 ts=1765917754.435
```

or

```
[DISCOVER_EVENT] type=ADDED pod=cifar10-54-...-11-1 group=11 rank=1/2 node=ip-192-168-72-145.ec2.internal ip=192.168.72.194 del_ts=None ts=1765917754.435
[DISCOVER_SKIP] pod=cifar10-54-...-11-1 reason=node_mismatch group=11 rank=1 expected_node=ip-192-168-66-70.ec2.internal actual_node=ip-192-168-72-145.ec2.internal ip=192.168.72.194 ts=1765917754.435
```

---

## Files Created on Error

When a job fails, the following logs are now collected in `experiment_results/{date}/errors/`:

### Existing Files
- `{job_name}_error.txt` - Failing job pod logs
- `allocator.txt` - Allocator logs
- `allocator_previous.txt` - Allocator logs from previous restart (if any)
- `width-calculator.txt` - Width calculator logs
- `width-calculator_previous.txt` - Width calculator logs from previous restart
- `nodes.txt` - Cluster nodes status
- `pods.txt` - All pods status

### New Files Added
- **`supervisor.txt`** - Full supervisor logs (last 50k lines)
- **`supervisor_previous.txt`** - Supervisor logs from previous restart (if any)
- **`supervisor_{job_name}.txt`** - Filtered supervisor logs showing only discovery events and events related to the failing job

---

## Expected Benefits

### 1. Prevent Known Failure Modes
- **Terminating pods**: Won't use IPs from pods being deleted
- **Wrong node placement**: Catches and skips pods on incorrect nodes
- **Event ordering issues**: First valid pod wins, no corruption from late events
- **Deleted events**: Won't process deletion events

### 2. Complete Debug Trail
If the error happens again, we'll see:
- Exact sequence of events supervisor received
- Which pods were accepted/rejected and why
- Expected vs actual node placement for each pod
- Complete timing of all events
- Job's allocation state at discovery time
- Final IP list construction step-by-step

### 3. Non-Breaking
- All checks are graceful (skip and continue)
- No assertions that could crash experiments
- Falls back to previous behavior if allocation fetch fails
- Experiments succeed even with partial data

---

## How to Use

### Normal Operation
No changes needed - supervisor runs automatically with enhanced logging.

### When Debugging Failures

1. Check `experiment_results/{date}/errors/supervisor_{job_name}.txt` for focused view
2. Check `experiment_results/{date}/errors/supervisor.txt` for complete supervisor activity
3. Look for `[DISCOVER_SKIP]` lines to see what was rejected and why
4. Compare `[DISCOVER_ALLOC]` with actual pod nodes to verify consistency

### Log Search Examples

Find all discovery attempts for a job:
```bash
grep "DISCOVER.*cifar10-54" errors/supervisor.txt
```

Find all rejected pods:
```bash
grep "DISCOVER_SKIP" errors/supervisor.txt
```

Find node mismatches:
```bash
grep "node_mismatch" errors/supervisor.txt
```

---

## Testing Status

- ✅ Syntax validation passed (both files compile without errors)
- ⚠️ Runtime testing pending (will be validated in next experiment run)
- ⚠️ Log extraction pending validation

---

## Next Steps

1. Deploy changes to cluster
2. Run experiment and verify supervisor logs appear
3. If error recurs, analyze supervisor logs to identify root cause
4. Based on findings, potentially add more defensive checks or fix underlying issue

---

## Rollback Plan

If issues arise, revert both files:
```bash
git checkout HEAD~1 sched/adaptdl_sched/supervisor.py our_utils/check_running_health.py
```

Then redeploy scheduler.
