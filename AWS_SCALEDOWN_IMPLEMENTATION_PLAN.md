# Direct AWS ASG Control for Scale-Down (Simplified)

## Overview

Accelerate scale-down by directly calling AWS ASG API after placeholder pod deletion, reducing scale-down time from 5-10 minutes to ~2 minutes (60s checkpoint wait + AWS termination).

## Key Design Decisions

1. **Keep expander logic unchanged**: All existing placeholder pod logic remains the same
2. **Simple background task**: After `cluster_expander.fit()`, fire a background task that waits 60s then calls AWS API
3. **No complex state tracking**: Fire-and-forget approach, no pending state or cancellation logic needed
4. **60-second checkpoint wait**: Passive wait allows jobs to save checkpoints before node termination
5. **Feature flag controlled**: Environment variable `ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN` enables/disables feature (default: disabled)
6. **AWS authentication**: Use EKS IRSA (IAM Roles for Service Accounts) or EC2 instance IAM role
7. **Graceful error handling**: If AWS API fails, log error but placeholder deletion already happened (K8s autoscaler will eventually scale down)

## Architecture

### Current Scale-Down Flow
```
Policy → desired_nodes=2
  ↓
allocator._allocate() calls cluster_expander.fit(active_nodes)
  ↓
ClusterExpander deletes placeholder pods
  ↓
K8s autoscaler sees no pending pods
  ↓
After 5-10 minutes, K8s autoscaler scales down ASG
  ↓
ASG terminates excess nodes
```

### New Scale-Down Flow (Simplified)
```
Policy → desired_nodes=2
  ↓
allocator._allocate() calls cluster_expander.fit(active_nodes)
  ↓
ClusterExpander deletes placeholder pods (unchanged)
  ↓
allocator identifies nodes to terminate (nodes NOT in active_nodes)
  ↓
allocator fires background task: trigger_aws_scaledown(nodes_to_terminate)
  ↓
[Background task] Sleep 60 seconds (checkpoint grace period)
  ↓
[Background task] Map K8s node names to EC2 instance IDs
  ↓
[Background task] AWS API: terminate-instance-in-auto-scaling-group for each node
  ↓
ASG terminates specific instances (~2 min total time)
```

**Key Insight**: We don't replace the expander logic, we just add a shortcut AWS call to speed up the scale-down that would happen anyway.

## Implementation Plan

### Step 1: Create Simplified AWS Helper Module

**File**: `sched/adaptdl_sched/aws_scaledown.py` (NEW FILE, ~140 lines)

**Purpose**: Minimal AWS-specific helper for direct ASG control

**Implementation**:
```python
import asyncio
import logging
import os

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

LOG = logging.getLogger(__name__)


async def trigger_aws_scaledown(asg_name: str, nodes_to_terminate: list, wait_seconds: int = 60):
    """
    Background task: Wait then directly terminate specific EC2 instances.
    Fire-and-forget - no return value, just logs on completion/error.

    Args:
        asg_name: AWS Auto Scaling Group name
        nodes_to_terminate: List of Kubernetes node names to terminate
        wait_seconds: Grace period for checkpoint saving
    """
    try:
        LOG.info(f"[AWS ScaleDown] Scheduled termination of {len(nodes_to_terminate)} nodes in {wait_seconds}s: {nodes_to_terminate}")

        # Wait for checkpoint grace period
        await asyncio.sleep(wait_seconds)

        # Call AWS API in thread pool (boto3 is synchronous)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _terminate_instances_sync, asg_name, nodes_to_terminate)

        LOG.info(f"[AWS ScaleDown] SUCCESS: Terminated {len(nodes_to_terminate)} instances")
    except Exception as e:
        LOG.error(f"[AWS ScaleDown] FAILED: {e}", exc_info=True)


def _terminate_instances_sync(asg_name: str, node_names: list):
    """Synchronous AWS API call to terminate specific instances (runs in thread pool)."""
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not installed. Install with: pip install boto3")

    ec2 = boto3.client('ec2')
    autoscaling = boto3.client('autoscaling')

    # Map Kubernetes node names to EC2 instance IDs
    instance_ids = []
    for node_name in node_names:
        # Kubernetes node name format: ip-10-0-1-23.ec2.internal
        # Extract instance ID by querying EC2 API
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'private-dns-name', 'Values': [node_name]},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )

        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_ids.append(instance['InstanceId'])
                LOG.info(f"[AWS ScaleDown] Mapped {node_name} -> {instance['InstanceId']}")

    if not instance_ids:
        LOG.warning(f"[AWS ScaleDown] No instances found for nodes: {node_names}")
        return

    # Terminate each instance in the ASG
    for instance_id in instance_ids:
        autoscaling.terminate_instance_in_auto_scaling_group(
            InstanceId=instance_id,
            ShouldDecrementDesiredCapacity=True  # Reduce desired capacity
        )
        LOG.info(f"[AWS ScaleDown] Terminated instance: {instance_id}")
```

**Key Features**:
- Terminates specific nodes (not in active_nodes list) rather than relying on ASG to choose
- Maps Kubernetes node names to EC2 instance IDs via EC2 API
- Uses `terminate-instance-in-auto-scaling-group` for precise control
- Automatically decrements ASG desired capacity
- Fire-and-forget design with comprehensive logging

### Step 2: Add Configuration Functions

**File**: `sched/adaptdl_sched/config.py` (lines 90-100)

Add three new configuration getters:

```python
def get_aws_asg_name():
    """Get AWS Auto Scaling Group name."""
    return os.getenv("ADAPTDL_AWS_ASG_NAME", "eks-12xlarge-cbd-1007-c8ccdfb1-0ed8-acd8-ee69-7bdc2245b084")

def get_enable_direct_asg_scaledown():
    """Check if direct AWS ASG scale-down is enabled."""
    return os.getenv("ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN", "false").lower() == "true"

def get_scaledown_wait_seconds():
    """Get wait time before executing scale-down."""
    return int(os.getenv("ADAPTDL_SCALEDOWN_WAIT_SECONDS", "60"))
```

**Note**: ASG name has default hardcoded value but can be overridden via environment variable for flexibility.

### Step 3: Integrate into Allocator (Minimal Changes)

**File**: `sched/adaptdl_sched/allocator.py`

**Change 1** - Import at top (after line 36):
```python
from adaptdl_sched.aws_scaledown import trigger_aws_scaledown
import adaptdl_sched.config as config
```

**Change 2** - Load config in `__init__()` (after line 74):
```python
# AWS direct scale-down configuration (optional feature)
self._aws_scaledown_enabled = config.get_enable_direct_asg_scaledown()
self._aws_asg_name = None
self._aws_scaledown_wait = 60

if self._aws_scaledown_enabled:
    self._aws_asg_name = config.get_aws_asg_name()
    self._aws_scaledown_wait = config.get_scaledown_wait_seconds()
    LOG.info(f"AWS direct scale-down enabled: ASG={self._aws_asg_name}, wait={self._aws_scaledown_wait}s")
```

**Change 3** - Add AWS call after `cluster_expander.fit()` (after line 443):

Keep existing line 443 unchanged:
```python
self._cluster_expander.fit(active_nodes)
```

Add immediately after:
```python
# Trigger direct AWS scale-down if enabled and scaling down
if (self._aws_scaledown_enabled and
    self._desired_num_nodes is not None and
    self._desired_num_nodes < len(nodes)):

    # Identify nodes to terminate (nodes NOT in active_nodes)
    active_node_names = set(n for n in active_nodes if not n.startswith("~"))
    all_node_names = set(nodes.keys())
    nodes_to_terminate = list(all_node_names - active_node_names)

    if nodes_to_terminate:
        # Fire background task (fire-and-forget)
        asyncio.create_task(
            trigger_aws_scaledown(
                self._aws_asg_name,
                nodes_to_terminate,
                self._aws_scaledown_wait
            )
        )
        LOG.info(f"[AWS ScaleDown] Triggered termination of {len(nodes_to_terminate)} nodes: {nodes_to_terminate}")
```

**That's it!** No other changes needed. The expander logic runs as usual, then we identify excess nodes and fire a background task to terminate them.

### Step 4: Add boto3 Dependency

**File**: `sched/requirements.txt`

Add at end:
```
boto3>=1.26.0
```

### Step 5: Configure IAM Role (AWS Setup)

Since you're using EKS, you'll use **IRSA (IAM Roles for Service Accounts)** via eksctl. This binds an IAM role to the `adaptdl` ServiceAccount.

**Step 5a: Create IAM Policy**

Create a file `adaptdl-autoscaling-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
        "ec2:DescribeInstances"
      ],
      "Resource": "*"
    }
  ]
}
```

Create the policy:
```bash
aws iam create-policy \
  --policy-name AdaptDLAutoScaling \
  --policy-document file://adaptdl-autoscaling-policy.json
```

Note the policy ARN from the output (format: `arn:aws:iam::YOUR_ACCOUNT_ID:policy/AdaptDLAutoScaling`)

**Step 5b: Associate IAM Role with ServiceAccount**

```bash
# Create IAM role with autoscaling permissions and associate with SA
eksctl create iamserviceaccount \
  --name=adaptdl \
  --namespace=adaptdl \
  --cluster=YOUR_CLUSTER_NAME \
  --attach-policy-arn=arn:aws:iam::YOUR_ACCOUNT_ID:policy/AdaptDLAutoScaling \
  --approve \
  --override-existing-serviceaccounts
```

**Step 5c: Verify**

After setup, verify the annotation:
```bash
kubectl get sa adaptdl -n adaptdl -o yaml | grep role-arn
```

You should see:
```yaml
annotations:
  eks.amazonaws.com/role-arn: arn:aws:iam::YOUR_ACCOUNT_ID:role/eksctl-YOUR_CLUSTER-addon-iamserviceaccount-Role1-...
```

### Step 6: Enable Feature via ConfigMap

Update `helm/adaptdl-sched/templates/config.yaml` to add these environment variables:

```yaml
ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN: "true"
ADAPTDL_SCALEDOWN_WAIT_SECONDS: "60"
```

Then redeploy with your normal `./helm/update_adaptdl.sh` script. The scheduler will automatically pick up the new configuration.

## Critical Files to Modify

1. **sched/adaptdl_sched/aws_scaledown.py** (NEW FILE, ~140 lines)
   - `trigger_aws_scaledown()`: Async function to schedule termination
   - `_terminate_instances_sync()`: Maps K8s node names to EC2 instance IDs and terminates them
   - No classes, no state - pure functional approach

2. **sched/adaptdl_sched/allocator.py** (~20 lines added)
   - Import helper function (line 36)
   - Load config in `__init__()` (line 74)
   - Identify nodes to terminate and fire background task (line 443)

3. **sched/adaptdl_sched/config.py** (~15 lines added)
   - Three new getter functions (line 90)

4. **sched/requirements.txt** (1 line added)
   - Add `boto3>=1.26.0`

5. **helm/adaptdl-sched/templates/config.yaml** (2 lines)
   - Add `ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN: "true"`
   - Add `ADAPTDL_SCALEDOWN_WAIT_SECONDS: "60"`

**No changes to**:
- `cluster_expander.py` - All existing logic unchanged
- No new methods in allocator - just a few lines added to existing `_allocate()`

**Total**: ~180 lines of new code across 3 files

## Potential Edge Cases

1. **Multiple scale-down requests**: Fire-and-forget means multiple background tasks might run. This is OK - AWS API is idempotent, last call wins
2. **AWS API failure**: Logged but doesn't block. K8s autoscaler will eventually scale down via placeholder deletion
3. **Allocator loop faster than 60s**: Won't happen - allocator loop is 60s, same as our wait time

## Testing Plan

1. **Manual testing** (recommended approach):
   - Deploy to EKS with IAM role configured
   - Enable feature via environment variable
   - Trigger scale-down by reducing job requirements
   - Monitor logs for:
     - "[AWS ScaleDown] Scheduled: ..." message
     - 60-second wait
     - "[AWS ScaleDown] SUCCESS: ..." message
   - Verify ASG desired capacity changed: `aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names YOUR_ASG`
   - Verify nodes terminate after ~2 minutes total

2. **Dry run testing** (without AWS calls):
   - Set `ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN=true` but use invalid ASG name
   - Verify background tasks fire and log errors gracefully
   - Confirm allocator loop continues unaffected

## Rollback Strategy

If issues arise:
1. Set `ADAPTDL_ENABLE_DIRECT_ASG_SCALEDOWN=false` in ConfigMap or helm values
2. Redeploy scheduler: `./helm/update_adaptdl.sh`
3. System immediately reverts to K8s autoscaler (no code changes needed)

**Risk**: Very low - feature is opt-in, fire-and-forget design means errors don't block allocator.

## Benefits

- **Faster scale-down**: ~2 minutes (60s wait + AWS termination) instead of 5-10 minutes
- **Checkpoint safety**: 60s grace period for jobs to save state before node termination
- **Precise control**: Terminates specific nodes not in active_nodes, rather than relying on ASG to choose
- **Simple implementation**: ~180 lines of code total, no complex state management
- **Non-intrusive**: Expander logic completely unchanged, works alongside existing system
- **Secure**: Uses EKS IRSA or EC2 IAM role, no credential management needed
- **Zero risk when disabled**: Feature flag defaults to false

## Estimated Implementation Time

- Step 1 (AWS helper module): 20 min
- Step 2 (Config functions): 5 min
- Step 3 (Allocator integration): 10 min
- Step 4 (Dependencies): 2 min
- Step 5 (IAM role setup): 10 min
- Step 6 (Enable via env vars): 5 min
- Testing: 20 min
- **Total**: ~1.2 hours

## Summary

This simplified approach adds direct AWS ASG control as an optional acceleration layer on top of the existing K8s autoscaler mechanism. The expander continues to delete placeholder pods (signaling scale-down intent), but now we also fire a background task that:

1. Waits 60 seconds for jobs to save checkpoints
2. Identifies which specific nodes to terminate (nodes NOT in active_nodes)
3. Maps Kubernetes node names to EC2 instance IDs
4. Calls `terminate-instance-in-auto-scaling-group` for each excess node

This gives you precise control over which nodes are terminated while maintaining all existing safety mechanisms. Simple, safe, and effective!
