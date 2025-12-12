# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import logging

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


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
