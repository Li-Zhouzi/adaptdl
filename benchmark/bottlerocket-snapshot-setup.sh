#!/bin/bash

# Correct approach using AWS Bottlerocket image cache solution

# Step 1: Clone and use AWS's official solution
git clone https://github.com/aws-samples/bottlerocket-images-cache
cd bottlerocket-images-cache

# Step 2: Create snapshot with your AdaptDL images
# Make sure the IAM role has AmazonEC2ContainerRegistryReadOnly permission
SNAPSHOT_ID=$(./snapshot.sh -r us-east-1 \
  399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:cifar10,\
399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:bert,\
399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:deepspeech2 | grep "Snapshot ID" | awk '{print $3}')

echo "Created snapshot: $SNAPSHOT_ID"

# Step 3: Create Launch Template with the snapshot mapped to /dev/xvdb
aws ec2 create-launch-template \
  --launch-template-name adaptdl-bottlerocket-cached \
  --launch-template-data '{
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/xvdb",
        "Ebs": {
          "SnapshotId": "'$SNAPSHOT_ID'",
          "VolumeSize": 50,
          "VolumeType": "gp3",
          "DeleteOnTermination": true
        }
      }
    ]
  }'

# Step 4: Create eksctl config for node group
cat > nodegroup-bottlerocket.yaml <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: <your-cluster-name>  # Replace with your cluster name
  region: us-east-1

managedNodeGroups:
  - name: adaptdl-br-cached
    amiFamily: Bottlerocket
    instanceTypes: ["<your-instance-type>"]  # Replace with desired type
    desiredCapacity: 2
    minSize: 1
    maxSize: 10
    launchTemplate:
      id: $(aws ec2 describe-launch-templates --launch-template-names adaptdl-bottlerocket-cached --query 'LaunchTemplates[0].LaunchTemplateId' --output text)
      version: "\$Latest"
    iam:
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
EOF

echo "To create the node group, run:"
echo "eksctl create nodegroup -f nodegroup-bottlerocket.yaml"

# Step 5: Update process (when images change)
cat > update-cache.sh <<'UPDATEEOF'
#!/bin/bash
# Run this when you need to update cached images

# Create new snapshot
NEW_SNAPSHOT_ID=$(./bottlerocket-images-cache/snapshot.sh -r us-east-1 \
  399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:cifar10,\
399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:bert,\
399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:deepspeech2 | grep "Snapshot ID" | awk '{print $3}')

# Create new version of launch template
aws ec2 create-launch-template-version \
  --launch-template-name adaptdl-bottlerocket-cached \
  --launch-template-data "{
    \"BlockDeviceMappings\": [
      {
        \"DeviceName\": \"/dev/xvdb\",
        \"Ebs\": {
          \"SnapshotId\": \"$NEW_SNAPSHOT_ID\",
          \"VolumeSize\": 50,
          \"VolumeType\": \"gp3\",
          \"DeleteOnTermination\": true
        }
      }
    ]
  }"

echo "Update node group to use new launch template version"
UPDATEEOF

chmod +x update-cache.sh