#!/bin/bash

# Script to create Bottlerocket AMI with pre-cached container images

# Step 1: Create Bottlerocket configuration with ECR credentials
cat > bottlerocket-config.toml <<EOF
[settings.aws]
region = "us-east-1"

[settings.container-registry.credentials]
"399790253372.dkr.ecr.us-east-1.amazonaws.com" = { credential-helper = "ecr-credential-helper" }

[settings.host-containers.admin]
enabled = true

# Pre-pull images on boot
[settings.bootstrap-commands]
"pull-adaptdl-images" = {
  commands = [
    ["ctr", "-n", "k8s.io", "images", "pull", "399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:cifar10"],
    ["ctr", "-n", "k8s.io", "images", "pull", "399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:bert"],
    ["ctr", "-n", "k8s.io", "images", "pull", "399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:deepspeech2"]
  ]
}
EOF

# Step 2: Build custom Bottlerocket AMI
# This requires the Bottlerocket build system
git clone https://github.com/bottlerocket-os/bottlerocket.git
cd bottlerocket

# Add custom configuration
cp ../bottlerocket-config.toml variants/aws-k8s-1.28/

# Build the AMI
cargo make -e BUILDSYS_VARIANT=aws-k8s-1.28 ami

# Step 3: Create AMI with pre-cached images
# Launch instance with custom Bottlerocket
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-xxxxx \
  --instance-type t3.large \
  --query 'Instances[0].InstanceId' \
  --output text)

# Wait for instance to be ready
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

# Pull all images (wait 15 minutes for images to be fully pulled)
echo "Waiting 15 minutes for images to be pulled..."
sleep 900

# Create AMI from instance
AMI_ID=$(aws ec2 create-image \
  --instance-id $INSTANCE_ID \
  --name "bottlerocket-adaptdl-$(date +%Y%m%d)" \
  --description "Bottlerocket with pre-cached AdaptDL images" \
  --query 'ImageId' \
  --output text)

echo "Created AMI: $AMI_ID"

# Step 4: Use the custom AMI for new nodes
echo "Custom AMI created: $AMI_ID"
echo ""
echo "To use this AMI for new nodes, you can:"
echo "1. Update your EKS node group launch template to use AMI: $AMI_ID"
echo "2. Configure your auto-scaling group to use this AMI"
echo "3. Use eksctl with the --ami flag when creating new node groups"
echo ""
echo "Example command for creating a new node group with this AMI:"
echo "eksctl create nodegroup --cluster=<your-cluster-name> --name=<nodegroup-name> --ami=$AMI_ID --instance-types=<your-instance-type>"