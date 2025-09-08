#!/bin/bash
set -e

# --- Configuration ---
INSTANCE_ID="i-0d37bb8b3cf6d3813"
REGION="us-east-1"
IMG="399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images:cifar10"
PLATFORM="linux/amd64"

# Bottlerocket admin + ctr command (matching original snapshot.sh)
CTR_CMD='apiclient exec admin sheltie ctr -a /run/containerd/containerd.sock -n k8s.io'

echo "========================================="
echo "Testing image pull for: $IMG"
echo "Instance: $INSTANCE_ID"
echo "Region: $REGION"
echo "========================================="

# Get ECR password locally
echo -n "Getting ECR password... "
PASS="$(aws ecr get-login-password --region "$REGION")"
if [ -z "$PASS" ]; then
    echo "ERROR: Failed to get ECR password"
    exit 1
fi
echo "OK (length: ${#PASS} chars)"

# Format password for ctr
ECRPWD="--user AWS:${PASS}"

# Show the command (without password)
echo ""
echo "Command to execute on instance:"
echo "  $CTR_CMD images pull --platform $PLATFORM $IMG --user AWS:[HIDDEN]"
echo ""

# Send the pull command via SSM
echo -n "Sending SSM command... "
CMDID=$(aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "Pull Images" \
  --parameters commands="$CTR_CMD images pull --platform $PLATFORM $IMG $ECRPWD" \
  --query "Command.CommandId" --output text)
echo "Command ID: $CMDID"

# Monitor progress for large images
echo "Monitoring image pull (8.6GB image may take 15-30 min)..."
START_TIME=$(date +%s)
while true; do
    STATUS=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CMDID" --instance-id "$INSTANCE_ID" --query 'Status' --output text)
    ELAPSED=$(( $(date +%s) - START_TIME ))
    echo -ne "\r[$((ELAPSED/60))m $((ELAPSED%60))s] Status: $STATUS  "
    
    if [[ "$STATUS" == "Success" ]]; then
        echo ""
        echo "✓ SUCCESS: Image pulled successfully!"
        break
    elif [[ "$STATUS" == "Failed" ]] || [[ "$STATUS" == "TimedOut" ]]; then
        echo ""
        echo "✗ FAILED: Command $STATUS"
        
        # Get full details
        OUTPUT=$(aws ssm get-command-invocation --region "$REGION" --command-id "$CMDID" --instance-id "$INSTANCE_ID" --output json)
        echo "StdOut: $(echo "$OUTPUT" | jq -r '.StandardOutputContent')"
        echo "StdErr: $(echo "$OUTPUT" | jq -r '.StandardErrorContent')"
        echo "Exit Code: $(echo "$OUTPUT" | jq -r '.ResponseCode')"
        break
    fi
    
    # Timeout after 45 minutes
    if [[ $ELAPSED -gt 2700 ]]; then
        echo ""
        echo "✗ TIMEOUT: >45 minutes"
        break
    fi
    
    sleep 5
done
