#!/bin/bash
set -e

INSTANCE_ID="i-0d37bb8b3cf6d3813"
REGION="us-east-1"

echo "Testing if AWS CLI exists on Bottlerocket instance..."
echo ""

# Test 1: Check if aws CLI exists
echo "Test 1: Checking for AWS CLI..."
CMDID=$(aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["apiclient exec admin -- sheltie which aws || echo AWS CLI not found"]' \
  --query "Command.CommandId" --output text)

sleep 2
aws ssm get-command-invocation --region "$REGION" --command-id "$CMDID" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text

echo ""
echo "Test 2: Try to get ECR password on instance..."
CMDID=$(aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["apiclient exec admin -- sheltie bash -c \"aws ecr get-login-password --region us-east-1 2>&1 || echo Failed\""]' \
  --query "Command.CommandId" --output text)

sleep 2
aws ssm get-command-invocation --region "$REGION" --command-id "$CMDID" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text

echo ""
echo "Test 3: Check what tools ARE available..."
CMDID=$(aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["apiclient exec admin -- sheltie ls /usr/bin/ | head -20"]' \
  --query "Command.CommandId" --output text)

sleep 2
aws ssm get-command-invocation --region "$REGION" --command-id "$CMDID" --instance-id "$INSTANCE_ID" \
  --query 'StandardOutputContent' --output text