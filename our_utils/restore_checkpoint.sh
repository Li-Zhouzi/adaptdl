#!/bin/bash

# This script restores a local checkpoint file to the AdaptDL scheduler pod.
# It requires `jq` to be installed for JSON parsing.

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-local-checkpoint-file>"
  exit 1
fi

LOCAL_CHECKPOINT_PATH="$1"

if [ ! -f "$LOCAL_CHECKPOINT_PATH" ]; then
  echo "Error: Local checkpoint file not found at '$LOCAL_CHECKPOINT_PATH'"
  exit 1
fi

echo "Looking for scheduler pod..."

# Find running scheduler pod's namespace and name
# We use jq to parse the json output from kubectl
POD_INFO=$(kubectl get pods -o json | jq -r '.items[] | select(.metadata.name | contains("sched")) | select(.status.phase == "Running") | "\(.metadata.namespace) \(.metadata.name)"')

if [ -z "$POD_INFO" ]; then
  echo "No running scheduler pod found."
  echo "Please ensure:"
  echo "1. You have kubectl configured and access to the cluster."
  echo "2. The scheduler pod is running."
  echo "3. 'jq' is installed (https://stedolan.github.io/jq/)."
  exit 1
fi

read -r NAMESPACE POD_NAME <<< "$POD_INFO"

echo "Found scheduler pod $POD_NAME in namespace $NAMESPACE"

# As seen in fetch_checkpoint.py, the container is width-calculator
CONTAINER_NAME="global-profiler"
REMOTE_PATH="/pollux/checkpoint/global-profile-state"

echo "Uploading checkpoint from $LOCAL_CHECKPOINT_PATH to pod $POD_NAME container $CONTAINER_NAME..."

kubectl cp "$LOCAL_CHECKPOINT_PATH" "${NAMESPACE}/${POD_NAME}:${REMOTE_PATH}" -c "$CONTAINER_NAME"

echo "Successfully uploaded checkpoint."