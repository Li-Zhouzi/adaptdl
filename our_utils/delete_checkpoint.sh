#!/bin/bash

# Simple script to delete the global-profile-state file

# Find the scheduler pod
POD_NAME=$(kubectl get pods -n adaptdl -o name | grep sched | head -1 | sed 's/pod\///')

if [ -z "$POD_NAME" ]; then
    echo "Error: No scheduler pod found"
    exit 1
fi

echo "Deleting global-profile-state file from pod: $POD_NAME"

# Delete the file
kubectl exec $POD_NAME -n adaptdl -c global-profiler -- rm -f /pollux/checkpoint/global-profile-state

if [ $? -eq 0 ]; then
    echo "Successfully deleted global-profile-state file"
else
    echo "Failed to delete global-profile-state file"
fi 