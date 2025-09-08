# ssh -i ~/.ssh/new_data_copier root@35.175.253.30 "rm -rf /mnt/efs/pollux/checkpoint/*"

# Find the scheduler pod
POD_NAME=$(kubectl get pods -n adaptdl -o name | grep sched | head -1 | sed 's/pod\///')

if [ -z "$POD_NAME" ]; then
    echo "Error: No scheduler pod found"
    exit 1
fi

echo "Deleting job checkpoints file from pod: $POD_NAME"

kubectl exec $POD_NAME -c global-profiler -- sh -c "rm -rf /pollux/checkpoint/pollux/checkpoint/*"