#!/bin/bash

# Start port forwarding from local registry to Kubernetes cluster
echo "Setting up port forwarding from local Docker registry to Kubernetes cluster..."
kubectl port-forward svc/docker-registry -n adaptdl 5000:5000 &
PF_PID=$!

# Store the PID so we can kill it later
echo "Port forwarding process started with PID: $PF_PID"
echo $PF_PID > .port-forward.pid

echo "Port forwarding is now active. Your local registry at localhost:32000 is accessible in the cluster as docker-registry.adaptdl:5000"
echo "To stop port forwarding, run: kill $(cat .port-forward.pid)" 