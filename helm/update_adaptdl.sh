#!/bin/bash
 docker buildx build --platform linux/amd64 -t 399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-sched:latest -f sched/Dockerfile . --push
 helm uninstall adaptdl
 cd helm
helm install adaptdl ./adaptdl-sched -n adaptdl \
  --set image.repository=399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-sched \
  --set image.tag=latest


# upload the global profile state to the scheduler pod
# sleep 60
# cd ..
# ./our_utils/restore_checkpoint.sh ./checkpoint/global-profile-state