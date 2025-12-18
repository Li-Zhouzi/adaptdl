#!/bin/bash
our_utils/clean_checkpoints.sh # clean job checkpoints
# our_utils/delete_checkpoint.sh # clean global checkpoint


python benchmark/run_workload.py pollux benchmark/workloads/workload-1-cbd.csv\
 --repository 399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images