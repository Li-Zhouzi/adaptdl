#!/bin/bash
our_utils/clean_checkpoints.sh

python benchmark/run_workload.py pollux benchmark/workloads/workload-test2.csv\
 --repository 399790253372.dkr.ecr.us-east-1.amazonaws.com/adaptdl-images