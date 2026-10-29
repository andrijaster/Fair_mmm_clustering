#!/bin/bash
# Experiments added after the internal review: GPU (3 parallel) and CPU (9 parallel) queues
# side by side, then the timing runs one at a time with the machine otherwise idle.
set -u
cd "$(dirname "$0")"
mkdir -p logs results

run_queue() {  # file, parallelism
  xargs -P "$2" -I{} bash -c '{} >> logs/'"$(basename "$1" .txt)"'.log 2>&1 || echo "FAILED: {}" >> logs/failed_rev.log' < "$1"
  echo "$(date) done $1" >> logs/progress_rev.log
}

(run_queue jobs_rev/gpu_rev.txt 3) &
GPU=$!
(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run_queue jobs_rev/cpu_rev.txt 9) &
CPU=$!
wait $GPU $CPU
run_queue jobs_rev/excl_rev.txt 1
echo "$(date) REVISION DONE" >> logs/progress_rev.log
