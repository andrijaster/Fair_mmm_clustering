#!/bin/bash
# Runs every experiment: GPU queue (3 parallel) and CPU queue (9 parallel) side by
# side, then the exclusive CPU timing runs alone. Progress: logs/*.log
set -u
cd "$(dirname "$0")"
mkdir -p logs results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_queue() {  # file, parallelism
  xargs -P "$2" -I{} bash -c '{} >> logs/'"$(basename "$1" .txt)"'.log 2>&1 || echo "FAILED: {}" >> logs/failed.log' < "$1"
  echo "$(date) done $1" >> logs/progress.log
}

(
  for q in gpu_main gpu_stochastic gpu_ablation; do run_queue jobs/$q.txt 3; done
) &
GPU=$!
(
  for q in cpu_main cpu_scale cpu_k; do run_queue jobs/$q.txt 9; done
) &
CPU=$!
wait $GPU $CPU

# CPU timing of FMM uses all cores: run alone
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
run_queue jobs/excl_scale.txt 1
echo "$(date) ALL DONE" >> logs/progress.log
