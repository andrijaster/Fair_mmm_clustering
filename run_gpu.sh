#!/bin/bash
# GPU queue (restarted with 10 k-means restarts for the initialisation): main grid,
# stochastic study, ablations, census case study. The CPU baselines keep running
# under run_all.sh, whose exclusive timing phase starts only after them.
set -u
cd "$(dirname "$0")"
mkdir -p logs results

run_queue() {  # file, parallelism
  xargs -P "$2" -I{} bash -c '{} >> logs/'"$(basename "$1" .txt)"'.log 2>&1 || echo "FAILED: {}" >> logs/failed_gpu.log' < "$1"
  echo "$(date) done $1" >> logs/progress.log
}

for q in gpu_main gpu_stochastic gpu_ablation; do run_queue jobs_gpu/$q.txt 3; done
python3 case_census.py >> logs/census.log 2>&1 || echo "FAILED: census" >> logs/failed_gpu.log
echo "$(date) GPU DONE" >> logs/progress.log
