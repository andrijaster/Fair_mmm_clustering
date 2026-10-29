#!/bin/bash
cd ~/fair_mmm/code
export OMP_NUM_THREADS=1
xargs -P 2 -I{} bash -c "{} >> logs/cpu_exact.log 2>&1 || echo FAILED: {} >> logs/failed_exact.log" < jobs_gpu/cpu_exact.txt &
until grep -q "GPU DONE" logs/progress.log 2>/dev/null; do sleep 60; done
xargs -P 3 -I{} bash -c "{} >> logs/gpu_exact.log 2>&1 || echo FAILED: {} >> logs/failed_exact.log" < jobs_gpu/gpu_exact.txt
wait
echo "$(date) EXACT DONE" >> logs/progress.log
