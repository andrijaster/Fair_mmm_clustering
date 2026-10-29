#!/bin/bash
# Finish the exclusive timing phase early: keep the CPU curve up to n = 3M (the full
# 6.4M set takes ~1 h per CPU run), then time the GPU runs alone, then re-run the three
# baseline jobs that failed at start-up (CUDA out of memory).
cd ~/fair_mmm/code
until python3 -c "
import json, sys
ok = any(r['method'] == 'fmm' and r['device'] == 'cpu' and r['n'] == 3000000 and r['seed'] == 2
         for r in map(json.loads, open('results/scale.jsonl')))
sys.exit(0 if ok else 1)"; do sleep 30; done
pkill -f "[r]un_all.sh"
pkill -f "[x]args -P 1"
pkill -f "[r]un_exp.py --data gowalla_full --n 0 --method fmm --lam 10 --device cpu"
sleep 5
grep -v -- "--device cpu" jobs/excl_scale.txt > jobs_gpu/gpu_scale_timing.txt
while read -r cmd; do bash -c "$cmd" >> logs/gpu_scale_timing.log 2>&1 || echo "FAILED: $cmd" >> logs/failed_gpu.log; done < jobs_gpu/gpu_scale_timing.txt
echo "$(date) SCALE DONE" >> logs/progress.log
export OMP_NUM_THREADS=1
xargs -P 3 -I{} bash -c "{} >> logs/retry.log 2>&1" < jobs_gpu/retry.txt
echo "$(date) RETRY DONE" >> logs/progress.log
