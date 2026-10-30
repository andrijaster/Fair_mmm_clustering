#!/bin/bash
# comparison of smooth maxima on the GPU, three jobs at a time
cd "$(dirname "$0")"
python3 jobs_smoothing.py
xargs -P 3 -I{} sh -c '{} > /dev/null 2>> logs/failed_smoothing.log || echo "FAILED {}" >> logs/failed_smoothing.log' < jobs_rev/smoothing.txt
echo "SMOOTHING DONE $(date)" >> logs/progress.log
