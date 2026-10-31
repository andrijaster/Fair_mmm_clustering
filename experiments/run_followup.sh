#!/bin/bash
# follow-up runs for the smooth maxima, growing n and k = 100 on the GPU
cd "$(dirname "$0")"
python3 jobs_followup.py
xargs -P 3 -I{} sh -c '{} > /dev/null 2>> logs/failed_followup.log || echo "FAILED {}" >> logs/failed_followup.log' < jobs_rev/followup.txt
echo "FOLLOWUP SMALL DONE $(date)" >> logs/progress.log
xargs -P 1 -I{} sh -c '{} > /dev/null 2>> logs/failed_followup.log || echo "FAILED {}" >> logs/failed_followup.log' < jobs_rev/followup_big.txt
echo "FOLLOWUP DONE $(date)" >> logs/progress.log
