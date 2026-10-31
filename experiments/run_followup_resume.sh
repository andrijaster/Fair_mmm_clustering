#!/bin/bash
# resume the follow-up queue: only jobs without a result
cd "$(dirname "$0")"
python3 resume_filter.py jobs_rev/followup.txt > jobs_rev/followup_rest.txt
python3 resume_filter.py jobs_rev/followup_big.txt > jobs_rev/followup_big_rest.txt
echo "FOLLOWUP RESUMED $(date): $(wc -l < jobs_rev/followup_rest.txt) + $(wc -l < jobs_rev/followup_big_rest.txt) jobs" >> logs/progress.log
xargs -P 3 -I{} sh -c '{} > /dev/null 2>> logs/failed_followup.log || echo "FAILED {}" >> logs/failed_followup.log' < jobs_rev/followup_rest.txt
echo "FOLLOWUP SMALL DONE $(date)" >> logs/progress.log
xargs -P 1 -I{} sh -c '{} > /dev/null 2>> logs/failed_followup.log || echo "FAILED {}" >> logs/failed_followup.log' < jobs_rev/followup_big_rest.txt
echo "FOLLOWUP DONE $(date)" >> logs/progress.log
