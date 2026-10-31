"""Print the jobs of a job list whose result is not yet in their output file (to resume a queue).

usage: python3 resume_filter.py jobs_rev/followup.txt > jobs_rev/followup_rest.txt
"""

import json
import shlex
import sys
from pathlib import Path

DEFAULTS = {"lam": "10", "seed": "0", "smoothing": "lse", "k": "10", "n": "0", "eps0": "2.0"}


def key(d):
    return (d["data"], d["method"], float(d["lam"]), int(d["seed"]), d["smoothing"], bool(d["match_error"]),
            round(float(d["eps0"]), 6), int(d["n"]), int(d["k"]))


def job_args(line):
    tok = shlex.split(line)
    a = dict(DEFAULTS, match_error=False)
    for i, t in enumerate(tok):
        if t == "--match_error":
            a["match_error"] = True
        elif t.startswith("--") and i + 1 < len(tok) and not tok[i + 1].startswith("--"):
            a[t[2:]] = tok[i + 1]
    return a


done = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    a = job_args(line)
    out = a["out"]
    if out not in done:
        done[out] = set()
        if Path(out).exists():
            for l in Path(out).read_text().splitlines():
                r = json.loads(l)
                if "max_ratio" in r:
                    rec = dict(DEFAULTS, match_error=False)
                    rec.update({k: v for k, v in r.items() if v is not None})
                    done[out].add(key(rec))
    if key(a) not in done[out]:
        print(line)
