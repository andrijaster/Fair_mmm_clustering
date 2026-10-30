"""Job list for the comparison of smooth maxima (log-sum-exp, sparsemax, 1.5-entmax, bump):
seven benchmarks, lambda in {1, 10, inf}, three seeds; the regularised maxima with the
temperatures of log-sum-exp (same smoothness) and with error-matched temperatures."""

from pathlib import Path

R = "python3 run_exp.py --method fmm"
DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
jobs = []
for d in DATASETS:
    for lam in [1, 10, "inf"]:
        for s in range(3):
            o = f"--data {d} --lam {lam} --seed {s} --out results/smoothing.jsonl"
            jobs.append(f"{R} {o}")
            for sm in ["sparsemax", "entmax15", "bump"]:
                jobs.append(f"{R} --smoothing {sm} {o}")
                jobs.append(f"{R} --smoothing {sm} --match_error {o}")
Path("jobs_rev").mkdir(exist_ok=True)
Path("jobs_rev/smoothing.txt").write_text("\n".join(jobs) + "\n")
print(len(jobs))
