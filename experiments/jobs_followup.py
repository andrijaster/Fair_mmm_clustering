"""Follow-up jobs for the comparison of smooth maxima:
(a) seven more seeds (3-9) for log-sum-exp and the regularised maxima with matched temperatures;
(b) log-sum-exp with its temperatures rescaled as in matched BumpMax (eps * psi_b(1) / ln n,
    beta * psi_b(1) / ln k), which separates the shape of the regulariser from the temperature;
(c) Gowalla subsamples of growing size (lambda = inf): log-sum-exp, BumpMax with the temperatures of
    log-sum-exp (error bound independent of n) and with matched temperatures;
(d) k = 100 centres on 10^6 Gowalla check-ins.
Small jobs go to jobs_rev/followup.txt (three at a time), the full Gowalla runs to followup_big.txt."""

import math
from pathlib import Path

import numpy as np

R = "python3 run_exp.py"
DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
PSI_B1 = math.e * 0.21938393439552029  # e E_1(1)
K = 10
small, big = [], []
for d in DATASETS:  # (a)
    for lam in [1, 10, "inf"]:
        for s in range(3, 10):
            o = f"--data {d} --lam {lam} --seed {s} --out results/smoothing.jsonl"
            small.append(f"{R} --method fmm {o}")
            for sm in ["sparsemax", "entmax15", "bump"]:
                small.append(f"{R} --method fmm --smoothing {sm} --match_error {o}")
for d in DATASETS:  # (b)
    n = np.load(f"data/{d}.npy", mmap_mode="r").shape[0]
    co, ci = PSI_B1 / math.log(n), PSI_B1 / math.log(K)
    t = f"--eps0 {2 * co:.6g} --eps1 {200 * co:.6g} --beta0 {2 * ci:.6g} --beta1 {200 * ci:.6g}"
    for lam in [1, 10, "inf"]:
        for s in range(10):
            small.append(f"{R} --method fmm {t} --data {d} --lam {lam} --seed {s} --out results/lse_control.jsonl")
for n in [10000, 100000, 1000000, 0]:  # (c)
    for s in range(3):
        o = f"--data gowalla_full --n {n} --lam inf --seed {s} --out results/growing_n.jsonl"
        jobs = [f"{R} --method fmm {o}", f"{R} --method fmm --smoothing bump {o}",
                f"{R} --method fmm --smoothing bump --match_error {o}"]
        (big if n == 0 else small).extend(jobs)
for s in range(3):  # (d)
    o = f"--data gowalla_full --n 1000000 --k 100 --seed {s} --out results/k100.jsonl"
    small.append(f"{R} --method kmeans {o}")
    for lam in [10, "inf"]:
        small.append(f"{R} --method fmm --lam {lam} {o}")
Path("jobs_rev").mkdir(exist_ok=True)
Path("jobs_rev/followup.txt").write_text("\n".join(small) + "\n")
Path("jobs_rev/followup_big.txt").write_text("\n".join(big) + "\n")
print(len(small), len(big))
