"""Cache the fairness thresholds of every (dataset, subsample, k) used by the jobs."""

from pathlib import Path

import numpy as np
import torch

from fairmmm import _core as fmm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "delta"
OUT.mkdir(parents=True, exist_ok=True)

combos = [(d, 0, 10) for d in ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]]
combos += [(d, 0, k) for d in ["adult", "bank", "diabetes", "skin"] for k in [5, 20, 50]]
combos += [("gowalla_full", n, 10) for n in [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 0]]

for d, n, k in combos:
    path = OUT / f"{d}_n{n}_k{k}.npy"
    if path.exists():
        continue
    x = np.load(ROOT / "data" / f"{d}.npy")
    if n and n < len(x):  # same subsample as run_exp.py
        x = x[np.random.default_rng(12345).choice(len(x), n, replace=False)]
    delta = fmm.compute_thresholds(torch.from_numpy(x).to("cuda"), k)
    np.save(path, delta.cpu().numpy())
    print(path.name, len(x), flush=True)
