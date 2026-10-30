"""Draw the small samples of distinct points used by exact_small.py (ten per dataset, 60 points each;
exact_small.py uses the first n of every sample). Writes samples.npz next to this script."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]

out = {}
for d in DATASETS:
    x = np.load(ROOT / "data" / f"{d}.npy", mmap_mode="r")
    rng = np.random.default_rng(2024)
    for s in range(10):
        idx = rng.choice(len(x), 400, replace=False)
        u = np.unique(np.asarray(x[idx], dtype=np.float64), axis=0)
        out[f"{d}_{s}"] = u[np.random.default_rng(s).permutation(len(u))[:60]]
np.savez_compressed(ROOT / "samples.npz", **out)
