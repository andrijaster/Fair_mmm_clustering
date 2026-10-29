"""Centres on the gowalla benchmark (k=10) mapped back to latitude/longitude.

Re-creates the 100k sample of prepare_data.py, runs k-means and FMM on the
standardised data with the cached thresholds, and stores centres (lat, lon) and
per-point ratios for the world map figure.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fairmmm import _core as fmm

ROOT = Path(__file__).resolve().parent
with gzip.open(ROOT / "data/raw/gowalla.txt.gz", "rt") as f:
    raw = pd.read_csv(f, sep="\t", header=None, usecols=[2, 3]).values.astype(np.float64)
raw = raw[(np.abs(raw[:, 0]) <= 90) & (np.abs(raw[:, 1]) <= 180)]
# prepare_data.py: rng(0) is first used for this sample (adult/bank/diabetes draw nothing)
sample = raw[np.random.default_rng(0).choice(len(raw), 100_000, replace=False)]
mu, sd = sample.mean(0), sample.std(0)
x_std = np.load(ROOT / "data/gowalla.npy")
assert np.allclose((sample - mu) / sd, x_std, atol=1e-4), "sample mismatch"

x = torch.from_numpy(x_std).cuda()
delta = torch.from_numpy(np.load(ROOT / "data/delta/gowalla_n0_k10.npy")).cuda()
out = {"latlon": sample.tolist(), "delta_std": delta.cpu().numpy().tolist(), "methods": {}}
for name, lam in [("kmeans", None), ("fmm_lam1", 1.0), ("fmm_lam10", 10.0), ("fmm_laminf", math.inf)]:
    c = fmm.kmeans(x, 10, 0) if lam is None else fmm.fmm(x, delta, 10, fmm.FMMConfig(lam=lam, seed=0))
    r = fmm.evaluate(x, c, delta)
    m = torch.cdist(x.double(), c.double()).min(1).values / delta.double()
    out["methods"][name] = {"centers_latlon": (c.cpu().numpy() * sd + mu).tolist(), "ratio": m.cpu().numpy().tolist(), **r}
    print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
(ROOT / "results/gowalla_map.json").write_text(json.dumps(out))
