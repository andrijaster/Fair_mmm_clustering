"""Quick pilot: k-means, FMM lambda sweep and (optionally) LSPP on one dataset."""

import json
import math
import sys
import time

import numpy as np
import torch

from fairmmm import _core as fmm

name = sys.argv[1]
run_lspp = len(sys.argv) > 2 and sys.argv[2] == "lspp"
k = 10
dev = "cuda"
x = torch.from_numpy(np.load(f"data/{name}.npy")).to(dev)
delta = fmm.compute_thresholds(x, k)


def report(tag, c, secs):
    r = fmm.evaluate(x, c, delta)
    print(f"{name:9s} {tag:22s} cost={r['kmeans_cost']:.4g} maxr={r['max_ratio']:.3f} "
          f"viol={r['frac_violating']:.4f} meanr={r['mean_ratio']:.3f} t={secs:.1f}s", flush=True)
    return r


t = time.time()
c = fmm.kmeans(x, k, 0)
report("kmeans", c, time.time() - t)

for lam in [0.1, 1.0, 10.0, 100.0, math.inf]:
    cfg = fmm.FMMConfig(lam=lam, iters=2000)
    t = time.time()
    c = fmm.fmm(x, delta, k, cfg)
    report(f"fmm lam={lam}", c, time.time() - t)

if run_lspp:
    sys.path.insert(0, "../gr")
    from individually_fair_clustering import fair_clustering_algorithms as fca

    xn = x.cpu().numpy().astype(np.float64)
    dn = delta.cpu().numpy().astype(np.float64)
    for algo in ["Greedy", "LSPP"]:
        t = time.time()
        if algo == "Greedy":
            c = fca.Greedy(xn, k=k, dist_threshold_vec=dn, coeff_anchor=3.0)
        else:
            c = fca.LocalSearchPlusPlus(xn, k=k, dist_threshold_vec=dn, coeff_anchor=3.0,
                                        coeff_search=1.0, number_of_iterations=500, use_lloyd=True)
        report(algo, np.array(c), time.time() - t)
