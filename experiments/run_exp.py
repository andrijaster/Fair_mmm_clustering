"""Run one (dataset, method, seed) job and append the result as a JSON line.

usage: python3 run_exp.py --data adult --method fmm --lam 10 --seed 0 --out results/main.jsonl
methods: kmeans, fmm, tkm, greedy, lspp, icml20
"""

import argparse
import os
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from fairmmm import _core as fmm
except ImportError:  # flat layout of the original experiment machine
    import fmm

ROOT = Path(__file__).resolve().parent
# google-research checkout holding individually_fair_clustering/ (baselines of Bateni et al.)
sys.path.insert(0, os.environ.get("FAIRMMM_BASELINES", str(ROOT / "gr")))

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lam", type=float, default=10.0)
ap.add_argument("--tilt", type=float, default=1.0)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--hard", type=int, default=0)
ap.add_argument("--eps1", type=float, default=200.0)
ap.add_argument("--beta1", type=float, default=200.0)
ap.add_argument("--eps0", type=float, default=2.0)
ap.add_argument("--beta0", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--n", type=int, default=0, help="subsample size (0 = all)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--slow_sample", type=int, default=4000)
ap.add_argument("--lspp_iters", type=int, default=500)
ap.add_argument("--log_curve", action="store_true")
ap.add_argument("--delta", default="sample", choices=["sample", "exact"],
                help="sample: estimator of Bateni et al.; exact: ceil(n/k)-th nearest neighbour")
ap.add_argument("--no_select", action="store_true", help="return the last iterate (no best-iterate check)")
ap.add_argument("--nosmooth", action="store_true", help="ablation: subgradient of the exact objective")
ap.add_argument("--smoothing", default="lse", choices=["lse", "sparsemax", "entmax15", "bump"],
                help="smooth max/min used in the surrogate")
ap.add_argument("--match_error", action="store_true",
                help="regularised max: temperatures scaled to the error bound of log-sum-exp")
ap.add_argument("--init", default="kmeans", choices=["kmeans", "kmeans1", "random"])
ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
ap.add_argument("--radius_seed", type=int, default=100, help="seed of the 500-point sample of the radii")
ap.add_argument("--tkm_select", action="store_true", help="TKM: return the best iterate of its own objective")
ap.add_argument("--coeff_anchor", type=float, default=3.0)
ap.add_argument("--coeff_search", type=float, default=1.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)
dev = a.device
x_all = np.load(ROOT / "data" / f"{a.data}.npy")
if a.n and a.n < len(x_all):
    x_all = x_all[np.random.default_rng(12345).choice(len(x_all), a.n, replace=False)]
# thresholds are computed once on the GPU and cached, so that every method sees the
# same delta; CPU-only jobs then never touch the GPU (evaluation is float64 either way)
cache = ROOT / "data" / "delta" / (f"{a.data}_n{a.n}_k{a.k}" + ("_exact" if a.delta == "exact" else "")
                                  + (f"_rs{a.radius_seed}" if a.radius_seed != 100 else "") + ".npy")
eval_dev = "cuda" if dev == "cuda" else "cpu"
x_eval = torch.from_numpy(x_all).to(eval_dev)
if cache.exists():
    delta_eval = torch.from_numpy(np.load(cache)).to(eval_dev)
else:
    thr = fmm.exact_thresholds if a.delta == "exact" else fmm.compute_thresholds
    delta_eval = (thr(x_eval.to("cuda"), a.k) if a.delta == "exact"
                  else thr(x_eval.to("cuda"), a.k, seed=a.radius_seed)).to(eval_dev)
    cache.parent.mkdir(exist_ok=True)
    tmp = cache.with_suffix(f".{np.random.randint(1 << 30)}.tmp.npy")
    np.save(tmp, delta_eval.cpu().numpy())
    tmp.rename(cache)
x = x_eval.to(dev)
delta = delta_eval.to(dev)

if dev == "cuda":
    torch.cuda.synchronize()
extra = {}
t0 = time.time()
if a.method == "kmeans":
    c = fmm.kmeans(x, a.k, a.seed)
elif a.method == "fmm":
    cfg = fmm.FMMConfig(lam=a.lam, iters=a.iters, lr=a.lr, batch=a.batch, hard_buffer=a.hard,
                        eps0=a.eps0, eps1=a.eps1, beta0=a.beta0, beta1=a.beta1, seed=a.seed,
                        eval_every=50, select_best=not a.no_select, hard=a.nosmooth, init=a.init,
                        smoothing=a.smoothing, match_error=a.match_error)
    if a.batch and not a.log_curve:  # best-iterate tracking needs full passes: off for stochastic runs
        cfg.eval_every = 10 ** 9
    c = fmm.fmm(x, delta, a.k, cfg)
    extra["time_init"] = cfg.time_init
    if a.log_curve:
        extra["curve"] = cfg.log
elif a.method == "tkm":
    c = fmm.tilted_kmeans(x, a.k, a.tilt, iters=a.iters, lr=a.lr, seed=a.seed, select_best=a.tkm_select)
elif a.method == "ratio_greedy":
    c = fmm.ratio_greedy(x, delta, a.k)
elif a.method == "wkm_delta":  # k-means with weights 1/delta_i^2 (penalises distance relative to the radius)
    try:
        from fairmmm import _weighted as fx
    except ImportError:
        import fmm_ext as fx
    c = fx.weighted_kmeans(x, 1.0 / delta.pow(2), a.k, a.seed)
else:
    from individually_fair_clustering import fair_clustering_algorithms as fca

    xn = x_all.astype(np.float64)
    dn = delta_eval.cpu().numpy().astype(np.float64)
    if a.method == "icml20" and len(xn) > a.slow_sample:
        idx = np.sort(np.random.default_rng(a.seed).choice(len(xn), a.slow_sample, replace=False))
        xn, dn = xn[idx], dn[idx]
    try:
        if a.method == "greedy":
            c = fca.Greedy(xn, k=a.k, dist_threshold_vec=dn, coeff_anchor=3.0)
        elif a.method == "lspp":
            c = fca.LocalSearchPlusPlus(xn, k=a.k, dist_threshold_vec=dn, coeff_anchor=a.coeff_anchor,
                                        coeff_search=a.coeff_search,
                                        number_of_iterations=a.lspp_iters, use_lloyd=True)
        elif a.method == "icml20":
            c = fca.LocalSearchICML2020(xn, k=a.k, dist_threshold_vec=dn, coeff_anchor=3.0, coeff_search=1.0,
                                        epsilon=0.01, use_lloyd=False)
        else:
            raise SystemExit("unknown method " + a.method)
        c = np.array(c)
    except RuntimeError as e:
        c, extra["error"] = None, str(e)
if dev == "cuda":
    torch.cuda.synchronize()
secs = time.time() - t0

res = {k_: v for k_, v in vars(a).items() if k_ != "out"}
res["time"] = secs
res["n_points"] = int(len(x_all))
if c is not None:
    c = torch.as_tensor(np.asarray(c.cpu() if torch.is_tensor(c) else c), device=x_eval.device)
    res.update(fmm.evaluate(x_eval, c, delta_eval))
res.update(extra)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps({k_: v for k_, v in res.items() if k_ != "curve"}))
