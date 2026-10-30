"""Optimality gap of FMM (lambda = inf) on small instances: the min-max-min problem
min_C max_i min_j ||x_i - c_j|| / delta_i is solved exactly as a mixed-integer convex
quadratic program with SCIP and compared with FMM, k-means and the ratio greedy.

usage: python exact_small.py samples.npz out.jsonl [--n 30] [--k 3] [--limit 120] [--jobs 8]
       python exact_small.py out.jsonl table.tex --table      (LaTeX rows of the appendix table)
"""

import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import torch
from pyscipopt import Model, quicksum

from fairmmm import _core as fmm


def solve_exact(x, delta, k, warm, limit):
    """min T s.t. ||x_i - c_j||^2 <= T delta_i^2 + M_i (1 - z_ij), sum_j z_ij = 1;
    alpha* = sqrt(T*). Centres are confined to the bounding box of the data (an optimal
    centre can be projected onto the convex hull of its cluster), which gives M_i."""
    n, d = x.shape
    lo, hi = x.min(0), x.max(0)
    big = (np.maximum(x - lo, hi - x) ** 2).sum(1)
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", limit)
    m.setParam("parallel/maxnthreads", 1)
    t = m.addVar("T", lb=0.0)
    c = [[m.addVar(f"c{j}_{l}", lb=lo[l], ub=hi[l]) for l in range(d)] for j in range(k)]
    # symmetry breaking: point i may only join clusters 0..i
    z = [[m.addVar(f"z{i}_{j}", vtype="B") if j <= i else None for j in range(k)] for i in range(n)]
    for i in range(n):
        m.addCons(quicksum(v for v in z[i] if v is not None) == 1)
        for j in range(min(i + 1, k)):
            m.addCons(quicksum((x[i, l] - c[j][l]) * (x[i, l] - c[j][l]) for l in range(d))
                      <= t * delta[i] ** 2 + big[i] * (1 - z[i][j]))
    m.setObjective(t, "minimize")
    # warm start from the FMM centres (clusters relabelled to respect the symmetry breaking)
    a = np.linalg.norm(x[:, None] - warm[None], axis=2).argmin(1)
    order = list(dict.fromkeys(a.tolist()))
    order += [j for j in range(k) if j not in order]
    warm, a = warm[order], np.array([order.index(v) for v in a])
    sol = m.createSol()
    r = np.linalg.norm(x - warm[a], axis=1) / delta
    m.setSolVal(sol, t, float((r ** 2).max()) * (1 + 1e-9))
    for j in range(k):
        for l in range(d):
            m.setSolVal(sol, c[j][l], float(np.clip(warm[j, l], lo[l], hi[l])))
    for i in range(n):
        for j in range(min(i + 1, k)):
            m.setSolVal(sol, z[i][j], float(a[i] == j))
    m.addSol(sol, free=True)
    t0 = time.time()
    m.optimize()
    centres = np.array([[m.getVal(v) for v in row] for row in c])
    return {"status": m.getStatus(), "time": time.time() - t0,
            "lower": math.sqrt(max(m.getDualbound(), 0.0)), "upper": math.sqrt(max(m.getPrimalbound(), 0.0)),
            "centres": centres}


def alpha(x, c, delta):
    return float((np.linalg.norm(x[:, None] - c[None], axis=2).min(1) / delta).max())


DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]


def table(results, out):
    """Median (maximum) relative gap alpha / alpha* - 1 in percent, per dataset and size."""
    rows = [json.loads(line) for line in open(results)]
    sizes = sorted({r["n"] for r in rows})

    def cells(q):
        out = []
        for m in ("alpha_fmm", "alpha_kmeans", "alpha_greedy"):
            g = 100 * (np.array([r[m] / r["alpha_lower"] for r in q]) - 1)
            out.append(f"{np.median(g):.1f} ({g.max():.1f})")
        return out

    lines = []
    for d in DATASETS + ["all"]:
        if d == "all":
            lines.append("\\midrule")
        c = []
        for n in sizes:
            c += cells([r for r in rows if r["n"] == n and d in ("all", r["data"])])
        name = "all" if d == "all" else f"\\textsf{{{d}}}"
        lines.append(name + " & " + " & ".join(c) + " \\\\")
    open(out, "w").write("\n".join(lines) + "\n")
    for n in sizes:
        q = [r for r in rows if r["n"] == n]
        g = np.array([r["alpha_fmm"] / r["alpha_lower"] - 1 for r in q])
        print(n, len(q), {s: sum(r["status"] == s for r in q) for s in {r["status"] for r in q}},
              f"fmm within 1%: {np.mean(g <= 0.01):.2f}, within 5%: {np.mean(g <= 0.05):.2f}",
              f"milp time median {np.median([r['time_milp'] for r in q]):.1f}s max {max(r['time_milp'] for r in q):.1f}s",
              f"fmm time median {np.median([r['time_fmm'] for r in q]):.1f}s")


def job(args):
    name, x, n, k, limit = args
    torch.set_num_threads(1)
    x = x[:n]
    xt = torch.from_numpy(x).float()
    delta_t = fmm.exact_thresholds(xt, k)
    delta = delta_t.double().numpy()
    seed = int(name.rsplit("_", 1)[1])
    t0 = time.time()
    c_fmm = fmm.fmm(xt, delta_t, k, fmm.FMMConfig(lam=math.inf, seed=seed)).double().numpy()
    t_fmm = time.time() - t0
    c_km = fmm.kmeans(xt, k, seed).double().numpy()
    c_gr = fmm.ratio_greedy(xt, delta_t, k).double().numpy()
    res = solve_exact(x, delta, k, c_fmm, limit)
    return {"instance": name, "data": name.rsplit("_", 1)[0], "n": n, "k": k, "status": res["status"],
            "time_milp": res["time"], "time_fmm": t_fmm, "alpha_lower": res["lower"], "alpha_opt": res["upper"],
            "alpha_milp_check": alpha(x, res["centres"], delta),
            "alpha_fmm": alpha(x, c_fmm, delta), "alpha_kmeans": alpha(x, c_km, delta),
            "alpha_greedy": alpha(x, c_gr, delta)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=float, default=120)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        table(a.samples, a.out)
        raise SystemExit
    s = np.load(a.samples)
    names = [nm for nm in s.files if int(nm.rsplit("_", 1)[1]) < a.seeds and (not a.only or nm.startswith(a.only))]
    with Pool(a.jobs) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(job, [(nm, s[nm], a.n, a.k, a.limit) for nm in names]):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["instance"], r["status"], f"{r['time_milp']:.1f}s", f"opt={r['alpha_opt']:.3f}",
                  f"lb={r['alpha_lower']:.3f}", f"fmm={r['alpha_fmm']:.3f}", f"km={r['alpha_kmeans']:.3f}", flush=True)
