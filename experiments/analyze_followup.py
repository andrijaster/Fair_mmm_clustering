"""Tables and summaries of the follow-up runs (jobs_followup.py): paired comparison of the smooth
maxima over ten seeds, log-sum-exp with rescaled temperatures, Gowalla subsamples of growing size,
and k = 100 centres.

usage: python analyze_followup.py results ../paper/tables
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
LAMS = [1.0, 10.0, float("inf")]
PAIRED = [("bump", True, "BumpMax"), ("entmax15", True, "1.5-entmax"), ("sparsemax", True, "sparsemax"),
          ("lse_scaled", False, "log-sum-exp, rescaled")]


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [r for r in map(json.loads, p.read_text().splitlines()) if "max_ratio" in r]


def lam_tex(lam):
    return "\\infty" if math.isinf(lam) else f"{lam:g}"


def paired(res, tab):
    by = defaultdict(dict)  # (data, lam, variant) -> seed -> row
    for r in load(res / "smoothing.jsonl"):
        sm = r.get("smoothing", "lse")
        me = bool(r.get("match_error")) and sm != "lse"
        if sm == "lse" or me:
            by[(r["data"], float(r["lam"]), sm, me)][r["seed"]] = r
    for r in load(res / "lse_control.jsonl"):
        by[(r["data"], float(r["lam"]), "lse_scaled", False)][r["seed"]] = r
    rows, allpairs, summary = [], defaultdict(list), defaultdict(lambda: [0, 0, 0, 0])
    for d in DATASETS:
        for i, lam in enumerate(LAMS):
            base = by.get((d, lam, "lse", False), {})
            a0 = [base[s]["max_ratio"] for s in sorted(base)]
            cells = [f"{np.mean(a0):.3f} ({np.std(a0, ddof=1):.3f})" if len(a0) > 1 else "--"]
            for sm, me, _ in PAIRED:
                rs = by.get((d, lam, sm, me), {})
                seeds = sorted(set(rs) & set(base))
                if len(seeds) < 2:
                    cells.append("--")
                    continue
                diff = np.array([rs[s]["max_ratio"] - base[s]["max_ratio"] for s in seeds])
                allpairs[sm] += list(diff)
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                mark = "^{*}" if p < 0.05 else ""
                summary[sm][0] += 1
                summary[sm][1] += diff.mean() < 0
                summary[sm][2] += p < 0.05 and diff.mean() < 0
                summary[sm][3] += p < 0.05 and diff.mean() > 0
                cells.append(f"${diff.mean():+.3f}{mark}$")
            name = f"\\multirow{{3}}{{*}}{{\\textsf{{{d}}}}}" if i == 0 else ""
            rows.append(f"{name} & ${lam_tex(lam)}$ & " + " & ".join(cells) + " \\\\")
        if d != DATASETS[-1]:
            rows.append("\\midrule")
    (tab / "smoothing_paired_rows.tex").write_text("\n".join(rows) + "\n")
    for sm, me, lab in PAIRED:
        n, lower, sig_lower, sig_higher = summary[sm]
        x = np.array(allpairs[sm])
        if len(x):
            print(f"{lab:24s}: lower mean alpha in {lower}/{n} settings, significantly lower {sig_lower}, "
                  f"significantly higher {sig_higher}; over all {len(x)} pairs median diff {np.median(x):+.4f}, "
                  f"mean {x.mean():+.4f}, Wilcoxon p={wilcoxon(x).pvalue:.2g}, worst {x.max():+.3f}")
    for lam in LAMS:
        for sm, me, lab in PAIRED:
            diffs = []
            for d in DATASETS:
                base, rs = by.get((d, lam, "lse", False), {}), by.get((d, lam, sm, me), {})
                seeds = sorted(set(rs) & set(base))
                if seeds:
                    diffs.append((d, np.mean([rs[s]["max_ratio"] - base[s]["max_ratio"] for s in seeds]),
                                  np.mean([rs[s]["kmeans_cost"] / base[s]["kmeans_cost"] for s in seeds])))
            if diffs:
                print(f"  lam={lam_tex(lam):7s} {lab:24s} " + ", ".join(f"{d} {a:+.3f} (cost {c:.2f}x)" for d, a, c in diffs))
    for sm, me, lab in PAIRED:  # running time relative to log-sum-exp, median over (data, lam)
        t = []
        for d in DATASETS:
            for lam in LAMS:
                base, rs = by.get((d, lam, "lse", False), {}), by.get((d, lam, sm, me), {})
                if base and rs:
                    t.append(np.mean([r["time"] for r in rs.values()]) / np.mean([r["time"] for r in base.values()]))
        if t:
            print(f"time ratio {lab}: median {np.median(t):.2f} range {min(t):.2f}-{max(t):.2f}")


def growing(res, tab):
    g = defaultdict(list)
    for r in load(res / "growing_n.jsonl"):
        sm = r.get("smoothing", "lse")
        g[(r["n_points"], sm, bool(r.get("match_error")) and sm != "lse")].append(r)
    rows = []
    for n in sorted({k[0] for k in g}):
        cells = []
        for v in [("lse", False), ("bump", False), ("bump", True)]:
            rs = g.get((n,) + v)
            cells.append(f"{np.mean([r['max_ratio'] for r in rs]):.3f} & {np.mean([r['time'] for r in rs]):.0f}"
                         if rs else "-- & --")
        rows.append(f"{n:,} & {math.log(n) / 200:.3f} & " + " & ".join(cells) + " \\\\")
        print("growing n", n, {f"{k[1]}{'-m' if k[2] else ''}": (round(np.mean([r['max_ratio'] for r in v]), 4),
                                                                round(np.std([r['max_ratio'] for r in v]), 4),
                                                                len(v)) for k, v in g.items() if k[0] == n})
    (tab / "growing_n_rows.tex").write_text("\n".join(rows) + "\n")


def k100(res, tab):
    g = defaultdict(list)
    for r in load(res / "k100.jsonl"):
        g[(r["method"], float(r["lam"]) if r["method"] == "fmm" else None)].append(r)
    km = g.get(("kmeans", None))
    if not km:
        return
    c0 = np.mean([r["kmeans_cost"] for r in km])
    rows = []
    for key, lab in [(("kmeans", None), "$k$-means"), (("fmm", 10.0), "FMM, $\\lambda=10$"),
                     (("fmm", float("inf")), "FMM, $\\lambda=\\infty$")]:
        rs = g.get(key)
        if rs:
            rows.append(f"{lab} & {np.mean([r['kmeans_cost'] for r in rs]) / c0:.2f} & "
                        f"{np.mean([r['max_ratio'] for r in rs]):.3f} & {100 * np.mean([r['frac_violating'] for r in rs]):.1f} & "
                        f"{np.mean([r['time'] for r in rs]):.0f} \\\\")
            print("k100", lab, len(rs), rows[-1])
    (tab / "k100_rows.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    res, tab = Path(sys.argv[1]), Path(sys.argv[2])
    tab.mkdir(parents=True, exist_ok=True)
    paired(res, tab)
    growing(res, tab)
    k100(res, tab)
