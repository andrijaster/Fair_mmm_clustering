"""Table and summary of the comparison of smooth maxima (results of jobs_smoothing.py).

usage: python analyze_smoothing.py results/smoothing.jsonl ../paper/tables/smoothing_rows.tex
"""

import json
import sys
from collections import defaultdict

import numpy as np

DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
LAMS = [1.0, 10.0, float("inf")]
VARIANTS = [("lse", False), ("sparsemax", False), ("entmax15", False), ("bump", False),
            ("sparsemax", True), ("entmax15", True), ("bump", True)]


def main(src, out):
    by = defaultdict(list)
    for line in open(src):
        r = json.loads(line)
        if "max_ratio" not in r:
            continue
        key = (r["data"], float(r["lam"]), r.get("smoothing", "lse"), bool(r.get("match_error")) and r.get("smoothing") != "lse")
        by[key].append(r)
    km = {}  # k-means cost from the LSE runs' reference: use the cost ratio to the LSE lambda=1 run is not needed;
    # costs are reported relative to the k-means cost stored with the main results if available
    try:
        for line in open(src.replace("smoothing.jsonl", "main_gpu.jsonl")):
            r = json.loads(line)
            if r.get("method") == "kmeans" and "kmeans_cost" in r:
                km.setdefault(r["data"], []).append(r["kmeans_cost"])
    except FileNotFoundError:
        pass
    km = {d: float(np.mean(v)) for d, v in km.items()}
    rows, wins, times = [], defaultdict(lambda: [0, 0]), defaultdict(list)
    for d in DATASETS:
        for i, lam in enumerate(LAMS):
            cells = []
            base = by.get((d, lam, "lse", False))
            for sm, me in VARIANTS:
                rs = by.get((d, lam, sm, me))
                if not rs:
                    cells.append("--")
                    continue
                a = np.mean([r["max_ratio"] for r in rs])
                c = np.mean([r["kmeans_cost"] for r in rs]) / km[d] if d in km else np.nan
                cells.append(f"{c:.2f} / {a:.3f}")
                times[(sm, me)].append(np.mean([r["time"] for r in rs]) / np.mean([r["time"] for r in base]) if base else np.nan)
                if base and sm != "lse":
                    ab = np.mean([r["max_ratio"] for r in base])
                    cb = np.mean([r["kmeans_cost"] for r in base])
                    cs = np.mean([r["kmeans_cost"] for r in rs])
                    w = wins[(sm, me, lam)]
                    w[0] += a < ab - 1e-3
                    w[1] += 1
            lam_tex = "\\infty" if lam == float("inf") else f"{lam:g}"
            name = f"\\multirow{{3}}{{*}}{{\\textsf{{{d}}}}}" if i == 0 else ""
            rows.append(f"{name} & ${lam_tex}$ & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule" if d != DATASETS[-1] else "")
    open(out, "w").write("\n".join(r for r in rows if r) + "\n")
    for (sm, me, lam), (w, n) in sorted(wins.items()):
        print(f"{sm:9s} {'matched' if me else 'same   '} lam={lam:g}: lower alpha than LSE on {w}/{n}")
    for (sm, me), t in times.items():
        print(f"time ratio to LSE {sm} {'matched' if me else 'same'}: median {np.nanmedian(t):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
