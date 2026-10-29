"""Tables (LaTeX) and figures (PDF) for the paper from results/*.jsonl."""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "paper_out" / "figs"
TAB = ROOT / "paper_out" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

REGION_NAME = {"la_county": "Los Angeles Co.", "california": "California", "texas": "Texas", "us48": "US (48)"}
DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
# categorical slots (reference palette): FMM blue, TKM orange, LSPP aqua; other baselines ink
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#52514e",
    "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42,
})


def load(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def lam_key(v):
    return math.inf if v in ("inf", float("inf")) or (isinstance(v, float) and math.isinf(v)) else float(v)


def ms(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fmt_sci(m, s):
    e = int(math.floor(math.log10(abs(m)))) if m else 0
    return f"{m / 10**e:.2f}\\,$\\pm$\\,{s / 10**e:.2f}e{e}"


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        g[key[0] if len(keys) == 1 else key].append(r)
    return g


# ---------------------------------------------------------------------------- main table
def main_table(rows):
    ok = [r for r in rows if "kmeans_cost" in r]
    by = defaultdict(list)
    for r in ok:
        if r["method"] == "fmm":
            tag = ("fmm", lam_key(r["lam"]))
        elif r["method"] == "tkm":
            tag = ("tkm", r["tilt"])
        else:
            tag = (r["method"], None)
        by[(r["data"], tag)].append(r)
    lines = []
    summary = {}
    for d in DATASETS:
        if not any(k[0] == d for k in by):
            continue
        km = np.mean([r["kmeans_cost"] for r in by.get((d, ("kmeans", None)), [])] or [np.nan])
        # TKM: tilt with the lowest mean max-ratio (favourable to the baseline)
        tk = [k for k in by if k[0] == d and k[1][0] == "tkm"]
        best_t = min(tk, key=lambda k: np.mean([r["max_ratio"] for r in by[k]]))[1] if tk else None
        order = [("kmeans", None), ("greedy", None), ("icml20", None), ("lspp", None)]
        if best_t:
            order.append(best_t)
        order += [("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]
        names = {"kmeans": "$k$-means", "greedy": "Greedy", "icml20": "ICML20$^\\dagger$", "lspp": "LSPP"}
        first = True
        for tag in order:
            rs = by.get((d, tag))
            if not rs:
                continue
            if tag[0] == "fmm":
                label = "FMM ($\\lambda=" + ("\\infty" if math.isinf(tag[1]) else f"{tag[1]:g}") + "$)"
            elif tag[0] == "tkm":
                label = f"TKM ($t={tag[1]:g}$)"
            else:
                label = names[tag[0]]
            c, cs = ms([r["kmeans_cost"] for r in rs])
            a, as_ = ms([r["max_ratio"] for r in rs])
            v, _ = ms([r["frac_violating"] for r in rs])
            t, _ = ms([r["time"] for r in rs])
            summary[(d, tag)] = (c, a, v, t, len(rs))
            dname = f"\\multirow{{{len([o for o in order if (d, o) in by])}}}{{*}}{{{d}}}" if first else ""
            first = False
            lines.append(f"{dname} & {label} & {fmt_sci(c, cs)} & {c / km:.3f} & {a:.3f}\\,$\\pm$\\,{as_:.3f} "
                         f"& {100 * v:.2f} & {t:.1f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (TAB / "main_rows.tex").write_text("\n".join(lines) + "\n")
    return by, summary


# ---------------------------------------------------------------------------- Pareto fronts
def pareto(by):
    """Cost (relative to k-means) vs. maximal violation; axes zoomed on the region where the
    methods differ (TKM's high-tilt end and Greedy may fall outside, see caption)."""
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
    axes = axes.ravel()
    for ax, d in zip(axes, DATASETS):
        km = np.mean([r["kmeans_cost"] for r in by.get((d, ("kmeans", None)), [])] or [np.nan])
        if not np.isfinite(km):
            continue
        pts_x = []
        for meth, col, lab in [("tkm", ORANGE, "TKM (tilts)"), ("fmm", BLUE, r"FMM ($\lambda$ grid)")]:
            keys = sorted([k for k in by if k[0] == d and k[1][0] == meth], key=lambda k: k[1][1])
            xs = [np.mean([r["max_ratio"] for r in by[k]]) for k in keys]
            ys = [np.mean([r["kmeans_cost"] for r in by[k]]) / km for k in keys]
            ax.plot(xs, ys, "-o", color=col, label=lab, markeredgecolor="white", markeredgewidth=0.6, zorder=3,
                    clip_on=True)
            if meth == "fmm":
                pts_x += xs
        for meth, mk, col, lab in [("kmeans", "s", INK, "$k$-means"), ("greedy", "v", MUTED, "Greedy"),
                                   ("icml20", "D", MUTED, "ICML20"), ("lspp", "*", AQUA, "LSPP")]:
            rs = by.get((d, (meth, None)))
            if rs:
                x = np.mean([r["max_ratio"] for r in rs])
                ax.plot(x, np.mean([r["kmeans_cost"] for r in rs]) / km,
                        mk, color=col, markersize=8 if mk == "*" else 5, label=lab, linestyle="none",
                        markeredgecolor="white", markeredgewidth=0.6, zorder=4, clip_on=True)
                if meth in ("kmeans", "lspp", "icml20"):
                    pts_x.append(x)
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlim(min(pts_x + [1.0]) - 0.04, max(pts_x) + 0.12)
        ax.set_ylim(0.96, 2.3)
        ax.set_title(d)
    for ax in axes[4:7]:
        ax.set_xlabel(r"max ratio $\alpha$")
    for ax in axes[[0, 4]]:
        ax.set_ylabel("cost / $k$-means cost")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].axis("off")
    axes[7].legend(h, l, loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pareto.pdf")
    plt.close(fig)


def dominance(rows):
    """Per dataset: fraction of LSPP runs dominated (cost and alpha both <=) by the
    FMM front of the same seed."""
    out = {}
    for d in DATASETS:
        ls = [r for r in rows if r["data"] == d and r["method"] == "lspp" and "kmeans_cost" in r]
        if not ls:
            continue
        dom = 0
        for r in ls:
            front = [f for f in rows if f["data"] == d and f["method"] == "fmm" and f["seed"] == r["seed"]
                     and "kmeans_cost" in f]
            dom += any(f["kmeans_cost"] <= r["kmeans_cost"] and f["max_ratio"] <= r["max_ratio"] for f in front)
        out[d] = (dom, len(ls))
    return out


# ---------------------------------------------------------------------------- scalability
def scale():
    rows = [r for r in load("scale") if "kmeans_cost" in r]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    series = [
        ("FMM, GPU full batch", lambda r: r["method"] == "fmm" and r["device"] == "cuda" and not r["batch"], BLUE, "-o"),
        ("FMM, GPU mini-batch + memory", lambda r: r["method"] == "fmm" and r["batch"], BLUE, "--o"),
        ("FMM, CPU (12 threads)", lambda r: r["method"] == "fmm" and r["device"] == "cpu", ORANGE, "-s"),
        ("$k$-means (GPU)", lambda r: r["method"] == "kmeans", INK, ":^"),
        ("LSPP (CPU, 1 thread)", lambda r: r["method"] == "lspp", AQUA, "-*"),
    ]
    lines = []
    for lab, f, col, sty in series:
        g = group([r for r in rows if f(r)], "n_points")
        if not g:
            continue
        ns = sorted(g)
        axes[0].plot(ns, [np.mean([r["time"] for r in g[n]]) for n in ns], sty, color=col, label=lab,
                     markeredgecolor="white", markeredgewidth=0.6)
        axes[1].plot(ns, [np.mean([r["max_ratio"] for r in g[n]]) for n in ns], sty, color=col, label=lab,
                     markeredgecolor="white", markeredgewidth=0.6)
        for n in ns:
            lines.append(f"{lab} & {n} & {np.mean([r['time'] for r in g[n]]):.1f} & "
                         f"{np.mean([r['max_ratio'] for r in g[n]]):.3f} & {np.mean([r['kmeans_cost'] for r in g[n]]):.4g} \\\\")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of points $n$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("running time (s)")
    axes[1].set_ylabel(r"max ratio $\alpha$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "scale.pdf")
    plt.close(fig)
    (TAB / "scale_rows.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- stochastic
def stochastic():
    rows = [r for r in load("stochastic") if "kmeans_cost" in r]
    if not rows:
        return
    lines = []
    for d in sorted({r["data"] for r in rows}):
        for lam in sorted({lam_key(r["lam"]) for r in rows}):
            g = group([r for r in rows if r["data"] == d and lam_key(r["lam"]) == lam], "batch", "hard")
            for (b, h), rs in sorted(g.items()):
                c, cs = ms([r["kmeans_cost"] for r in rs])
                a, as_ = ms([r["max_ratio"] for r in rs])
                t, _ = ms([r["time"] for r in rs])
                ls = "\\infty" if math.isinf(lam) else f"{lam:g}"
                dn = "gowalla-full (10\\textsuperscript{6})" if d == "gowalla_full" else d
                lines.append(f"{dn} & ${ls}$ & {b if b else 'full'} & {h} & {fmt_sci(c, cs)} & "
                             f"{a:.3f}\\,$\\pm$\\,{as_:.3f} & {t:.1f} \\\\")
    (TAB / "stochastic_rows.tex").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, (d, lam) in zip(axes, [("covtype", math.inf), ("gowalla_full", math.inf)]):
        for (b, h), col, sty in [((0, 0), INK, "-"), ((4096, 0), ORANGE, "-"), ((4096, 1024), BLUE, "-"),
                                 ((1024, 0), ORANGE, ":"), ((1024, 1024), BLUE, ":")]:
            rs = [r for r in rows if r["data"] == d and lam_key(r["lam"]) == lam and r["batch"] == b
                  and r["hard"] == h and r.get("curve")]
            if not rs:
                continue
            it = np.array([p[0] for p in rs[0]["curve"]])
            a = np.mean([[p[3] for p in r["curve"]][: len(it)] for r in rs], axis=0)
            lab = "full batch" if not b else f"batch {b}" + (", memory 1024" if h else ", no memory")
            ax.plot(it, a, sty, color=col, label=lab)
        ax.set_title(("gowalla ($10^6$ check-ins)" if d == "gowalla_full" else d) + r", $\lambda=\infty$")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"max ratio $\alpha$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stochastic.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- ablations
def ablations():
    # with many duplicated points (diabetes has 134 distinct points) the sampled radius of
    # Bateni et al. is 0 for k >= 20, so alpha is infinite for every method: rows omitted
    rows = [r for r in load("k") if "kmeans_cost" in r and math.isfinite(r["max_ratio"])]
    lines = []
    for (d, k), g in sorted(group(rows, "data", "k").items()):
        km = np.mean([r["kmeans_cost"] for r in g if r["method"] == "kmeans"] or [np.nan])
        cells = []
        for m, lam in [("kmeans", None), ("lspp", None), ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]:
            rs = [r for r in g if r["method"] == m and (lam is None or lam_key(r["lam"]) == lam)]
            cells.append(f"{np.mean([r['kmeans_cost'] for r in rs]) / km:.2f} / {np.mean([r['max_ratio'] for r in rs]):.2f}"
                         if rs else "--")
        lines.append(f"{d} & {k} & " + " & ".join(cells) + " \\\\")
    (TAB / "k_rows.tex").write_text("\n".join(lines) + "\n")

    rows = [r for r in load("schedule") if "kmeans_cost" in r]
    lines = []
    for d, g in sorted(group(rows, "data").items()):
        variants = group(g, "eps0", "eps1", "iters")
        for (e0, e1, it), rs in sorted(variants.items()):
            kind = "fixed" if e0 == e1 else "geometric"
            c, _ = ms([r["kmeans_cost"] for r in rs])
            a, as_ = ms([r["max_ratio"] for r in rs])
            lines.append(f"{d} & {kind} & {e0:g}$\\to${e1:g} & {it} & {c:.4g} & {a:.3f}\\,$\\pm$\\,{as_:.3f} \\\\")
    (TAB / "schedule_rows.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- census
def census():
    dedup = {}
    for r in load("census"):  # some regions were run twice; keep the last run of each (region, method, seed)
        dedup[(r["region"], r["method"], r["seed"])] = r
    rows = list(dedup.values())
    if not rows:
        return
    lines = []
    order = ["wkmeans", "fmm_lam1.0", "fmm_lam10.0", "fmm_laminf", "fmm_lam10_bal1.0", "fmm_lam10_bal10.0",
             "fmm_lam10_bal100.0"]
    label = {"wkmeans": "weighted $k$-means", "fmm_lam1.0": "FMM $\\lambda=1$", "fmm_lam10.0": "FMM $\\lambda=10$",
             "fmm_laminf": "FMM $\\lambda=\\infty$", "fmm_lam10_bal1.0": "FMM $\\lambda=10$, $\\gamma=1$",
             "fmm_lam10_bal10.0": "FMM $\\lambda=10$, $\\gamma=10$",
             "fmm_lam10_bal100.0": "FMM $\\lambda=10$, $\\gamma=100$"}
    for reg in ["la_county", "california", "texas", "us48"]:
        g = [r for r in rows if r["region"] == reg]
        if not g:
            continue
        for m in order:
            rs = [r for r in g if r["method"] == m]
            if not rs:
                continue
            f = lambda key: np.mean([r[key] for r in rs])
            lines.append(f"{REGION_NAME[reg]} & {label[m]} & {f('max_ratio'):.2f} & {100 * f('pop_violating'):.1f} & "
                         f"{f('w_mean_dist'):.1f} & {f('max_dist'):.0f} & {f('norm_entropy'):.3f} & "
                         f"{100 * f('min_load'):.2f}--{100 * f('max_load'):.1f} \\\\")
        lines.append("\\midrule")
    (TAB / "census_rows.tex").write_text("\n".join(lines[:-1]) + "\n")

    pts_p = RES / "census_california_points.npy"
    if not pts_p.exists():
        return
    pts = np.load(pts_p)  # lat, lon, pop, delta
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, m in zip(axes, ["wkmeans", "fmm_lam10.0", "fmm_lam10_bal10.0"]):
        rs = [r for r in rows if r["region"] == "california" and r["method"] == m and r["seed"] == 0]
        if not rs:
            continue
        c = np.array(rs[0]["centers_latlon"])
        # ratio of each block group (great-circle distance / delta)
        lat1, lon1 = np.radians(pts[:, 0])[:, None], np.radians(pts[:, 1])[:, None]
        lat2, lon2 = np.radians(c[:, 0])[None], np.radians(c[:, 1])[None]
        hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        dist = (2 * 6371 * np.arcsin(np.sqrt(hav))).min(1)
        ratio = dist / pts[:, 3]
        sc = ax.scatter(pts[:, 1], pts[:, 0], c=np.clip(ratio, 0, 2), s=0.6, cmap="Blues", vmin=0, vmax=2,
                        linewidths=0, rasterized=True)
        bad = ratio > 1
        ax.scatter(pts[bad, 1], pts[bad, 0], s=1.2, color=ORANGE, linewidths=0, rasterized=True)
        ax.scatter(c[:, 1], c[:, 0], marker="*", s=60, color=INK, edgecolors="white", linewidths=0.6, zorder=5)
        ax.set_title(f"{label[m]}\n$\\alpha$={rs[0]['max_ratio']:.2f}, "
                     f"{100 * rs[0]['pop_violating']:.1f}% pop. violating")
        ax.set_aspect(1 / np.cos(np.radians(37)))
        ax.set_xlabel("longitude")
        ax.grid(False)
    axes[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes, shrink=0.7, label=r"$d(x,C)/\delta(x)$ (clipped at 2)")
    fig.savefig(FIG / "census_california.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- appendix tables
def full_grid(by):
    """Rows: lambda (FMM) or tilt (TKM); columns: datasets; cell 'cost ratio / alpha'."""
    for meth, fname, label in [("fmm", "grid_fmm.tex", "\\lambda"), ("tkm", "grid_tkm.tex", "t")]:
        params = sorted({k[1][1] for k in by if k[1][0] == meth})
        lines = []
        for prm in params:
            cells = []
            for d in DATASETS:
                km = np.mean([r["kmeans_cost"] for r in by.get((d, ("kmeans", None)), [])] or [np.nan])
                rs = by.get((d, (meth, prm)))
                cells.append(f"{np.mean([r['kmeans_cost'] for r in rs]) / km:.3f} / {np.mean([r['max_ratio'] for r in rs]):.3f}"
                             if rs else "--")
            ps = "\\infty" if math.isinf(prm) else f"{prm:g}"
            lines.append(f"${ps}$ & " + " & ".join(cells) + " \\\\")
        (TAB / fname).write_text("\n".join(lines) + "\n")


def exact():
    rows = [r for r in load("exact") if "kmeans_cost" in r]
    rows += [r for r in load("main") if r["method"] == "lspp" and r["data"] in ("adult", "bank", "shuttle")
             and "kmeans_cost" in r and r["seed"] < 5]
    rows += [r for r in load("main_gpu") if r["data"] in ("adult", "bank", "shuttle") and r["seed"] < 5
             and (r["method"] == "kmeans" or (r["method"] == "fmm" and lam_key(r["lam"]) in (0.1, 1.0, 10.0, math.inf)))]
    if not rows:
        return
    lines = []
    for d in ["adult", "bank", "shuttle"]:
        km = {dl: np.mean([r["kmeans_cost"] for r in rows if r["data"] == d and r["method"] == "kmeans"
                           and r.get("delta", "sample") == dl] or [np.nan]) for dl in ("sample", "exact")}
        for m, lam in [("kmeans", None), ("lspp", None), ("fmm", 0.1), ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]:
            cells = []
            for dl in ("sample", "exact"):
                rs = [r for r in rows if r["data"] == d and r["method"] == m and r.get("delta", "sample") == dl
                      and (lam is None or lam_key(r["lam"]) == lam)]
                cells.append(f"{np.mean([r['kmeans_cost'] for r in rs]) / km[dl]:.3f} & {np.mean([r['max_ratio'] for r in rs]):.3f}"
                             if rs else "-- & --")
            name = {"kmeans": "$k$-means", "lspp": "LSPP"}.get(m) or (
                "FMM $\\lambda=" + ("\\infty" if math.isinf(lam) else f"{lam:g}") + "$")
            lines.append(f"{d} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (TAB / "exact_rows.tex").write_text("\n".join(lines[:-1]) + "\n")


def compact(by):
    """Main-text table: methods x datasets; panels alpha, cost / k-means cost, time."""
    tilt = {}
    for d in DATASETS:
        tk = [k for k in by if k[0] == d and k[1][0] == "tkm"]
        if tk:
            tilt[d] = min(tk, key=lambda k: np.mean([r["max_ratio"] for r in by[k]]))[1]
    methods = [("$k$-means", ("kmeans", None)), ("Greedy", ("greedy", None)), ("ICML20$^\dagger$", ("icml20", None)),
               ("LSPP", ("lspp", None)), ("TKM$^{\\ast}$", "tkm"), ("FMM, $\lambda=1$", ("fmm", 1.0)),
               ("FMM, $\lambda=10$", ("fmm", 10.0)), ("FMM, $\lambda=\infty$", ("fmm", math.inf))]

    def cell(d, tag, f):
        tag = tilt.get(d) if tag == "tkm" else tag
        rs = by.get((d, tag)) if tag else None
        return f(rs, d) if rs else None

    km = {d: np.mean([r["kmeans_cost"] for r in by.get((d, ("kmeans", None)), [])] or [np.nan]) for d in DATASETS}
    panels = [("Maximal violation $\\alpha$ (lower is fairer)", lambda rs, d: np.mean([r["max_ratio"] for r in rs]), "{:.2f}", True),
              ("Cost relative to $k$-means", lambda rs, d: np.mean([r["kmeans_cost"] for r in rs]) / km[d], "{:.2f}", False),
              ("Running time (s)", lambda rs, d: np.mean([r["time"] for r in rs]), "{:.0f}", False)]
    out = ["\\begin{tabular}{l" + "r" * len(DATASETS) + "}", "\\toprule",
           "Method & " + " & ".join(f"\\textsf{{{d}}}" for d in DATASETS) + " \\\\"]
    for title, f, fmt, bold in panels:
        out += ["\\midrule", f"\\multicolumn{{{len(DATASETS) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        vals = {(name, d): cell(d, tag, f) for name, tag in methods for d in DATASETS}
        best = {d: min(v for (n, dd), v in vals.items() if dd == d and v is not None) for d in DATASETS}
        for name, tag in methods:
            cells = []
            for d in DATASETS:
                v = vals[(name, d)]
                if v is None:
                    cells.append("--")
                else:
                    t = fmt.format(v)
                    cells.append(f"\\textbf{{{t}}}" if bold and t == fmt.format(best[d]) else t)
            out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "main_compact.tex").write_text("\n".join(out) + "\n")
    (TAB / "tkm_tilts.tex").write_text(", ".join(f"\\textsf{{{d}}}: $t={tilt[d][1]:g}$" for d in DATASETS if d in tilt) + "\n")


if __name__ == "__main__":
    # baselines of Bateni et al. (CPU) from main.jsonl; GPU methods (k-means init with
    # 10 restarts) from main_gpu.jsonl
    rows = [r for r in load("main") if r["method"] in ("greedy", "lspp", "icml20")] + load("main_gpu")
    by, summary = main_table(rows)
    pareto(by)
    full_grid(by)
    compact(by)
    exact()
    dom = dominance(rows)
    print("LSPP runs dominated by the FMM front of the same seed:", dom)
    scale()
    stochastic()
    ablations()
    census()
    for (d, tag), (c, a, v, t, n) in sorted(summary.items(), key=lambda kv: (DATASETS.index(kv[0][0]), str(kv[0][1]))):
        print(f"{d:9s} {str(tag):22s} n={n:2d} cost={c:.4g} alpha={a:.3f} viol={100 * v:.2f}% t={t:.1f}s")
