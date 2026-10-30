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
W = 5.15  # text width of the journal template (inches): figures are drawn at their printed size
# categorical slots (reference palette): FMM blue, TKM orange, LSPP aqua; other baselines ink
BLUE, ORANGE, AQUA, INK, MUTED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984", "#6250d6"
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
    "lines.linewidth": 1.3, "lines.markersize": 3.5, "pdf.fonttype": 42,
})
FMM_LABEL = lambda lam: "FMM, $\\lambda=" + ("\\infty" if math.isinf(lam) else f"{lam:g}") + "$"


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
    return f"$({m / 10**e:.2f}\\pm{s / 10**e:.2f})\\cdot10^{{{e}}}$"


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        g[key[0] if len(keys) == 1 else key].append(r)
    return g


def tag_of(r):
    if r["method"] == "fmm":
        return ("fmm", lam_key(r["lam"]))
    if r["method"] == "tkm":
        return ("tkm", (r["lr"], r["tilt"]))
    return (r["method"], None)


def index(rows):
    by = defaultdict(list)
    for r in rows:
        if "kmeans_cost" in r and math.isfinite(r["max_ratio"]):
            by[(r["data"], tag_of(r))].append(r)
    return by


def mean_of(rs, key):
    return float(np.mean([r[key] for r in rs]))


def best_tkm(by, d):
    """TKM configuration (learning rate, tilt) with the lowest mean alpha: favourable to TKM."""
    keys = [k for k in by if k[0] == d and k[1][0] == "tkm"]
    return min(keys, key=lambda k: mean_of(by[k], "max_ratio"))[1][1] if keys else None


NAMES = {"kmeans": "$k$-means", "greedy": "Greedy", "icml20": "ICML20$^\\dagger$", "lspp": "LSPP",
         "ratio_greedy": "Ratio greedy", "wkm_delta": "$k$-means, $w_i=\\delta_i^{-2}$"}
ORDER = [("kmeans", None), ("wkm_delta", None), ("ratio_greedy", None), ("greedy", None), ("icml20", None),
         ("lspp", None), "tkm", ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]


def label_of(tag):
    if tag[0] == "fmm":
        return FMM_LABEL(tag[1])
    if tag[0] == "tkm":
        return "TKM$^{\\ast}$"
    return NAMES[tag[0]]


# ---------------------------------------------------------------------------- main tables
def main_table(by):
    """Appendix: absolute cost, ratio to k-means, alpha, share of unfair points, time (two tables)."""
    files = {"main_rows_a.tex": DATASETS[:4], "main_rows_b.tex": DATASETS[4:]}
    for fname, dsets in files.items():
        lines = []
        for d in dsets:
            km = mean_of(by[(d, ("kmeans", None))], "kmeans_cost")
            tags = [(best_tkm(by, d) and ("tkm", best_tkm(by, d))) if t == "tkm" else t for t in ORDER]
            tags = [t for t in tags if t and (d, t) in by]
            for i, tag in enumerate(tags):
                rs = by[(d, tag)]
                c, cs = ms([r["kmeans_cost"] for r in rs])
                a, as_ = ms([r["max_ratio"] for r in rs])
                v, _ = ms([r["frac_violating"] for r in rs])
                t, _ = ms([r["time"] for r in rs])
                dname = f"\\multirow{{{len(tags)}}}{{*}}{{\\textsf{{{d}}}}}" if i == 0 else ""
                lines.append(f"{dname} & {label_of(tag)} & {fmt_sci(c, cs)} & {c / km:.3f} & ${a:.3f}\\pm{as_:.3f}$ "
                             f"& {100 * v:.2f} & {t:.1f} \\\\")
            lines.append("\\midrule")
        (TAB / fname).write_text("\n".join(lines[:-1]) + "\n")


def compact(by):
    """Main text: methods x datasets; panels alpha, share of unfair points, cost ratio, time."""
    km = {d: mean_of(by[(d, ("kmeans", None))], "kmeans_cost") for d in DATASETS}
    rowtags = [(label_of(t if t != "tkm" else ("tkm", None)), t) for t in ORDER]

    def cell(d, tag, f):
        if tag == "tkm":
            bt = best_tkm(by, d)
            tag = ("tkm", bt) if bt else None
        rs = by.get((d, tag)) if tag else None
        return f(rs, d) if rs else None

    panels = [("Maximal violation $\\alpha$ (lower is fairer)", lambda rs, d: mean_of(rs, "max_ratio"), "{:.2f}", True),
              ("Points not served fairly, $r_i>1$ (\\%)", lambda rs, d: 100 * mean_of(rs, "frac_violating"), "{:.1f}", True),
              ("Cost relative to $k$-means", lambda rs, d: mean_of(rs, "kmeans_cost") / km[d], "{:.2f}", False)]
    out = ["\\begin{tabular}{l" + "r" * len(DATASETS) + "}", "\\toprule",
           "Method & " + " & ".join(f"\\textsf{{{d}}}" for d in DATASETS) + " \\\\"]
    for title, f, fmt, bold in panels:
        out += ["\\midrule", f"\\multicolumn{{{len(DATASETS) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        vals = {(name, d): cell(d, tag, f) for name, tag in rowtags for d in DATASETS}
        best = {d: min(v for (n, dd), v in vals.items() if dd == d and v is not None) for d in DATASETS}
        for name, tag in rowtags:
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
    tk = {d: best_tkm(by, d) for d in DATASETS}
    (TAB / "tkm_best.tex").write_text(", ".join(f"\\textsf{{{d}}} $({v[0]:g},{v[1]:g})$" for d, v in tk.items() if v)
                                      + "\n")


def full_grid(by):
    """FMM for every lambda; tuned TKM for every tilt (best learning rate per tilt)."""
    lines = []
    for lam in sorted({k[1][1] for k in by if k[1][0] == "fmm"}):
        cells = []
        for d in DATASETS:
            km = mean_of(by[(d, ("kmeans", None))], "kmeans_cost")
            rs = by.get((d, ("fmm", lam)))
            cells.append(f"{mean_of(rs, 'kmeans_cost') / km:.3f} / {mean_of(rs, 'max_ratio'):.3f}" if rs else "--")
        lines.append(("$\\infty$" if math.isinf(lam) else f"${lam:g}$") + " & " + " & ".join(cells) + " \\\\")
    (TAB / "grid_fmm.tex").write_text("\n".join(lines) + "\n")
    lines = []
    tilts = sorted({k[1][1][1] for k in by if k[1][0] == "tkm"})
    for q in tilts:
        cells = []
        for d in DATASETS:
            km = mean_of(by[(d, ("kmeans", None))], "kmeans_cost")
            keys = [k for k in by if k[0] == d and k[1][0] == "tkm" and k[1][1][1] == q]
            if not keys:
                cells.append("--")
                continue
            k = min(keys, key=lambda kk: mean_of(by[kk], "max_ratio"))
            cells.append(f"{mean_of(by[k], 'kmeans_cost') / km:.3f} / {mean_of(by[k], 'max_ratio'):.3f}")
        lines.append(f"${q:g}$ & " + " & ".join(cells) + " \\\\")
    (TAB / "grid_tkm.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- fronts
def front_points(by, d, meth, front_rows=None, key="max_ratio"):
    """(fairness, cost / k-means) of each parameter setting of a method; key = max_ratio (alpha) or
    frac_violating (share of points with r_i > 1)."""
    km = mean_of(by[(d, ("kmeans", None))], "kmeans_cost")
    if meth == "fmm":
        keys = sorted([k for k in by if k[0] == d and k[1][0] == "fmm"], key=lambda k: k[1][1])
        return [(mean_of(by[k], key), mean_of(by[k], "kmeans_cost") / km) for k in keys]
    if meth == "tkm":
        keys = [k for k in by if k[0] == d and k[1][0] == "tkm"]
        return [(mean_of(by[k], key), mean_of(by[k], "kmeans_cost") / km) for k in keys]
    if meth == "lspp":
        g = group([r for r in front_rows if r["data"] == d and "kmeans_cost" in r], "coeff_anchor", "coeff_search")
        return [(mean_of(rs, key), mean_of(rs, "kmeans_cost") / km) for rs in g.values()]
    if (d, (meth, None)) in by:  # single-setting methods
        return [(mean_of(by[(d, (meth, None))], key), mean_of(by[(d, (meth, None))], "kmeans_cost") / km)]
    return []


def hypervolume(points, ref):
    """Area dominated by a set of (alpha, cost) points (both minimised) inside the reference box."""
    nd = []
    for a, c in sorted({(a, c) for a, c in points if a < ref[0] and c < ref[1]}):
        if not nd or c < nd[-1][1]:
            nd.append((a, c))
    hv = 0.0
    for i, (a, c) in enumerate(nd):
        a_next = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        hv += (a_next - a) * (ref[1] - c)
    return hv


HV_METHODS = ["kmeans", "lspp1", "lspp", "tkm", "wkm_delta", "ratio_greedy", "fmm"]


def hv_points(by, d, meth, front_rows, key="max_ratio"):
    if meth == "lspp1":
        return front_points(by, d, "lspp_default", key=key)
    if meth == "lspp":
        return front_points(by, d, "lspp", front_rows, key) if any(r["data"] == d for r in front_rows) else []
    return front_points(by, d, meth, key=key)


def fronts(by, front_rows):
    """Hypervolume of the fronts (fairness, cost / k-means) per dataset; reference point = fa x the
    worst fairness value of k-means and default LSPP, cost ratio cc (default fa = 1.1, cc = 2.5)."""
    by = dict(by)
    for d in DATASETS:  # default LSPP as a single-setting method
        if (d, ("lspp", None)) in by:
            by[(d, ("lspp_default", None))] = by[(d, ("lspp", None))]

    def ref_of(d, key, fa, cc):
        base = [mean_of(by[(d, ("kmeans", None))], key)]
        if (d, ("lspp", None)) in by:
            base.append(mean_of(by[(d, ("lspp", None))], key))
        return (fa * max(base), cc)

    lines = []
    for d in DATASETS:
        ref = ref_of(d, "max_ratio", 1.1, 2.5)
        cells = []
        for meth in HV_METHODS:
            pts = hv_points(by, d, meth, front_rows)
            cells.append(f"{hypervolume(pts, ref):.3f}" if pts else "--")
        lines.append(f"\\textsf{{{d}}} & " + " & ".join(cells) + " \\\\")
    (TAB / "hv_rows.tex").write_text("\n".join(lines) + "\n")
    # ratio of the FMM hypervolume to the best other front (all other methods), several reference points;
    # also for the share of unfairly served points instead of alpha
    for key in ["max_ratio", "frac_violating"]:
        for fa, cc in [(1.05, 2.0), (1.1, 2.0), (1.1, 2.5), (1.1, 3.0), (1.2, 2.5), (1.2, 3.0)]:
            ratios = []
            for d in DATASETS:
                ref = ref_of(d, key, fa, cc)
                best = max(hypervolume(hv_points(by, d, m, front_rows, key), ref) for m in HV_METHODS if m != "fmm")
                best_m = max((m for m in HV_METHODS if m != "fmm"),
                             key=lambda m: hypervolume(hv_points(by, d, m, front_rows, key), ref))
                fm = hypervolume(hv_points(by, d, "fmm", front_rows, key), ref)
                ratios.append((fm / best if best > 0 else float("inf"), d, best_m))
            print(f"HV[{key}] ref ({fa}, {cc}): FMM / best other min {min(r[0] for r in ratios):.2f} "
                  f"max {max(r[0] for r in ratios):.2f} :: " + ", ".join(f"{d} {r:.2f} ({m})" for r, d, m in ratios))
    for d in DATASETS:
        pts = front_points(by, d, "lspp", front_rows) if any(r["data"] == d for r in front_rows) else []
        if pts:
            print(f"LSPP front {d}: alpha {min(p[0] for p in pts):.3f}-{max(p[0] for p in pts):.3f}, "
                  f"cost {min(p[1] for p in pts):.3f}-{max(p[1] for p in pts):.3f}")


def pareto(by, front_rows):
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.5))
    axes = axes.ravel()
    for ax, d in zip(axes, DATASETS):
        km = mean_of(by[(d, ("kmeans", None))], "kmeans_cost")
        xs_all, ys_all = [], []
        for meth, col, mk, lab in [("tkm", ORANGE, "o", "TKM (non-dominated settings)"), ("fmm", BLUE, "o", r"FMM ($\lambda$ grid)")]:
            pts = front_points(by, d, meth)
            if meth == "tkm":  # only the non-dominated tuned configurations
                pts = sorted(pts)
                nd = []
                for a, c in pts:
                    if not nd or c < nd[-1][1]:
                        nd.append((a, c))
                pts = nd
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col, label=lab, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3, ms=3)
            xs_all += [p[0] for p in pts]
            ys_all += [p[1] for p in pts]
        if any(r["data"] == d for r in front_rows):
            pts = sorted(front_points(by, d, "lspp", front_rows))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "^", color=AQUA, ms=3.5, label="LSPP (9 settings)",
                    markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        for meth, mk, col, lab, size in [("kmeans", "s", INK, "$k$-means", 4), ("icml20", "D", MUTED, "ICML20", 3.5),
                                         ("lspp", "*", AQUA, "LSPP (default)", 8), ("wkm_delta", "P", VIOLET,
                                         "$k$-means, $w_i=\\delta_i^{-2}$", 4.5), ("ratio_greedy", "X", MUTED,
                                         "ratio greedy", 4.5)]:
            rs = by.get((d, (meth, None)))
            if rs:
                x, y = mean_of(rs, "max_ratio"), mean_of(rs, "kmeans_cost") / km
                ax.plot(x, y, mk, color=col, ms=size, label=lab, linestyle="none", markeredgecolor="white",
                        markeredgewidth=0.5, zorder=5)
                xs_all.append(x)
                ys_all.append(y)
        ax.axvline(1.0, color=MUTED, linewidth=0.7, linestyle=":")
        ymax = min(max(ys_all) * 1.05, 3.2)
        ax.set_ylim(0.95, ymax)
        ax.set_xlim(min(xs_all + [1.0]) - 0.04, min(max(xs_all), 3.2) + 0.08)
        ax.set_title(d, family="sans-serif")
    for ax in axes[4:7]:
        ax.set_xlabel(r"maximal violation $\alpha$")
    for ax in axes[[0, 4]]:
        ax.set_ylabel("cost / $k$-means cost")
    handles, labels = {}, []
    for ax in axes[:7]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[7].axis("off")
    axes[7].legend(handles.values(), handles.keys(), loc="center", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    fig.savefig(FIG / "pareto.pdf")
    plt.close(fig)


def dominance(rows, by):
    """Number of LSPP runs strictly dominated (no worse in cost and alpha, better in one) by the
    mean of some lambda of the FMM grid, and by the mean k-means solution."""
    out = {}
    for d in DATASETS:
        ls = [r for r in rows if r["data"] == d and r["method"] == "lspp" and "kmeans_cost" in r]
        if not ls:
            continue
        fmm = [(mean_of(by[k], "kmeans_cost"), mean_of(by[k], "max_ratio")) for k in by if k[0] == d and k[1][0] == "fmm"]
        km = [(mean_of(by[(d, ("kmeans", None))], "kmeans_cost"), mean_of(by[(d, ("kmeans", None))], "max_ratio"))]

        def dom(c, a, cand):
            return any(cc <= c and aa <= a and (cc < c or aa < a) for cc, aa in cand)
        out[d] = (sum(dom(r["kmeans_cost"], r["max_ratio"], fmm) for r in ls),
                  sum(dom(r["kmeans_cost"], r["max_ratio"], km) for r in ls), len(ls))
    return out


def stats(by):
    """Bootstrap 95% CI of the difference in mean alpha and one-sided p-values. FMM with seed s
    starts from the k-means solution with seed s, so that comparison is paired (Wilcoxon
    signed-rank); runs of LSPP are independent of FMM (Mann-Whitney U)."""
    from scipy.stats import mannwhitneyu, wilcoxon

    rng = np.random.default_rng(0)
    lines = []
    for d in DATASETS:
        cells = []
        fr = {r["seed"]: r["max_ratio"] for r in by[(d, ("fmm", 1.0))]}
        f = np.array(list(fr.values()))
        kr = {r["seed"]: r["max_ratio"] for r in by[(d, ("kmeans", None))]}
        seeds = sorted(set(fr) & set(kr))
        dd = np.array([fr[t] - kr[t] for t in seeds])
        boot = [rng.choice(dd, len(dd)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = wilcoxon(dd, alternative="less").pvalue
        cells.append(f"${dd.mean():+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$ & {p:.1e}")
        o = np.array([r["max_ratio"] for r in by.get((d, ("lspp", None)), [])])
        if not len(o):
            cells.append("-- & --")
        else:
            diffs = [rng.choice(f, len(f)).mean() - rng.choice(o, len(o)).mean() for _ in range(10000)]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p = mannwhitneyu(f, o, alternative="less").pvalue
            cells.append(f"${f.mean() - o.mean():+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$ & {p:.1e}")
        lines.append(f"\\textsf{{{d}}} & " + " & ".join(cells) + " \\\\")
    (TAB / "stats_rows.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- scalability
def scale():
    rows = [r for r in load("scale") if "kmeans_cost" in r]
    excl = [r for r in load("scale_v2") if "kmeans_cost" in r]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
    series = [
        ("FMM, GPU, full batch", rows, lambda r: r["method"] == "fmm" and r["device"] == "cuda" and not r["batch"], BLUE, "-o"),
        ("FMM, GPU, mini-batch + memory", rows, lambda r: r["method"] == "fmm" and r["batch"]  # b0 <= n - h only
         and r["batch"] + r.get("hard", 0) <= r["n_points"], BLUE, "--D"),
        ("FMM, CPU, 12 threads", excl or rows, lambda r: r["method"] == "fmm" and r["device"] == "cpu"
         and r.get("threads", 0) in (0, 12), ORANGE, "-s"),
        ("FMM, CPU, 1 thread", excl, lambda r: r["method"] == "fmm" and r.get("threads") == 1, ORANGE, ":s"),
        ("$k$-means, GPU", rows, lambda r: r["method"] == "kmeans", INK, ":^"),
        ("LSPP, CPU, 1 thread", excl or rows, lambda r: r["method"] == "lspp", AQUA, "-*"),
    ]
    lines = []
    for lab, src, f, col, sty in series:
        g = group([r for r in src if f(r)], "n_points")
        if not g:
            continue
        ns = sorted(g)
        axes[0].plot(ns, [mean_of(g[n], "time") for n in ns], sty, color=col, label=lab, markeredgecolor="white",
                     markeredgewidth=0.5)
        axes[1].plot(ns, [mean_of(g[n], "max_ratio") for n in ns], sty, color=col, label=lab, markeredgecolor="white",
                     markeredgewidth=0.5)
        for n in ns:
            ti = np.mean([r.get("time_init", np.nan) for r in g[n]])
            lines.append(f"{lab} & {n:,} & {mean_of(g[n], 'time'):.1f} & "
                         + (f"{ti:.1f}" if np.isfinite(ti) else "--")
                         + f" & {mean_of(g[n], 'max_ratio'):.3f} \\\\")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of points $n$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("running time (s)")
    axes[1].set_ylabel(r"maximal violation $\alpha$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.4)
    fig.savefig(FIG / "scale.pdf")
    plt.close(fig)
    (TAB / "scale_rows.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- stochastic
def stochastic():
    rows = [r for r in load("stochastic_v2") if "kmeans_cost" in r]
    if not rows:
        return
    for d, fname in [("covtype", "stochastic_rows_covtype.tex"), ("gowalla_full", "stochastic_rows_gowalla.tex")]:
        lines = []
        for lam in sorted({lam_key(r["lam"]) for r in rows}):
            g = group([r for r in rows if r["data"] == d and lam_key(r["lam"]) == lam], "batch", "hard")
            for (b, h), rs in sorted(g.items()):
                c, cs = ms([r["kmeans_cost"] for r in rs])
                a, as_ = ms([r["max_ratio"] for r in rs])
                t, _ = ms([r["time"] for r in rs])
                ls = "\\infty" if math.isinf(lam) else f"{lam:g}"
                lines.append(f"${ls}$ & {b if b else 'full'} & {h if b else '--'} & {fmt_sci(c, cs)} & "
                             f"${a:.3f}\\pm{as_:.3f}$ & {t:.1f} \\\\")
            lines.append("\\midrule")
        (TAB / fname).write_text("\n".join(lines[:-1]) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
    for ax, (d, lam) in zip(axes, [("covtype", math.inf), ("gowalla_full", math.inf)]):
        for (b, h), col, sty in [((0, 0), INK, "-"), ((4096, 0), ORANGE, "-"), ((4096, 1024), BLUE, "-"),
                                 ((65536, 0), ORANGE, "--"), ((65536, 1024), BLUE, "--")]:
            rs = [r for r in rows if r["data"] == d and lam_key(r["lam"]) == lam and r["batch"] == b
                  and r["hard"] == h and r.get("curve")]
            if not rs:
                continue
            it = np.array([p[0] for p in rs[0]["curve"]])
            curves = np.array([[p[3] for p in r["curve"]][: len(it)] for r in rs])
            m, s = curves.mean(0), curves.std(0)
            lab = "full batch" if not b else f"batch {b}, " + ("memory 1024" if h else "no memory")
            ax.plot(it, m, sty, color=col, label=lab, lw=1.1)
            ax.fill_between(it, m - s, m + s, color=col, alpha=0.08, linewidth=0)
        ax.set_title(("gowalla ($10^6$ check-ins)" if d == "gowalla_full" else d) + r", $\lambda=\infty$")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"maximal violation $\alpha$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1), pad=0.4)
    fig.savefig(FIG / "stochastic.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- sensitivity and ablations
def sensitivity():
    # with many duplicated points (diabetes has 134 distinct points) the sampled radius of
    # Bateni et al. is 0 for k >= 20, so alpha is infinite for every method: rows omitted
    rows = [r for r in load("k") if "kmeans_cost" in r and math.isfinite(r["max_ratio"])]
    lines = []
    for (d, k), g in sorted(group(rows, "data", "k").items()):
        km = np.mean([r["kmeans_cost"] for r in g if r["method"] == "kmeans"] or [np.nan])
        cells = []
        for m, lam in [("kmeans", None), ("lspp", None), ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]:
            rs = [r for r in g if r["method"] == m and (lam is None or lam_key(r["lam"]) == lam)]
            cells.append(f"{mean_of(rs, 'kmeans_cost') / km:.2f} / {mean_of(rs, 'max_ratio'):.2f}" if rs else "--")
        lines.append(f"\\textsf{{{d}}} & {k} & " + " & ".join(cells) + " \\\\")
    (TAB / "k_rows.tex").write_text("\n".join(lines) + "\n")

    rows = [r for r in load("schedule") if "kmeans_cost" in r]
    lines = []
    for d, g in sorted(group(rows, "data").items()):
        for (e0, e1, it), rs in sorted(group(g, "eps0", "eps1", "iters").items()):
            kind = "fixed" if e0 == e1 else "geometric"
            c, cs = ms([r["kmeans_cost"] for r in rs])
            a, as_ = ms([r["max_ratio"] for r in rs])
            lines.append(f"\\textsf{{{d}}} & {kind} & {e0:g}$\\to${e1:g} & {it} & {fmt_sci(c, cs)} & ${a:.3f}\\pm{as_:.3f}$ \\\\")
    (TAB / "schedule_rows.tex").write_text("\n".join(lines) + "\n")

    variants = [("default", "default"), ("no_select", "no best-iterate check"), ("init_kmeans1", "init.\\ single $k$-means"),
                ("init_random", "init.\\ random points"), ("nosmooth", "no smoothing (subgradient)"),
                ("beta_low", "$\\beta=\\varepsilon/10$"), ("beta_high", "$\\beta=10\\varepsilon$"),
                ("lr_0.005", "learning rate $0.005$"), ("lr_0.05", "learning rate $0.05$")]
    ab = {v: [r for r in load(f"ablation_{v}") if "kmeans_cost" in r] for v, _ in variants}
    if any(ab.values()):
        dsets = ["adult", "bank", "skin", "covtype"]
        kmc = {d: mean_of([r for r in load("main_gpu") if r["data"] == d and r["method"] == "kmeans"], "kmeans_cost")
               for d in dsets}
        lines = []
        for lam in [10.0, math.inf]:  # one block of rows per lambda, one column per dataset
            lam_tex = "\\infty" if lam == math.inf else "10"
            lines += ["\\midrule" if lines else "",
                      f"\\multicolumn{{5}}{{l}}{{\\emph{{$\\lambda={lam_tex}$}}}} \\\\"]
            for v, lab in variants:
                cells = []
                for d in dsets:
                    rs = [r for r in ab[v] if r["data"] == d and lam_key(r["lam"]) == lam]
                    cells.append(f"{mean_of(rs, 'kmeans_cost') / kmc[d]:.2f} / {mean_of(rs, 'max_ratio'):.3f}" if rs else "--")
                lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        (TAB / "ablation_rows.tex").write_text("\n".join(l for l in lines if l) + "\n")

    rows = [r for r in load("radius_seeds") if "kmeans_cost" in r]
    if rows:
        base = {}
        for r in load("main_gpu") + [r for r in load("main") if r["method"] == "lspp"]:
            if r["data"] in ("adult", "bank", "diabetes", "shuttle") and "kmeans_cost" in r:
                r = dict(r)
                r["radius_seed"] = 100
                base.setdefault(r["data"], []).append(r)
        rows = rows + [r for v in base.values() for r in v if r["seed"] < 3 or r["method"] == "lspp"]
        lines = []
        for d in ["adult", "bank", "diabetes", "shuttle"]:
            for rs_ in [100, 101, 102, 103]:
                g = [r for r in rows if r["data"] == d and r.get("radius_seed", 100) == rs_]
                cells = []
                for m, lam in [("kmeans", None), ("lspp", None), ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]:
                    rr = [r for r in g if r["method"] == m and (lam is None or lam_key(r["lam"]) == lam)]
                    if m == "lspp" and rs_ == 100:
                        rr = rr[:1]
                    cells.append(f"{mean_of(rr, 'max_ratio'):.3f}" if rr else "--")
                lines.append(f"\\textsf{{{d}}} & {rs_} & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        (TAB / "radius_rows.tex").write_text("\n".join(lines[:-1]) + "\n")


# ---------------------------------------------------------------------------- census and exact radii
def census():
    dedup = {}
    for r in load("census"):  # some regions were run twice; keep the last run of each (region, method, seed)
        dedup[(r["region"], r["method"], r["seed"])] = r
    rows = list(dedup.values())
    if not rows:
        return
    order = ["wkmeans", "fmm_lam1.0", "fmm_lam10.0", "fmm_laminf", "fmm_lam10_bal1.0", "fmm_lam10_bal10.0",
             "fmm_lam10_bal100.0"]
    label = {"wkmeans": "weighted $k$-means", "fmm_lam1.0": "FMM, $\\lambda=1$", "fmm_lam10.0": "FMM, $\\lambda=10$",
             "fmm_laminf": "FMM, $\\lambda=\\infty$", "fmm_lam10_bal1.0": "FMM, $\\lambda=10$, $\\gamma=1$",
             "fmm_lam10_bal10.0": "FMM, $\\lambda=10$, $\\gamma=10$",
             "fmm_lam10_bal100.0": "FMM, $\\lambda=10$, $\\gamma=100$"}
    lines = []
    for reg in ["la_county", "california", "texas", "us48"]:
        g = [r for r in rows if r["region"] == reg]
        ms_ = [m for m in order if any(r["method"] == m for r in g)]
        for i, m in enumerate(ms_):
            rs = [r for r in g if r["method"] == m]
            f = lambda key: np.mean([r[key] for r in rs])
            rn = f"\\multirow{{{len(ms_)}}}{{*}}{{{REGION_NAME[reg]}}}" if i == 0 else ""
            lines.append(f"{rn} & {label[m]} & {f('max_ratio'):.2f} & {100 * f('pop_violating'):.1f} & "
                         f"{f('w_mean_dist'):.1f} & {f('max_dist'):.0f} & {f('norm_entropy'):.3f} & "
                         f"{100 * f('min_load'):.2f}--{100 * f('max_load'):.2f} \\\\")
        lines.append("\\midrule")
    (TAB / "census_rows.tex").write_text("\n".join(lines[:-1]) + "\n")


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
        for i, (m, lam) in enumerate([("kmeans", None), ("lspp", None), ("fmm", 0.1), ("fmm", 1.0), ("fmm", 10.0),
                                      ("fmm", math.inf)]):
            cells = []
            for dl in ("sample", "exact"):
                rs = [r for r in rows if r["data"] == d and r["method"] == m and r.get("delta", "sample") == dl
                      and (lam is None or lam_key(r["lam"]) == lam)]
                cells.append(f"{mean_of(rs, 'kmeans_cost') / km[dl]:.3f} & {mean_of(rs, 'max_ratio'):.3f}" if rs else "-- & --")
            name = NAMES.get(m) or FMM_LABEL(lam)
            dn = f"\\multirow{{6}}{{*}}{{\\textsf{{{d}}}}}" if i == 0 else ""
            lines.append(f"{dn} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (TAB / "exact_rows.tex").write_text("\n".join(lines[:-1]) + "\n")


if __name__ == "__main__":
    # baselines of Bateni et al. (CPU) from main.jsonl; GPU methods (k-means initialisation with
    # ten restarts) from main_gpu.jsonl; added baselines and tuned TKM from the revision runs
    rows = ([r for r in load("main") if r["method"] in ("greedy", "lspp", "icml20")]
            + [r for r in load("main_gpu") if r["method"] != "tkm"]
            + load("baselines_v2") + load("tkm_tuned"))
    front_rows = load("lspp_front")
    by = index(rows)
    main_table(by)
    compact(by)
    full_grid(by)
    pareto(by, front_rows)
    fronts(by, front_rows)
    stats(by)
    exact()
    scale()
    stochastic()
    sensitivity()
    census()
    print("LSPP runs strictly dominated (FMM front, k-means run, total):", dominance(rows, by))
    for d in DATASETS:
        for tag in [("kmeans", None), ("wkm_delta", None), ("ratio_greedy", None), ("lspp", None),
                    ("tkm", best_tkm(by, d)), ("fmm", 1.0), ("fmm", 10.0), ("fmm", math.inf)]:
            rs = by.get((d, tag))
            if rs:
                print(f"{d:9s} {str(tag):26s} n={len(rs):2d} cost={mean_of(rs, 'kmeans_cost'):.4g} "
                      f"alpha={mean_of(rs, 'max_ratio'):.3f} viol={100 * mean_of(rs, 'frac_violating'):.2f}%")
