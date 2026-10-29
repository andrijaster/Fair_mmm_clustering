"""Map figures for the paper (static PDF): Los Angeles County with the road network,
California / Texas / contiguous US heat maps, the change-of-access map and the
Gowalla world map. Run locally with .venv-maps (numpy, matplotlib, pyshp).

Inputs: results_local/{census.jsonl, census_<region>_points.npy, gowalla_map.json}
        data_maps/*.zip (Census TIGER/cartographic boundaries, Natural Earth)
"""

import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_local"
MAPS = ROOT / "data_maps"
FIG = ROOT.parent / "paper" / "figs"  # the paper repository is checked out next to experiments/
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, ROAD = "#0b0b0b", "#8a8984", "#c9c8c3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# sequential ratio scale (one hue, light -> dark) and diverging access-change scale
SEQ = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#9cc3f0", "#2a78d6", "#123f75"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#a9c9ef", "#e9e8e4", "#f4b393", "#eb6834"])
LABEL = {"wkmeans": "weighted $k$-means", "fmm_lam1.0": r"FMM, $\lambda=1$", "fmm_lam10.0": r"FMM, $\lambda=10$",
         "fmm_laminf": r"FMM, $\lambda=\infty$", "fmm_lam10_bal10.0": r"FMM, $\lambda=10$, $\gamma=10$",
         "fmm_lam10_bal1.0": r"FMM, $\lambda=10$, $\gamma=1$", "fmm_lam10_bal100.0": r"FMM, $\lambda=10$, $\gamma=100$"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.5})


def read_shp(zname):
    z = zipfile.ZipFile(MAPS / zname)
    base = [n[:-4] for n in z.namelist() if n.endswith(".shp")][0]
    return shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                            shx=io.BytesIO(z.read(base + ".shx")))


def draw_shapes(ax, shapes, color, linewidth, zorder=1, **kw):
    """All parts of all shapes as one rasterized LineCollection (keeps the PDFs small)."""
    from matplotlib.collections import LineCollection

    segs = []
    for sh in shapes:
        pts = np.asarray(sh.points)
        parts = list(sh.parts) + [len(pts)]
        segs += [pts[a:b] for a, b in zip(parts[:-1], parts[1:]) if b - a > 1]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=linewidth, zorder=zorder, rasterized=True,
                                     capstyle="round"))


def runs(region):
    rows = [json.loads(l) for l in open(RES / "census.jsonl")]
    out = {}
    for r in rows:  # later duplicates (re-runs) overwrite earlier ones
        if r["region"] == region:
            out[(r["method"], r["seed"])] = r
    return out


def haversine_min(lat, lon, c):
    lat1, lon1 = np.radians(lat)[:, None], np.radians(lon)[:, None]
    lat2, lon2 = np.radians(c[:, 0])[None], np.radians(c[:, 1])[None]
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    return d.min(1), d.argmin(1)


def voronoi_edges(ax, c, extent, n=700):
    """Draw boundaries of the nearest-facility regions (great-circle distance)."""
    lon = np.linspace(extent[0], extent[1], n)
    lat = np.linspace(extent[2], extent[3], n)
    LON, LAT = np.meshgrid(lon, lat)
    _, lab = haversine_min(LAT.ravel(), LON.ravel(), c)
    lab = lab.reshape(LAT.shape)
    edge = np.zeros_like(lab, dtype=bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    img = np.zeros(lab.shape + (4,))
    img[edge] = (0.3, 0.3, 0.3, 0.55)
    ax.imshow(img, extent=extent, origin="lower", interpolation="nearest", zorder=3, aspect="auto")


def region_panel(ax, pts, run, extent, aspect, dot=1.0, roads=None, outline=None, edges=True):
    lat, lon, pop, delta = pts.T
    c = np.array(run["centers_latlon"])
    dist, _ = haversine_min(lat, lon, c)
    ratio = dist / delta
    if roads is not None:
        for shapes, col, lw in roads:
            draw_shapes(ax, shapes, color=col, linewidth=lw, zorder=1)
    if outline is not None:
        draw_shapes(ax, outline, color=MUTED, linewidth=0.4, zorder=1)
    order = np.argsort(ratio)
    ok = ratio[order] <= 1
    sc = ax.scatter(lon[order][ok], lat[order][ok], c=ratio[order][ok], cmap=SEQ, vmin=0, vmax=1, s=dot,
                    linewidths=0, zorder=2, rasterized=True, alpha=0.85)
    ax.scatter(lon[order][~ok], lat[order][~ok], c=ORANGE, s=dot * 1.6, linewidths=0, zorder=2.5, rasterized=True)
    if edges:
        voronoi_edges(ax, c, extent)
    ax.scatter(c[:, 1], c[:, 0], marker="*", s=70, color=INK, edgecolors="white", linewidths=0.7, zorder=5)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    ax.set_title(f"{LABEL[run['method']]}\n$\\alpha={run['max_ratio']:.2f}$\n"
                 f"{100 * run['pop_violating']:.1f}% of population unfair", linespacing=1.4, fontsize=7.5)
    return sc


def colorbar_legend(fig, sc, axes):
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01, aspect=30)
    cb.set_label(r"ratio $d(\mathbf{x},C)/\delta(\mathbf{x})$", fontsize=7)
    cb.outline.set_linewidth(0.4)
    h = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label=r"not served fairly ($d>\delta$)"),
         plt.Line2D([], [], marker="*", ls="", color=INK, ms=8, label="facility (centre)"),
         plt.Line2D([], [], color="#404040", lw=0.8, label="service-area boundary")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.45, 0.10))


# ---------------------------------------------------------------------------- Los Angeles County
def la_county(methods=("wkmeans", "fmm_lam10.0", "fmm_lam10_bal10.0"), seed=0):
    pts = np.load(RES / "census_la_county_points.npy")
    rr = runs("la_county")
    roads = read_shp("tl_2020_06037_roads.zip")
    recs = roads.records()
    shp = roads.shapes()
    by = {"S1100": [], "S1200": [], "S1400": []}
    for rec, sh in zip(recs, shp):
        if rec["MTFCC"] in by:
            by[rec["MTFCC"]].append(sh)
    layers = [(by["S1400"], "#dcdbd6", 0.1), (by["S1200"], "#b9b8b2", 0.3), (by["S1100"], "#7d7c77", 0.6)]
    extent = (-118.95, -117.64, 33.70, 34.83)
    aspect = 1 / np.cos(np.radians(34.2))
    fig, axes = plt.subplots(1, len(methods), figsize=(7.4, 3.6))
    fig.subplots_adjust(wspace=0.12, bottom=0.12)
    for ax, m in zip(axes, methods):
        sc = region_panel(ax, pts, rr[(m, seed)], extent, aspect, dot=0.35, roads=layers)
    colorbar_legend(fig, sc, axes)
    fig.savefig(FIG / "map_la_county.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


def la_access_change(base="wkmeans", new="fmm_lam10.0", seed=0):
    """Hexbin map of the change in distance to the nearest facility, population weighted."""
    pts = np.load(RES / "census_la_county_points.npy")
    lat, lon, pop, delta = pts.T
    rr = runs("la_county")
    d0, _ = haversine_min(lat, lon, np.array(rr[(base, seed)]["centers_latlon"]))
    d1, _ = haversine_min(lat, lon, np.array(rr[(new, seed)]["centers_latlon"]))
    roads = read_shp("tl_2020_06037_roads.zip")
    main = [sh for rec, sh in zip(roads.records(), roads.shapes()) if rec["MTFCC"] in ("S1100", "S1200")]
    extent = (-118.95, -117.64, 33.70, 34.83)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    draw_shapes(ax, main, color=ROAD, linewidth=0.3, zorder=1)
    change = d1 - d0
    lim = np.percentile(np.abs(change), 98)
    hb = ax.hexbin(lon, lat, C=change * pop, reduce_C_function=np.sum, gridsize=60, extent=extent, cmap=DIV,
                   zorder=2, linewidths=0, mincnt=1)
    # convert population-weighted sum to population-weighted mean per hexagon
    hb_pop = ax.hexbin(lon, lat, C=pop, reduce_C_function=np.sum, gridsize=60, extent=extent, visible=False, mincnt=1)
    hb.set_array(hb.get_array() / hb_pop.get_array())
    hb.set_norm(TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    c0 = np.array(rr[(base, seed)]["centers_latlon"])
    c1 = np.array(rr[(new, seed)]["centers_latlon"])
    ax.scatter(c0[:, 1], c0[:, 0], marker="o", s=22, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4,
               label=LABEL[base])
    ax.scatter(c1[:, 1], c1[:, 0], marker="*", s=60, color=INK, edgecolors="white", linewidths=0.6, zorder=5,
               label=LABEL[new])
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect(1 / np.cos(np.radians(34.2)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("change in distance to nearest facility (km)\nnegative = closer under FMM", fontsize=7)
    ax.set_title(f"Change of access: {LABEL[new]} vs. {LABEL[base]}")
    # right: distribution of ratio vs. distance, population weighted
    ax = axes[1]
    for d, col, lab in [(d0, MUTED, LABEL[base]), (d1, BLUE, LABEL[new])]:
        r = d / delta
        o = np.argsort(r)
        cum = np.cumsum(pop[o]) / pop.sum()
        ax.plot(r[o], 1 - cum, color=col, label=lab, lw=1.5)
    ax.axvline(1, color=MUTED, ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"ratio $d(\mathbf{x},C)/\delta(\mathbf{x})$")
    ax.set_ylabel("share of population with larger ratio")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.5)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Population-weighted tail of the ratio")
    fig.tight_layout()
    fig.savefig(FIG / "map_la_access.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------------------- states and US
def state_maps(methods=("wkmeans", "fmm_lam10.0", "fmm_lam10_bal10.0"), seed=0):
    counties = read_shp("cb_2020_us_county_20m.zip")
    for region, fips, extent, lat0 in [("california", "06", (-124.6, -113.9, 32.4, 42.1), 37.0),
                                       ("texas", "48", (-106.8, -93.3, 25.7, 36.6), 31.0)]:
        if not (RES / f"census_{region}_points.npy").exists():
            continue
        pts = np.load(RES / f"census_{region}_points.npy")
        rr = runs(region)
        if not all((m, seed) in rr for m in methods):
            continue
        outl = [sh for rec, sh in zip(counties.records(), counties.shapes()) if rec["STATEFP"] == fips]
        fig, axes = plt.subplots(1, len(methods), figsize=(7.4, 3.2))
        for ax, m in zip(axes, methods):
            sc = region_panel(ax, pts, rr[(m, seed)], extent, 1 / np.cos(np.radians(lat0)), dot=0.8, outline=outl)
        colorbar_legend(fig, sc, axes)
        fig.savefig(FIG / f"map_{region}.pdf", bbox_inches="tight", dpi=400)
        plt.close(fig)


def us_map(methods=("wkmeans", "fmm_lam10.0"), seed=0):
    if not (RES / "census_us48_points.npy").exists():
        return
    rr = runs("us48")
    if not all((m, seed) in rr for m in methods):
        return
    pts = np.load(RES / "census_us48_points.npy")
    states = read_shp("cb_2020_us_state_20m.zip")
    outl = [sh for rec, sh in zip(states.records(), states.shapes()) if rec["STATEFP"] not in ("02", "15", "72")]
    extent = (-125.0, -66.5, 24.3, 49.6)
    fig, axes = plt.subplots(len(methods), 1, figsize=(7.0, 4.2 * len(methods)))
    for ax, m in zip(np.atleast_1d(axes), methods):
        sc = region_panel(ax, pts, rr[(m, seed)], extent, 1 / np.cos(np.radians(38)), dot=0.3, outline=outl,
                          edges=False)
    colorbar_legend(fig, sc, axes)
    fig.savefig(FIG / "map_us48.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


# ---------------------------------------------------------------------------- gowalla world
def gowalla_world():
    g = json.loads((RES / "gowalla_map.json").read_text())
    ll = np.array(g["latlon"])
    world = read_shp("ne_110m_admin_0_countries.zip")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5))
    for ax, (m, lab) in zip(axes, [("kmeans", "$k$-means"), ("fmm_lam10", r"FMM, $\lambda=10$")]):
        draw_shapes(ax, world.shapes(), color=ROAD, linewidth=0.4, zorder=1)
        rec = g["methods"][m]
        ratio = np.array(rec["ratio"])
        ax.hexbin(ll[:, 1], ll[:, 0], C=ratio, reduce_C_function=np.max, gridsize=(90, 30),
                  extent=(-180, 180, -60, 80), cmap=SEQ, vmin=0, vmax=2, zorder=2, linewidths=0, mincnt=1)
        bad = ratio > 1
        ax.scatter(ll[bad, 1], ll[bad, 0], s=0.4, color=ORANGE, zorder=3, rasterized=True, linewidths=0)
        c = np.array(rec["centers_latlon"])
        ax.scatter(c[:, 1], c[:, 0], marker="*", s=70, color=INK, edgecolors="white", linewidths=0.7, zorder=5)
        ax.set_xlim(-180, 180); ax.set_ylim(-58, 80); ax.set_aspect(1.2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(MUTED)
        ax.set_title(f"{lab}: $\\alpha={rec['max_ratio']:.2f}$, {100 * rec['frac_violating']:.1f}% of check-ins outside $\\delta$")
    fig.tight_layout()
    fig.savefig(FIG / "map_gowalla.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    todo = sys.argv[1:] or ["la", "access", "states", "us", "gowalla"]
    if "la" in todo:
        la_county()
    if "access" in todo:
        la_access_change()
    if "states" in todo:
        state_maps()
    if "us" in todo:
        us_map()
    if "gowalla" in todo:
        gowalla_world()
