"""Build the interactive facility-placement explorer (single HTML file with the data
embedded) from the census case-study results. Run with .venv-maps.

output: interactive/explorer.html  (template: interactive/template.html)
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import shapefile

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_local"
MAPS = ROOT / "data_maps"
OUT = ROOT / "interactive"

METHODS = ["wkmeans", "fmm_lam1.0", "fmm_lam10.0", "fmm_laminf", "fmm_lam10_bal1.0", "fmm_lam10_bal10.0",
           "fmm_lam10_bal100.0"]
REGIONS = {"la_county": ("Los Angeles County", "06", "037"), "california": ("California", "06", None),
           "texas": ("Texas", "48", None)}


def read_shp(zname):
    z = zipfile.ZipFile(MAPS / zname)
    base = [n[:-4] for n in z.namelist() if n.endswith(".shp")][0]
    return shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                            shx=io.BytesIO(z.read(base + ".shx")))


def lines_of(shapes, step=1, nd=4):
    out = []
    for sh in shapes:
        pts = np.asarray(sh.points)
        parts = list(sh.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            seg = pts[a:b]
            if len(seg) > 2 and step > 1:
                seg = np.vstack([seg[::step], seg[-1:]])
            seg = np.round(seg, nd)
            out.append(seg.ravel().tolist())
    return out


def main():
    runs = {}
    for l in open(RES / "census.jsonl"):
        r = json.loads(l)
        runs[(r["region"], r["method"], r["seed"])] = r
    counties = read_shp("cb_2020_us_county_500k.zip")
    crecs = counties.records()
    data = {"regions": {}}
    for reg, (name, st, co) in REGIONS.items():
        pts = np.load(RES / f"census_{reg}_points.npy")  # lat, lon, pop, delta(km)
        k = len(runs[(reg, "wkmeans", 0)]["centers_latlon"])
        centers = {m: np.round(np.array(runs[(reg, m, 0)]["centers_latlon"]), 5).tolist()
                   for m in METHODS if (reg, m, 0) in runs}
        outl = [sh for rec, sh in zip(crecs, counties.shapes())
                if rec["STATEFP"] == st and (co is None or rec["COUNTYFP"] == co)]
        entry = {
            "name": name, "k": k,
            "lat": np.round(pts[:, 0], 5).tolist(), "lon": np.round(pts[:, 1], 5).tolist(),
            "pop": pts[:, 2].astype(int).tolist(), "delta": np.round(pts[:, 3], 3).tolist(),
            "centers": centers, "outline": lines_of(outl, step=3 if co is None else 1),
        }
        if reg == "la_county":
            roads = read_shp("tl_2020_06037_roads.zip")
            sel = {"S1100": [], "S1200": []}
            for rec, sh in zip(roads.records(), roads.shapes()):
                if rec["MTFCC"] in sel:
                    sel[rec["MTFCC"]].append(sh)
            entry["roads"] = {"primary": lines_of(sel["S1100"], step=2), "secondary": lines_of(sel["S1200"], step=3)}
        data["regions"][reg] = entry
        print(reg, len(pts), "points", {m: len(c) for m, c in centers.items()})
    tpl = (OUT / "template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (OUT / "explorer.html").write_text(html)
    print("explorer.html", round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
