"""Case study: individually fair facility placement on US Census 2020 block-group
population centres (weighted by population).

Points are mapped to 3-D Cartesian coordinates on the Earth sphere (km), so that
Euclidean (chord) distances match great-circle distances to <0.1% at these scales.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fairmmm import _weighted as fx

ROOT = Path(__file__).resolve().parent
R_EARTH = 6371.0
REGIONS = {  # name: (state FIPS codes or None for contiguous US, k, county FIPS or None)
    "la_county": (["06"], 25, ["037"]),
    "california": (["06"], 20, None),
    "texas": (["48"], 20, None),
    "us48": (None, 100, None),
}
NON_CONTIGUOUS = {"02", "15", "72"}


def load(states, counties=None):
    df = pd.read_csv(ROOT / "data/raw/cenpop_bg.txt", dtype={"STATEFP": str, "COUNTYFP": str}, encoding="utf-8-sig")
    df = df[df.POPULATION > 0]
    df = df[~df.STATEFP.isin(NON_CONTIGUOUS)] if states is None else df[df.STATEFP.isin(states)]
    if counties is not None:
        df = df[df.COUNTYFP.isin(counties)]
    lat, lon = np.radians(df.LATITUDE.values), np.radians(df.LONGITUDE.values)
    xyz = R_EARTH * np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    return xyz, df.POPULATION.values.astype(np.float64), df[["LATITUDE", "LONGITUDE"]].values


def to_latlon(c):
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return np.c_[np.degrees(np.arcsin(c[:, 2])), np.degrees(np.arctan2(c[:, 1], c[:, 0]))]


METHODS = [("wkmeans", None, 0.0)] + [(f"fmm_lam{l}", l, 0.0) for l in [1.0, 10.0, math.inf]] + [
    (f"fmm_lam10_bal{a}", 10.0, a) for a in [1.0, 10.0, 100.0]]

if __name__ == "__main__":
    regions = sys.argv[1:] or list(REGIONS)
    out = ROOT / "results/census.jsonl"
    out.parent.mkdir(exist_ok=True)
    for reg in regions:
        states, k, counties = REGIONS[reg]
        xyz, w, latlon = load(states, counties)
        x = torch.tensor(xyz, device="cuda")
        wt = torch.tensor(w, device="cuda")
        t = time.time()
        delta = fx.weighted_thresholds(x, wt, k, chunk=256)
        print(reg, len(w), "points, thresholds in %.1fs" % (time.time() - t), flush=True)
        np.save(ROOT / f"results/census_{reg}_points.npy", np.c_[latlon, w, delta.cpu().numpy()])
        for seed in range(5):
            for name, lam, alpha in METHODS:
                t = time.time()
                if lam is None:
                    c = fx.weighted_kmeans(x, wt, k, seed)
                else:
                    c = fx.fmm_weighted(x, wt, delta, k, lam=lam, alpha=alpha, seed=seed)
                secs = time.time() - t
                r = fx.evaluate_weighted(x, wt, c, delta)
                r.update(region=reg, k=k, n=len(w), method=name, seed=seed, time=secs,
                         centers_latlon=to_latlon(c.cpu().numpy()).tolist())
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(reg, seed, name, {kk: round(v, 4) for kk, v in r.items() if isinstance(v, float)}, flush=True)
