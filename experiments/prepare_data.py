"""Download and preprocess the benchmark datasets of Bateni et al. (AISTATS 2024).

Each dataset is standardized (zero mean, unit variance per dimension) and saved
as data/<name>.npy (float32) and data/<name>.tsv (for the Google baseline runner).
Column choices follow Mahabadi & Vakilian (ICML 2020) / Backurs et al. (2019).
"""

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"
RNG = np.random.default_rng(0)


def fetch(url, name):
    path = RAW / name
    if not path.exists():
        print("downloading", url, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            path.write_bytes(r.read())
    return path


def save(name, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x).any(axis=1)]
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x = ((x - x.mean(axis=0)) / std).astype(np.float32)
    np.save(DATA / f"{name}.npy", x)
    np.savetxt(DATA / f"{name}.tsv", x, delimiter="\t", fmt="%.6f")
    print(f"{name}: n={x.shape[0]} d={x.shape[1]}", flush=True)


def adult():
    p = fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")
    df = pd.read_csv(p, header=None, skipinitialspace=True)
    df = df.dropna(how="all")
    # age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
    save("adult", df[[0, 2, 4, 10, 11, 12]].values)


def bank():
    p = fetch("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", "bank_marketing.zip")
    with zipfile.ZipFile(p) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read("bank.zip")))
        df = pd.read_csv(inner.open("bank-full.csv"), sep=";")
    save("bank", df[["age", "balance", "duration"]].values)


def diabetes():
    p = fetch(
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        "diabetes.zip",
    )
    with zipfile.ZipFile(p) as z:
        member = [m for m in z.namelist() if m.endswith("diabetic_data.csv")][0]
        df = pd.read_csv(z.open(member))
    # age is given as a decade bracket "[70-80)"; use the bracket midpoint
    age = df["age"].str.extract(r"\[(\d+)-(\d+)\)").astype(float).mean(axis=1)
    save("diabetes", np.c_[age.values, df["time_in_hospital"].values])


def gowalla():
    p = fetch("https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz", "gowalla.txt.gz")
    with gzip.open(p, "rt") as f:
        df = pd.read_csv(f, sep="\t", header=None, usecols=[2, 3])
    x = df.values.astype(np.float64)
    x = x[(np.abs(x[:, 0]) <= 90) & (np.abs(x[:, 1]) <= 180)]
    # the full set is used for the scalability experiment
    save("gowalla_full", x)
    save("gowalla", x[RNG.choice(len(x), 100_000, replace=False)])


def skin():
    p = fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt", "skin.txt")
    df = pd.read_csv(p, sep="\t", header=None)
    save("skin", df[[0, 1, 2]].values)  # B, G, R (class label dropped)


def shuttle():
    p = fetch("https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip", "shuttle.zip")
    import subprocess

    parts = []
    with zipfile.ZipFile(p) as z:
        for m in z.namelist():
            if m.endswith(("shuttle.trn", "shuttle.tst")):
                parts.append(pd.read_csv(z.open(m), sep=" ", header=None))
            elif m.endswith("shuttle.trn.Z"):  # unix compress; gzip can read it
                raw = subprocess.run(["gzip", "-dc"], input=z.read(m), capture_output=True, check=True).stdout
                parts.append(pd.read_csv(io.BytesIO(raw), sep=" ", header=None))
    df = pd.concat(parts)
    save("shuttle", df.iloc[:, :9].values)


def covtype():
    from sklearn.datasets import fetch_covtype

    save("covtype", fetch_covtype(data_home=str(RAW)).data)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
    for n in names:
        try:
            globals()[n]()
        except Exception as e:  # keep going, report at the end
            print(f"FAILED {n}: {e!r}", flush=True)
