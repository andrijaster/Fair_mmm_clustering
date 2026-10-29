"""Illustration for Section 3: k-means versus FMM on a small 2-D example with dense
clusters and a sparse background. Run with the package installed (fairmmm)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from fairmmm import FairMMM, fairness_radii

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs" / "toy.pdf"
INK, ORANGE, MUTED = "#0b0b0b", "#eb6834", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq", ["#e3eefb", "#9cc3f0", "#2a78d6", "#123f75"])
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "pdf.fonttype": 42})

# a dense "city core" (uniform disc) inside a sparse surrounding area
rng = np.random.default_rng(0)
ang, rad = rng.uniform(0, 2 * np.pi, 360), np.sqrt(rng.uniform(0, 1, 360))
core = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
sparse = rng.uniform(-6, 6, (140, 2))
X = np.concatenate([core, sparse[np.linalg.norm(sparse, axis=1) > 1.5]])
k = 5
radii = fairness_radii(X, k, method="exact", device="cpu")
models = [("$k$-means", FairMMM(k, lam=0.0, radii=radii, n_iter=800, device="cpu", random_state=1).fit(X)),
          (r"FMM, $\lambda=\infty$", FairMMM(k, lam=math.inf, radii=radii, n_iter=800, device="cpu",
                                             random_state=1).fit(X))]

fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.45), sharex=True, sharey=True)
for ax, (title, m) in zip(axes, models):
    d = np.linalg.norm(X[:, None] - m.cluster_centers_[None], axis=2).min(1)
    r = d / radii
    ok = r <= 1
    sc = ax.scatter(X[ok, 0], X[ok, 1], c=r[ok], cmap=SEQ, vmin=0, vmax=1, s=6, linewidths=0)
    ax.scatter(X[~ok, 0], X[~ok, 1], color=ORANGE, s=9, linewidths=0)
    i = int(r.argmax())  # the worst-served point: its fairness ball and its nearest centre
    j = int(np.linalg.norm(m.cluster_centers_ - X[i], axis=1).argmin())
    ax.add_patch(plt.Circle(X[i], radii[i], fill=False, color=INK, lw=0.8, ls="--"))
    ax.plot([X[i, 0], m.cluster_centers_[j, 0]], [X[i, 1], m.cluster_centers_[j, 1]], color=INK, lw=0.7)
    ax.scatter(*m.cluster_centers_.T, marker="*", s=70, color=INK, edgecolors="white", linewidths=0.7, zorder=5)
    ax.set_title(f"{title}\n$\\alpha={m.max_ratio_:.2f}$, {100 * (~ok).mean():.0f}% of points with $r_i>1$",
                 linespacing=1.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
cb.set_label(r"$r_i=d(\mathbf{x}_i,C)/\delta_i$ (orange: $r_i>1$)", fontsize=7)
fig.savefig(OUT, bbox_inches="tight")
print(OUT, [round(m.max_ratio_, 3) for _, m in models])
