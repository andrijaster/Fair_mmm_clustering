"""Figure for Section 3: smooth maxima of (t, 0) and their gradients, and the worst-case error
of each smoothing at equal smoothness. Run with the package installed (fairmmm)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

from fairmmm._smooth import E1_AT_1, KINDS, reg_max, reg_weights

OUT = Path(__file__).resolve().parent.parent / "paper" / "figs" / "smoothing.pdf"
BLUE, ORANGE, AQUA, INK, MUTED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984", "#6250d6"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "pdf.fonttype": 42})
STYLE = {"lse": (BLUE, "log-sum-exp"), "sparsemax": (AQUA, "sparsemax"), "entmax15": (VIOLET, "1.5-entmax"),
         "bump": (ORANGE, "BumpMax")}

t = torch.linspace(-3, 3, 601, dtype=torch.float64)
x = torch.stack([t, torch.zeros_like(t)], 1)
fig, axes = plt.subplots(1, 3, figsize=(5.15, 1.75))
for kind, (col, lab) in STYLE.items():
    if kind == "lse":
        val = torch.logsumexp(x, 1)
        grad = torch.softmax(x, 1)[:, 0]
    else:
        val = reg_max(x, 1.0, kind, dim=1, iters=80)
        grad = reg_weights(x, 1.0, kind, dim=1, iters=80)[:, 0]
    axes[0].plot(t, val, color=col, lw=1.4, label=lab)
    axes[1].plot(t, grad, color=col, lw=1.4)
axes[0].plot(t, x.max(1).values, color=INK, lw=0.8, ls="--", label="max")
axes[0].set_title(r"$F_\mu(t,0)$, $\mu=1$")
axes[1].set_title(r"$p^\star_1=\partial F_\mu/\partial v_1$")
for ax in axes[:2]:
    ax.set_xlabel("$t$")
axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.5)

# worst-case error times the Lipschitz constant of the gradient (independent of mu)
m = np.logspace(0.3, 7, 200)
prod = {"lse": np.log(m) / 2, "sparsemax": 0.5 * (1 - 1 / m), "entmax15": 4 / 3 * (1 - 1 / np.sqrt(m)),
        "bump": None}
for kind, (col, lab) in STYLE.items():
    if kind == "bump":  # h(1) - m h(1/m) with h(p) = e E_1(1 - ln p), times the Lipschitz constant 4/e
        from scipy.special import exp1
        y = (math.e * E1_AT_1 - m * math.e * exp1(1 + np.log(m))) * 4 / math.e
    else:
        y = prod[kind]
    axes[2].plot(m, y, color=col, lw=1.4)
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_yticks([0.3, 1, 3, 10])
axes[2].set_yticklabels(["0.3", "1", "3", "10"])
axes[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[2].set_xticks([1e1, 1e3, 1e5, 1e7])
axes[2].set_xlabel("number of entries $m$")
axes[2].set_title("worst error $\\times$ smoothness")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT, bbox_inches="tight")
print(OUT, {k: KINDS[k][0] for k in KINDS})
