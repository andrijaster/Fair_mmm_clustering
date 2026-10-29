"""Extension used in the facility-location case study: population weights and a
load-balance (uniformity) penalty.

    min_C  sum_x w_x min_j ||x-c_j||^2 / W  +  lam * Phi_eps(C)  +  alpha * P_beta(C)

    P_beta(C) = sum_j | D_j - 1/k |,   D_j = sum_x w_x softmin_j(beta * r_x) / W

P_beta is the non-uniformity penalty of the original manuscript, here driven by the
population share served by each centre.
"""

import math

import torch

from fmm import _schedule, kmeanspp, log_mean_exp, pairwise_dist, soft_ratio


@torch.no_grad()
def weighted_thresholds(x, w, k, chunk=512):
    """delta(x) = radius of the smallest ball around x holding >= W/k population."""
    target = w.sum() / k
    out = torch.empty(x.shape[0], device=x.device)
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i : i + chunk], x)
        ds, order = d.sort(dim=1)
        cum = w[order].cumsum(1)
        pos = (cum < target).sum(1).clamp_max(x.shape[0] - 1)
        out[i : i + chunk] = ds.gather(1, pos[:, None]).squeeze(1)
    return out.clamp_min(1e-6)


@torch.no_grad()
def weighted_kmeans(x, w, k, seed, iters=200, n_init=10):
    gen = torch.Generator().manual_seed(seed)
    best, best_cost = None, math.inf
    for _ in range(n_init):
        c = _weighted_lloyd(x, w, kmeanspp(x, k, gen), iters)
        cost = (w * pairwise_dist(x, c).min(1).values.pow(2)).sum().item()
        if cost < best_cost:
            best, best_cost = c, cost
    return best


@torch.no_grad()
def _weighted_lloyd(x, w, c, iters):
    k = c.shape[0]
    for _ in range(iters):
        a = pairwise_dist(x, c).argmin(1)
        num = torch.zeros_like(c).index_add_(0, a, x * w[:, None])
        den = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, a, w)
        new = torch.where(den[:, None] > 0, num / den.clamp_min(1e-12)[:, None], c)
        if (new - c).norm() < 1e-7:
            c = new
            break
        c = new
    return c


@torch.no_grad()
def evaluate_weighted(x, w, c, delta):
    d = pairwise_dist(x, c)
    m, a = d.min(1)
    W = w.sum()
    load = torch.zeros(c.shape[0], device=x.device, dtype=x.dtype).index_add_(0, a, w) / W
    p = load[load > 0]
    return {
        "w_cost": (w * m * m).sum().item(),
        "max_ratio": (m / delta).max().item(),
        "pop_violating": (w * (m > delta)).sum().item() / W.item(),
        "w_mean_dist": (w * m).sum().item() / W.item(),
        "max_dist": m.max().item(),
        "norm_entropy": (-(p * p.log()).sum() / math.log(c.shape[0])).item(),
        "max_load": load.max().item(),
        "min_load": load.min().item(),
    }


def fmm_weighted(x, w, delta, k, lam=10.0, alpha=0.0, iters=3000, lr=None, eps=(2.0, 200.0),
                 beta=(2.0, 200.0), seed=0, eval_every=50):
    torch.manual_seed(seed)
    W = w.sum()
    wn = w / W
    c0 = weighted_kmeans(x, w, k, seed)
    scale = x.std(0).mean().item()
    c = c0.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr or 0.02 * scale)
    cost_scale = (wn * pairwise_dist(x, c0).min(1).values.pow(2)).sum().item()
    pure = math.isinf(lam)  # pure min-max-min: cost term dropped

    def hard(cc):
        with torch.no_grad():
            d = pairwise_dist(x, cc)
            m, a = d.min(1)
            load = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, a, wn)
            fair = (m / delta).max().item()
            cost = 0.0 if pure else (wn * m * m).sum().item() / cost_scale
            return cost + (fair if pure else lam * fair) + alpha * (load - 1.0 / k).abs().sum().item()

    best_c, best_val = c0.clone(), hard(c0)
    for it in range(iters):
        t = it / max(iters - 1, 1)
        e, b = _schedule(*eps, t), _schedule(*beta, t)
        opt.zero_grad(set_to_none=True)
        s, r = soft_ratio(x, c, delta, b)
        fair = log_mean_exp(s, e)
        # cost normalised by the k-means solution so that lam has the same meaning across regions
        cost = (wn * r.mul(delta[:, None]).min(1).values.pow(2)).sum() / cost_scale
        loss = fair if pure else cost + lam * fair
        if alpha > 0:
            share = (wn[:, None] * torch.softmax(-b * r, dim=1)).sum(0)
            loss = loss + alpha * (share - 1.0 / k).abs().sum()
        loss.backward()
        opt.step()
        if (it + 1) % eval_every == 0 or it == iters - 1:
            v = hard(c.detach())
            if v < best_val:
                best_val, best_c = v, c.detach().clone()
    return best_c
