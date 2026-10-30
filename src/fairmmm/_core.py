"""Smooth min-max-min individually fair clustering (FMM).

Objective (variant (b) of the paper):

    min_C  (1/n) sum_x min_j ||x - c_j||^2  +  lam * Phi_eps(C)

    Phi_eps(C) = (1/eps) log( (1/n) sum_x exp(eps * s_beta(x, C)) )
    s_beta(x,C) = -(1/beta) log( (1/k) sum_j exp(-beta * ||x - c_j|| / delta(x)) )

Phi_eps is the Tsoukalas-Rustem smoothing of max_x min_j ||x-c_j||/delta(x)
(log-mean-exp instead of log-sum-exp; same minimizer, Phi <= true max + ln(k)/beta).
lam = inf means the pure min-max-min problem (cost term dropped).
"""

import math
import time
from dataclasses import dataclass, field

import torch

from ._smooth import KINDS, reg_max, reg_min


# ----------------------------------------------------------------------------
# thresholds and evaluation


def pairwise_dist(x, c):
    """Euclidean distances, numerically safe (no NaN gradient at 0)."""
    d2 = (x * x).sum(1, keepdim=True) - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return d2.clamp_min(1e-12).sqrt()


@torch.no_grad()
def compute_thresholds(x, k, n_sample=500, seed=100, chunk=1 << 16):
    """delta(x) = distance to the ceil(n_sample/k)-th closest of n_sample random points.

    Same estimator as Bateni et al. (2024): with n_sample=500, k=10 it is the
    50-th closest sampled point, i.e. an estimate of the n/k-th neighbour radius.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:n_sample].to(x.device)
    s = x[idx]
    rank = math.ceil(n_sample / k)
    out = torch.empty(x.shape[0], device=x.device)
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i : i + chunk], s)
        out[i : i + chunk] = d.kthvalue(rank, dim=1).values
    return out


@torch.no_grad()
def exact_thresholds(x, k, chunk=1024):
    """delta(x) = radius of the smallest ball around x holding ceil(n/k) points (x included)."""
    rank = math.ceil(x.shape[0] / k)
    out = torch.empty(x.shape[0], device=x.device)
    for i in range(0, x.shape[0], chunk):
        out[i : i + chunk] = torch.cdist(x[i : i + chunk], x).kthvalue(rank, dim=1).values
    return out.clamp_min(1e-6)


@torch.no_grad()
def evaluate(x, c, delta, chunk=1 << 16):
    x = x.to(torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64, device=x.device)
    delta = delta.to(torch.float64)
    cost, maxr, viol, sumr = 0.0, 0.0, 0, 0.0
    counts = torch.zeros(c.shape[0], dtype=torch.float64, device=x.device)
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i : i + chunk], c)
        m, a = d.min(1)
        r = m / delta[i : i + chunk]
        cost += (m * m).sum().item()
        maxr = max(maxr, r.max().item())
        viol += (r > 1.0).sum().item()
        sumr += r.sum().item()
        counts += torch.bincount(a, minlength=c.shape[0]).to(torch.float64)
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum().item() / math.log(c.shape[0])
    n = x.shape[0]
    return {
        "kmeans_cost": cost,
        "max_ratio": maxr,
        "frac_violating": viol / n,
        "mean_ratio": sumr / n,
        "norm_entropy": ent,
    }


# ----------------------------------------------------------------------------
# k-means (initialisation and the unfair baseline)


@torch.no_grad()
def kmeanspp(x, k, gen):
    n = x.shape[0]
    c = [x[torch.randint(n, (1,), generator=gen, device="cpu").item()]]
    d2 = ((x - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum()
        i = torch.multinomial(p.cpu(), 1, generator=gen).item()
        c.append(x[i])
        d2 = torch.minimum(d2, ((x - x[i]) ** 2).sum(1))
    return torch.stack(c)


@torch.no_grad()
def lloyd(x, c, iters=100, tol=1e-6):
    for _ in range(iters):
        a = pairwise_dist(x, c).argmin(1)
        new = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=c.shape[0]).to(x.dtype)
        empty = cnt == 0
        new = new / cnt.clamp_min(1)[:, None]
        new[empty] = c[empty]
        shift = (new - c).norm()
        c = new
        if shift < tol:
            break
    return c


def kmeans(x, k, seed, n_init=10):
    """Best of n_init k-means++/Lloyd runs (as sklearn's classic default)."""
    gen = torch.Generator().manual_seed(seed)
    best, best_cost = None, math.inf
    for _ in range(n_init):
        c = lloyd(x, kmeanspp(x, k, gen))
        cost = pairwise_dist(x, c).min(1).values.pow(2).sum().item()
        if cost < best_cost:
            best, best_cost = c, cost
    return best


# ----------------------------------------------------------------------------
# objective pieces


def soft_ratio(x, c, delta, beta):
    """s_beta(x, C): smooth min over centres of ||x - c_j|| / delta(x)."""
    r = pairwise_dist(x, c) / delta[:, None]
    return -(torch.logsumexp(-beta * r, dim=1) - math.log(c.shape[0])) / beta, r


def log_mean_exp(v, eps):
    return (torch.logsumexp(eps * v, dim=0) - math.log(v.shape[0])) / eps


@dataclass
class FMMConfig:
    lam: float = 1.0  # math.inf -> pure min-max-min
    eps0: float = 2.0  # outer temperature, annealed geometrically eps0 -> eps1
    eps1: float = 200.0
    beta0: float = 2.0  # inner temperature
    beta1: float = 200.0
    iters: int = 2000
    lr: float = 0.02
    batch: int = 0  # 0 = full batch
    hard_buffer: int = 0  # size of the hard-point memory (stochastic only)
    eval_every: int = 50
    select_best: bool = True  # return the best iterate under the exact objective (else the last one)
    hard: bool = False  # ablation: no smoothing, (sub)gradient of the exact max-min objective
    smoothing: str = "lse"  # "lse" (log-sum-exp), or a regularised max: "sparsemax", "entmax15", "bump"
    match_error: bool = False  # regularised max: temperatures giving the error bound of log-sum-exp
    init: str = "kmeans"  # "kmeans" (best of 10), "kmeans1" (single run) or "random" (k data points)
    seed: int = 0
    log: list = field(default_factory=list)
    time_init: float = 0.0


def _schedule(a, b, t):
    return a * (b / a) ** t


def fmm(x, delta, k, cfg: FMMConfig, init=None):
    """Optimise the smooth objective with Adam; returns the best iterate under
    the *hard* objective cost/n + lam * max_ratio (evaluated every eval_every)."""
    torch.manual_seed(cfg.seed)
    n = x.shape[0]
    t_init = time.time()
    if init is not None:
        c0 = torch.as_tensor(init, device=x.device, dtype=x.dtype)
    elif cfg.init == "kmeans":
        c0 = kmeans(x, k, cfg.seed)
    elif cfg.init == "kmeans1":
        c0 = kmeans(x, k, cfg.seed, n_init=1)
    elif cfg.init == "random":
        g = torch.Generator().manual_seed(cfg.seed)
        c0 = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    else:
        raise ValueError(cfg.init)
    if x.is_cuda:
        torch.cuda.synchronize()
    cfg.time_init = time.time() - t_init
    c = c0.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=cfg.lr)
    pure = math.isinf(cfg.lam)
    gen = torch.Generator(device=x.device).manual_seed(cfg.seed)

    def hard_obj(cc):
        with torch.no_grad():
            d = pairwise_dist(x, cc)
            m = d.min(1).values
            cost = m.pow(2).mean().item()
            maxr = (m / delta).max().item()
        return (maxr if pure else cost + cfg.lam * maxr), cost, maxr

    best_c, (best_val, _, _) = c0.clone(), hard_obj(c0)
    hard_idx = torch.empty(0, dtype=torch.long, device=x.device)
    t0 = time.time()
    for it in range(cfg.iters):
        t = it / max(cfg.iters - 1, 1)
        eps = _schedule(cfg.eps0, cfg.eps1, t)
        beta = _schedule(cfg.beta0, cfg.beta1, t)
        opt.zero_grad(set_to_none=True)
        if cfg.hard:  # ablation: exact objective, autograd gives a subgradient
            d = pairwise_dist(x, c)
            m = d.min(1).values
            fair = (m / delta).max()
            cost = m.pow(2).mean()
        elif cfg.smoothing != "lse" and (cfg.batch > 0 and cfg.batch < n):
            raise ValueError("regularised maxima are implemented for the full batch only")
        elif cfg.smoothing != "lse":  # regularised max and min with temperatures 1/eps and 1/beta (full batch)
            # same temperatures as log-sum-exp (same smoothness), or scaled so that the error bounds
            # psi(1) agree with the log-sum-exp bounds ln n / eps and ln k / beta
            mo, mi = 1.0 / eps, 1.0 / beta
            if cfg.match_error:
                cst = KINDS[cfg.smoothing][0]
                mo, mi = mo * math.log(n) / cst, mi * math.log(k) / cst
            r = pairwise_dist(x, c) / delta[:, None]
            fair = reg_max(reg_min(r, mi, cfg.smoothing, dim=1), mo, cfg.smoothing, dim=0)
            cost = r.mul(delta[:, None]).min(1).values.pow(2).mean()
        elif cfg.batch <= 0 or cfg.batch >= n:
            s, r = soft_ratio(x, c, delta, beta)
            fair = log_mean_exp(s, eps)
            cost = r.mul(delta[:, None]).min(1).values.pow(2).mean()
        else:
            b = torch.randint(n, (cfg.batch,), device=x.device, generator=gen)
            if cfg.hard_buffer > 0:  # keep H and B duplicate-free and disjoint
                b = torch.unique(b)
            if cfg.hard_buffer > 0 and hard_idx.numel() > 0:
                b = b[~torch.isin(b, hard_idx)]
            sb, rb = soft_ratio(x[b], c, delta[b], beta)
            cost = rb.mul(delta[b][:, None]).min(1).values.pow(2).mean()
            if cfg.hard_buffer > 0:
                # stratified estimator: exact over the hard set H, Monte Carlo over the rest
                sh, _ = soft_ratio(x[hard_idx], c, delta[hard_idx], beta) if hard_idx.numel() else (sb[:0], None)
                w_rest = math.log(max(n - hard_idx.numel(), 1) / max(b.numel(), 1))
                terms = torch.cat([eps * sh, eps * sb + w_rest])
                fair = (torch.logsumexp(terms, 0) - math.log(n)) / eps
                with torch.no_grad():  # update the memory with the worst points seen
                    cand = torch.cat([hard_idx, b])
                    vals = torch.cat([sh, sb]).detach()
                    top = vals.topk(min(cfg.hard_buffer, vals.numel())).indices
                    hard_idx = cand[top]
            else:
                fair = log_mean_exp(sb, eps)
        loss = fair if pure else cost + cfg.lam * fair
        loss.backward()
        opt.step()
        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            val, hc, hm = hard_obj(c.detach())
            cfg.log.append((it + 1, time.time() - t0, hc, hm, val))
            if val < best_val:
                best_val, best_c = val, c.detach().clone()
    return best_c if cfg.select_best else c.detach()


def tilted_kmeans(x, k, t_tilt, iters=2000, lr=0.02, seed=0, select_best=False, eval_every=50):
    """Re-implementation of tilted k-means (Zhu et al. 2024) by gradient descent:
    min_C (1/t) log mean_x exp(t * min_j ||x - c_j||^2). With select_best, the iterate with the
    lowest value of this (tilted) objective is returned, checked every eval_every iterations."""
    c = kmeans(x, k, seed).clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)

    def obj(cc):
        with torch.no_grad():
            m = pairwise_dist(x, cc).min(1).values.pow(2)
            return ((torch.logsumexp(t_tilt * m, 0) - math.log(x.shape[0])) / t_tilt).item()

    best_c, best_v = c.detach().clone(), obj(c)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        m = pairwise_dist(x, c).min(1).values.pow(2)
        loss = (torch.logsumexp(t_tilt * m, 0) - math.log(x.shape[0])) / t_tilt
        loss.backward()
        opt.step()
        if select_best and ((it + 1) % eval_every == 0 or it == iters - 1):
            v = obj(c)
            if v < best_v:
                best_v, best_c = v, c.detach().clone()
    return best_c if select_best else c.detach()


@torch.no_grad()
def ratio_greedy(x, delta, k):
    """Farthest-first traversal in the ratio metric (a priority k-center heuristic in the
    spirit of Gonzalez 1985 / Plesnik 1987): start at the point with the smallest radius, then
    repeatedly open a centre at the point with the largest ratio d(x_i, C) / delta_i."""
    first = int(delta.argmin())
    c = [x[first]]
    m = pairwise_dist(x, x[first : first + 1]).squeeze(1)
    for _ in range(1, k):
        i = int((m / delta).argmax())
        c.append(x[i])
        m = torch.minimum(m, pairwise_dist(x, x[i : i + 1]).squeeze(1))
    return torch.stack(c)
