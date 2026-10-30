"""Regularised maxima used as alternatives to log-sum-exp in the surrogate.

For a separable strictly convex regulariser psi on [0, 1] with psi(0) = 0,

    F_mu(x) = max_{p in simplex} <p, x> - sum_i psi(p_i) + psi(1)

is a smooth upper approximation of max_i x_i with gradient p*(x). Writing
psi'(p) = mu * h(p) with h the inverse of a link phi, the maximiser is
p_i = phi((x_i - tau) / mu) with tau fixed by sum_i p_i = 1, and

    F_mu(x) = tau + mu * sum_i Phi((x_i - tau) / mu) + psi(1),    Phi' = phi.

    kind        phi(u), u > 0          Phi(u)                      psi(1) / mu
    sparsemax   u                      u^2 / 2                     1/2
    entmax15    u^2 / 4                u^3 / 12                    4/3
    bump        exp(1 - 1/u)           e u e^{-1/u} (1 - R(1/u))   e E_1(1)

(phi = 0 for u <= 0). sparsemax and entmax15 are the value functions of
sparsemax (Martins & Astudillo 2016) and 1.5-entmax (Peters et al. 2019); bump uses
the flat function exp(1 - 1/u), so that F is infinitely differentiable and still
equals max x exactly when the largest entry leads the second by at least mu.
R(z) = z e^z E_1(z), evaluated with the rational approximation 5.1.56 of
Abramowitz & Stegun (absolute error below 5e-8 for z >= 1).
"""

import math

import torch

E1_AT_1 = 0.21938393439552029  # exponential integral E_1(1)
KINDS = {  # psi(1) / mu and the largest argument u at which phi(u) = 1
    "sparsemax": (0.5, 1.0),
    "entmax15": (4.0 / 3.0, 2.0),
    "bump": (math.e * E1_AT_1, 1.0),
}


def _phi(u, kind):
    if kind == "sparsemax":
        return u.clamp_min(0)
    if kind == "entmax15":
        return u.clamp_min(0).pow(2) / 4
    pos = u > 0
    return torch.where(pos, torch.exp(1 - 1 / torch.where(pos, u, torch.ones_like(u))), torch.zeros_like(u))


def _ratio_e1(z):
    """z e^z E_1(z) for z >= 1."""
    a = (8.5733287401, 18.0590169730, 8.6347608925, 0.2677737343)
    b = (9.5733223454, 25.6329561486, 21.0996530827, 3.9584969228)
    num = (((z + a[0]) * z + a[1]) * z + a[2]) * z + a[3]
    den = (((z + b[0]) * z + b[1]) * z + b[2]) * z + b[3]
    return num / den


def _Phi(u, kind):
    if kind == "sparsemax":
        return u.clamp_min(0).pow(2) / 2
    if kind == "entmax15":
        return u.clamp_min(0).pow(3) / 12
    pos = u > 0
    us = torch.where(pos, u, torch.ones_like(u))
    val = torch.exp(1 - 1 / us) * us * (1 - _ratio_e1(1 / us))
    return torch.where(pos, val, torch.zeros_like(u))


class _RegMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mu, kind, dim, iters):
        c, u1 = KINDS[kind]
        m = x.max(dim, keepdim=True).values
        lo, hi = m - mu * u1, m.clone()
        for _ in range(iters):  # sum_i phi((x_i - tau) / mu) is non-increasing in tau
            tau = (lo + hi) / 2
            big = _phi((x - tau) / mu, kind).sum(dim, keepdim=True) > 1
            lo, hi = torch.where(big, tau, lo), torch.where(big, hi, tau)
        tau = (lo + hi) / 2
        u = (x - tau) / mu
        p = _phi(u, kind)
        p = p / p.sum(dim, keepdim=True)
        ctx.save_for_backward(p)
        ctx.dim = dim
        return (tau + mu * _Phi(u, kind).sum(dim, keepdim=True) + c * mu).squeeze(dim)

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        return g.unsqueeze(ctx.dim) * p, None, None, None, None


def reg_max(x, mu, kind, dim=-1, iters=30):
    """Regularised maximum of x along dim with temperature mu; gradient = maximiser p*."""
    return _RegMax.apply(x, mu, kind, dim, iters)


def reg_min(x, mu, kind, dim=-1, iters=30):
    return -reg_max(-x, mu, kind, dim, iters)


def reg_weights(x, mu, kind, dim=-1, iters=40):
    """The maximiser p*(x) (the gradient of reg_max)."""
    xr = x.detach().requires_grad_(True)
    reg_max(xr, mu, kind, dim, iters).sum().backward()
    return xr.grad
