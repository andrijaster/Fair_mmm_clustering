"""Scikit-learn style estimator for individually fair min-max-min clustering."""

import math
import warnings

import numpy as np
import torch

from . import _core, _weighted


def _device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _as_tensor(a, device, dtype):
    if torch.is_tensor(a):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)


def fairness_radii(X, n_clusters, method="sample", weights=None, n_sample=500, seed=100, device="auto"):
    """Fairness radius delta(x_i) of every point (Jung et al., 2020).

    method:
        "sample"  distance to the ceil(n_sample/k)-th nearest of n_sample random points
                  (the estimator of Bateni et al., 2024; O(n * n_sample)),
        "exact"   distance to the ceil(n/k)-th nearest point of X, the point itself included
                  (O(n^2) time, chunked memory).
    With ``weights`` the exact weighted radius is returned: the smallest radius around x_i
    that holds a 1/k share of the total weight (O(n^2)).
    """
    dev = _device(device)
    dtype = torch.float64 if weights is not None else torch.float32
    x = _as_tensor(X, dev, dtype)
    if weights is not None:
        return _weighted.weighted_thresholds(x, _as_tensor(weights, dev, dtype), n_clusters).cpu().numpy()
    if method == "sample":
        return _core.compute_thresholds(x, n_clusters, n_sample=min(n_sample, x.shape[0]), seed=seed).cpu().numpy()
    if method == "exact":
        return _core.exact_thresholds(x, n_clusters).cpu().numpy()
    raise ValueError(f"unknown radius method {method!r}; use 'sample' or 'exact'")


def fairness_report(X, centers, radii, weights=None):
    """Clustering cost and individual-fairness statistics of a set of centres.

    Returns a dict with ``cost`` (sum of squared distances to the nearest centre, weighted if
    ``weights`` is given), ``max_ratio`` (alpha = max_i d(x_i, C) / delta_i),
    ``frac_unfair`` (share of points, or of the weight, with ratio > 1), ``mean_ratio`` and
    ``norm_entropy`` of the cluster sizes.
    """
    x = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    c = torch.as_tensor(np.asarray(centers), dtype=torch.float64)
    r = torch.as_tensor(np.asarray(radii), dtype=torch.float64)
    w = torch.ones(x.shape[0], dtype=torch.float64) if weights is None else torch.as_tensor(
        np.asarray(weights), dtype=torch.float64)
    d, a = torch.cdist(x, c).min(1)
    ratio = d / r
    share = torch.zeros(c.shape[0], dtype=torch.float64).index_add_(0, a, w) / w.sum()
    p = share[share > 0]
    return {
        "cost": float((w * d * d).sum()),
        "max_ratio": float(ratio.max()),
        "frac_unfair": float((w * (ratio > 1)).sum() / w.sum()),
        "mean_ratio": float((w * ratio).sum() / w.sum()),
        "norm_entropy": float(-(p * p.log()).sum() / math.log(c.shape[0])),
    }


class FairMMM:
    """Individually fair clustering by smooth min-max-min optimisation.

    Minimises  cost(C)/n + lam * Phi(C)  (Petrovic & Nikolic), where Phi is the
    Tsoukalas-Rustem smoothing of the maximal fairness violation
    alpha(C) = max_i min_j ||x_i - c_j|| / delta_i. ``lam=float('inf')`` solves the pure
    min-max-min problem; ``lam=0`` is k-means.

    Parameters
    ----------
    n_clusters : int
    lam : float, weight of the fairness term (default 10).
    radii : "sample", "exact" or array of shape (n,), the fairness radii delta_i.
    weights : optional array (n,), point weights (e.g. population); uses the weighted cost,
        weighted radii and the cost normalisation of the census case study.
    balance : float, weight gamma of the load-balance penalty (needs the weighted path; with
        ``weights=None`` unit weights are used).
    n_iter : int, Adam iterations (default 2000).
    lr : float or "auto"; "auto" = 0.02 x mean coordinate standard deviation.
    temperatures : (start, end) of the geometric temperature schedule (default (2, 200)).
    batch_size : int or None; mini-batch size (None = full batch).
    memory_size : int, size of the hard-point memory used with mini-batches.
    device : "auto", "cpu" or "cuda".
    random_state : int.

    Attributes
    ----------
    cluster_centers_ : ndarray (k, d)
    labels_ : ndarray (n,)
    radii_ : ndarray (n,)
    max_ratio_ : float, alpha of the returned centres
    cost_ : float, (weighted) k-means cost of the returned centres
    report_ : dict, see :func:`fairness_report`
    """

    def __init__(self, n_clusters=8, lam=10.0, radii="sample", weights=None, balance=0.0, n_iter=2000, lr="auto",
                 temperatures=(2.0, 200.0), batch_size=None, memory_size=0, device="auto", random_state=0):
        self.n_clusters = n_clusters
        self.lam = lam
        self.radii = radii
        self.weights = weights
        self.balance = balance
        self.n_iter = n_iter
        self.lr = lr
        self.temperatures = temperatures
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.device = device
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
        n = X.shape[0]
        k = self.n_clusters
        if not 2 <= k <= n:
            raise ValueError("n_clusters must be between 2 and n_samples")
        dev = _device(self.device)
        weighted = self.weights is not None or self.balance > 0
        w = None if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        if weighted and w is None:
            w = np.ones(n)

        if isinstance(self.radii, str):
            radii = fairness_radii(X, k, method=self.radii, weights=w if weighted else None, device=dev)
        else:
            radii = np.asarray(self.radii, dtype=np.float64)
            if radii.shape != (n,):
                raise ValueError("radii must have shape (n_samples,)")
        if (radii <= 0).any():
            warnings.warn("some fairness radii are zero (duplicated points); they are clamped at 1e-6")
            radii = np.maximum(radii, 1e-6)
        lr = 0.02 * float(X.std(0).mean()) if self.lr == "auto" else float(self.lr)
        eps = tuple(float(t) for t in self.temperatures)

        if weighted:
            if self.batch_size:
                warnings.warn("mini-batches are not supported with weights or balance; using the full batch")
            x = _as_tensor(X, dev, torch.float64)
            c = _weighted.fmm_weighted(x, _as_tensor(w, dev, torch.float64), _as_tensor(radii, dev, torch.float64), k,
                                       lam=self.lam, alpha=self.balance, iters=self.n_iter, lr=lr, eps=eps, beta=eps,
                                       seed=self.random_state)
        else:
            x = _as_tensor(X, dev, torch.float32)
            cfg = _core.FMMConfig(lam=self.lam, eps0=eps[0], eps1=eps[1], beta0=eps[0], beta1=eps[1], iters=self.n_iter,
                                  lr=lr, batch=int(self.batch_size or 0), hard_buffer=int(self.memory_size),
                                  seed=self.random_state)
            if self.batch_size:
                cfg.eval_every = 10 ** 9  # best-iterate tracking needs full passes
            c = _core.fmm(x, _as_tensor(radii, dev, torch.float32), k, cfg)

        self.cluster_centers_ = c.detach().cpu().numpy().astype(X.dtype if X.dtype.kind == "f" else np.float64)
        self.radii_ = radii
        self.report_ = fairness_report(X, self.cluster_centers_, radii, weights=w if weighted else None)
        self.max_ratio_ = self.report_["max_ratio"]
        self.cost_ = self.report_["cost"]
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """Index of the nearest centre of every row of X."""
        d = torch.cdist(torch.as_tensor(np.asarray(X), dtype=torch.float64),
                        torch.as_tensor(self.cluster_centers_, dtype=torch.float64))
        return d.argmin(1).numpy()

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
