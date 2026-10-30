import math

import numpy as np
import pytest

from fairmmm import FairMMM, fairness_radii, fairness_report


@pytest.fixture(scope="module")
def blobs():
    # dense blobs plus a sparse background: k-means serves the background poorly
    rng = np.random.default_rng(0)
    dense = np.concatenate([rng.normal(c, 0.3, size=(400, 2)) for c in [(0, 0), (4, 0), (0, 4)]])
    sparse = rng.uniform(-6, 10, size=(150, 2))
    return np.concatenate([dense, sparse]).astype(np.float32)


def test_radii(blobs):
    exact = fairness_radii(blobs, 5, method="exact", device="cpu")
    sample = fairness_radii(blobs, 5, method="sample", device="cpu")
    assert exact.shape == sample.shape == (len(blobs),)
    assert (exact > 0).all() and (sample > 0).all()
    assert np.corrcoef(exact, sample)[0, 1] > 0.9


def test_fairness_improves_with_lambda(blobs):
    radii = fairness_radii(blobs, 5, method="exact", device="cpu")
    km = FairMMM(5, lam=0.0, radii=radii, n_iter=300, device="cpu").fit(blobs)
    fair = FairMMM(5, lam=math.inf, radii=radii, n_iter=300, device="cpu").fit(blobs)
    assert fair.cluster_centers_.shape == (5, 2)
    assert fair.max_ratio_ < km.max_ratio_
    assert fair.labels_.shape == (len(blobs),)
    assert set(np.unique(fair.labels_)) <= set(range(5))


def test_report_matches_attributes(blobs):
    m = FairMMM(5, lam=10.0, n_iter=200, device="cpu").fit(blobs)
    rep = fairness_report(blobs, m.cluster_centers_, m.radii_)
    assert rep["max_ratio"] == pytest.approx(m.max_ratio_)
    assert 0.0 <= rep["frac_unfair"] <= 1.0


def test_minibatch_with_memory(blobs):
    m = FairMMM(5, lam=math.inf, n_iter=300, batch_size=256, memory_size=32, device="cpu").fit(blobs)
    assert np.isfinite(m.max_ratio_)


def test_weights_and_balance(blobs):
    w = np.random.default_rng(1).integers(1, 100, size=len(blobs)).astype(float)
    m = FairMMM(5, lam=10.0, weights=w, balance=10.0, n_iter=200, device="cpu").fit(blobs)
    assert m.report_["norm_entropy"] > 0.5


@pytest.mark.parametrize("kind,bound,lead", [("sparsemax", 0.5, 1.0), ("entmax15", 4 / 3, 2.0), ("bump", 0.5964, 1.0)])
def test_regularised_max(kind, bound, lead):
    import torch

    from fairmmm._smooth import reg_max, reg_weights

    torch.manual_seed(0)
    for n in [2, 50, 5000]:
        x = torch.randn(n, dtype=torch.float64)
        mu = 0.3
        err = (reg_max(x, mu, kind, iters=80) - x.max()).item()
        assert -1e-9 <= err <= bound * mu + 1e-9  # error independent of n
        p = reg_weights(x, mu, kind, iters=80)
        assert p.sum().item() == pytest.approx(1.0) and (p >= 0).all()
        y = x.clone()
        y[0] = x.max() + lead * mu + 1e-6  # exact once the largest entry leads by the margin
        assert reg_max(y, mu, kind, iters=80).item() == pytest.approx(y.max().item(), abs=1e-9)


def test_bumpmax_estimator(blobs):
    radii = fairness_radii(blobs, 5, method="exact", device="cpu")
    km = FairMMM(5, lam=0.0, radii=radii, n_iter=300, device="cpu").fit(blobs)
    m = FairMMM(5, lam=math.inf, radii=radii, n_iter=300, smoothing="bump", match_error=True, device="cpu").fit(blobs)
    assert m.max_ratio_ < km.max_ratio_
