"""fairmmm: individually fair clustering by smooth min-max-min optimisation.

    >>> from fairmmm import FairMMM
    >>> model = FairMMM(n_clusters=10, lam=10.0).fit(X)
    >>> model.cluster_centers_, model.max_ratio_
"""

from .estimator import FairMMM, fairness_radii, fairness_report

__all__ = ["FairMMM", "fairness_radii", "fairness_report"]
__version__ = "0.1.0"
