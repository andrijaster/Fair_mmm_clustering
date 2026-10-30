# fairmmm — individually fair min-max-min clustering

`fairmmm` implements **FMM**, the method of

> A. Petrović, M. Nikolić. *Individually Fair Min-Max-Min Clustering by Smooth Gradient-Based Optimization.* Submitted to Data Mining and Knowledge Discovery.

Individual fairness in clustering (Jung, Kannan & Lutz, 2020) asks that every point $x_i$ have a centre within its **fairness radius** $\delta_i$: the distance to its $\lceil n/k\rceil$-th nearest neighbour, i.e. the radius that holds the $n/k$ points a centre serves on average. The maximal violation is

$$\alpha(C)=\max_i\ \min_j\ \frac{\lVert x_i-c_j\rVert}{\delta_i},$$

and $\alpha(C)\le 1$ means every point is served within its radius. FMM minimises

$$\frac1n\sum_i\min_j\lVert x_i-c_j\rVert^2\;+\;\lambda\,\Phi_{\varepsilon,\beta}(C),$$

where $\Phi$ is a smooth (Tsoukalas–Rustem) approximation of $\alpha$ with error at most $\ln n/\varepsilon+\ln k/\beta$. The weight $\lambda$ moves from $k$-means ($\lambda=0$) to the pure min-max-min fairness problem ($\lambda=\infty$). Optimisation uses Adam with temperature continuation and runs on CPU or GPU; mini-batches use a memory of the worst-served points. Point weights (e.g. population) and a load-balance penalty are supported.

## Installation

The package needs Python ≥ 3.8, NumPy and PyTorch.

**With uv**

```bash
# as a dependency of your project
uv add git+https://github.com/andrijaster/Fair_mmm_clustering.git

# or into the current environment
uv pip install git+https://github.com/andrijaster/Fair_mmm_clustering.git

# from a clone, for development (creates .venv and installs the package with test extras)
git clone https://github.com/andrijaster/Fair_mmm_clustering.git
cd Fair_mmm_clustering
uv sync --extra test
uv run pytest
```

**With pip**

```bash
pip install git+https://github.com/andrijaster/Fair_mmm_clustering.git

# or from a clone
git clone https://github.com/andrijaster/Fair_mmm_clustering.git
cd Fair_mmm_clustering
pip install -e ".[test]"
pytest
```

PyTorch is installed from PyPI; for a specific CUDA build install `torch` first following
[pytorch.org](https://pytorch.org/get-started/locally/) (with uv: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`).
A CPU-only install works for everything; a GPU is used automatically when available.

## Usage

```python
import numpy as np
from fairmmm import FairMMM

X = np.random.default_rng(0).normal(size=(10_000, 2))
X = (X - X.mean(0)) / X.std(0)            # FMM's default learning rate assumes comparable scales

model = FairMMM(n_clusters=10, lam=10.0)   # lam=0: k-means, lam=float("inf"): pure min-max-min
model.fit(X)

model.cluster_centers_    # (10, 2) centres
model.labels_             # nearest centre of every point
model.max_ratio_          # alpha = max_i d(x_i, C) / delta_i   (<= 1: everyone served fairly)
model.report_             # cost, max_ratio, frac_unfair, mean_ratio, norm_entropy
```

### Main options

| argument | meaning | default |
|---|---|---|
| `n_clusters` | number of centres $k$ | 8 |
| `lam` | weight $\lambda$ of the fairness term; `float("inf")` = pure min-max-min | 10.0 |
| `radii` | `"sample"` (estimator of Bateni et al. 2024, fast), `"exact"` ($O(n^2)$), or an array of radii | `"sample"` |
| `weights` | point weights (e.g. population); enables weighted cost and weighted radii | `None` |
| `balance` | weight $\gamma$ of the load-balance penalty | 0.0 |
| `n_iter` | Adam iterations | 2000 |
| `lr` | learning rate; `"auto"` = 0.02 × mean coordinate std | `"auto"` |
| `temperatures` | start and end of the geometric temperature schedule | (2, 200) |
| `batch_size`, `memory_size` | mini-batch size and size of the hard-point memory | `None`, 0 |
| `smoothing` | smooth max/min of the surrogate: `"lse"` (log-sum-exp), `"sparsemax"`, `"entmax15"` or `"bump"` (BumpMax) | `"lse"` |
| `match_error` | regularised maxima: scale the temperatures to the error bound of log-sum-exp | `False` |
| `device` | `"auto"`, `"cpu"` or `"cuda"` | `"auto"` |

### Smooth maxima, including BumpMax

`fairmmm.reg_max(x, mu, kind, dim)` is a regularised maximum: a smooth upper approximation of
`x.max(dim)` whose gradient is a probability vector (`kind` = `"sparsemax"`, `"entmax15"` or
`"bump"`; `reg_min` is the corresponding minimum). Unlike log-sum-exp, whose error grows as
`mu * ln(n)`, its error is at most `0.5 mu`, `1.33 mu` and `0.596 mu` for any number of entries,
and it equals the maximum exactly once the largest entry leads the second by `mu` (`2 mu` for
`entmax15`). BumpMax is built from the flat function `exp(1 - 1/u)` and is the only one of the
three that is infinitely differentiable (paper, Section 4.3).

```python
import torch
from fairmmm import reg_max

x = torch.randn(1000, requires_grad=True)
v = reg_max(x, 0.1, "bump")     # within 0.06 of x.max()
v.backward()                     # x.grad: sparse weights summing to one
```

### Radii and evaluation without fitting

```python
from fairmmm import fairness_radii, fairness_report

radii = fairness_radii(X, n_clusters=10, method="exact")
fairness_report(X, centers, radii)     # evaluate any set of centres, e.g. from scikit-learn
```

### Weighted facility placement with load balance

```python
# xyz: coordinates (e.g. block-group centres on the sphere, in km), pop: population
model = FairMMM(n_clusters=25, lam=10.0, weights=pop, balance=10.0).fit(xyz)
model.report_["frac_unfair"]          # share of the population outside its radius
```

## Reproducing the paper

The `experiments/` folder contains every script used for the paper. Install the package with the experiment extras and fetch the baselines of Bateni et al. (2024) from google-research:

```bash
uv sync --extra experiments          # or: pip install -e ".[experiments]"
cd experiments
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/google-research/google-research.git gr
git -C gr sparse-checkout init --cone && git -C gr sparse-checkout set individually_fair_clustering && git -C gr checkout
```

| step | command |
|---|---|
| download and standardise the seven datasets | `python prepare_data.py` |
| cache the fairness radii (identical for every method) | `python precompute_delta.py` |
| write the job lists | `python make_jobs.py`, `python jobs_exact.py`, `python jobs_revision.py` |
| run everything (GPU and CPU queues) | `bash run_all.sh`, `bash run_gpu.sh`, `bash run_exact.sh` |
| added baselines, tuned TKM, LSPP parameter grid, ablations, radius samples, exclusive timings | `bash run_revision.sh` |
| census case study | `python case_census.py` |
| comparison of smooth maxima (log-sum-exp, sparsemax, 1.5-entmax, BumpMax) | `bash run_smoothing.sh`, then `python analyze_smoothing.py results/smoothing.jsonl ../paper/tables/smoothing_rows.tex` and `python smoothing_figure.py` |
| optimality gap on small instances (exact MIQCP with SCIP; needs `pip install pyscipopt`) | `python exact_small_samples.py`, `python exact_small.py samples.npz exact_small.jsonl --n 20` (and `--n 30`), then `python exact_small.py exact_small.jsonl ../paper/tables/exact_gap_rows.tex --table` |
| tables and figures | `python analyze.py`, `python make_maps.py`, `python toy_figure.py` |
| interactive maps | `python make_interactive.py` |

`run_exp.py` runs a single (dataset, method, seed) job and appends one JSON line to a results file, e.g.

```bash
python run_exp.py --data adult --method fmm --lam 10 --seed 0 --out results/main.jsonl
python run_exp.py --data adult --method lspp --seed 0 --device cpu --out results/main.jsonl
```

Map figures need the U.S. Census TIGER/Line and cartographic boundary files and Natural Earth countries in `experiments/data_maps/` (download links in `make_maps.py`).

## Repository layout

```
src/fairmmm/       the package (FairMMM estimator, radii, evaluation)
  _core.py         smooth min-max-min objective, full-batch and mini-batch optimisation
  _weighted.py     population weights and load-balance penalty
tests/             pytest suite
experiments/       scripts that produce every table and figure of the paper
```

## Citation

```bibtex
@article{PetrovicNikolic_FairMMM,
  author  = {Petrovi{\'c}, Andrija and Nikoli{\'c}, Mladen},
  title   = {Individually Fair Min-Max-Min Clustering by Smooth Gradient-Based Optimization},
  journal = {Data Mining and Knowledge Discovery},
  note    = {submitted},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
