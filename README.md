# Individually fair min-max-min clustering (FMM)

Code for *Individually Fair Min-Max-Min Clustering by Smooth Gradient-Based
Optimization* (A. Petrović, M. Nikolić).

FMM minimises the $k$-means cost plus $\lambda$ times a smooth surrogate of the
maximal individual-fairness violation of Jung et al. (2020),

$$\alpha(C)=\max_i \min_j \lVert x_i-c_j\rVert/\delta(x_i),$$

where $\delta(x)$ is the radius of the smallest ball around $x$ holding $\lceil n/k\rceil$
points. The surrogate is the Tsoukalas-Rustem smoothing of this min-max-min problem.

## Files

| file | purpose |
|---|---|
| `fmm.py` | FMM (full batch and mini-batch with hard-point memory), radii, evaluation, k-means, tilted k-means |
| `fmm_ext.py` | population weights and load-balance penalty (census case study) |
| `prepare_data.py` | downloads and standardises the seven benchmark datasets |
| `precompute_delta.py` | caches the fairness radii so all methods use identical thresholds |
| `run_exp.py` | runs one (dataset, method, seed) job and appends a JSON line |
| `make_jobs.py`, `jobs_exact.py` | generate the job lists of all experiments |
| `run_all.sh`, `run_gpu.sh`, `run_exact.sh` | run the job lists (GPU and CPU queues) |
| `case_census.py` | US Census 2020 facility-placement case study |
| `map_gowalla.py` | centres on the Gowalla benchmark in latitude/longitude |
| `analyze.py` | tables (LaTeX) and figures (PDF) |
| `make_maps.py` | map figures (needs `pyshp`) |
| `make_interactive.py`, `interactive/template.html` | interactive map explorer |

The baselines LSPP, ICML20 and Greedy are the open-source implementation of
Bateni et al. (AISTATS 2024):
`git clone` google-research and keep `individually_fair_clustering/` next to this
directory as `../gr/individually_fair_clustering`.

## Reproducing

```bash
python3 prepare_data.py            # data/*.npy
python3 precompute_delta.py        # data/delta/*.npy
python3 make_jobs.py               # jobs/*.txt
bash run_all.sh                    # all experiments (GPU + CPU queues)
python3 analyze.py                 # paper_out/tables, paper_out/figs
```

Requirements: Python ≥ 3.8, PyTorch ≥ 2.0 (CUDA for the GPU runs), NumPy, SciPy,
scikit-learn, pandas, matplotlib; `pyshp` for the maps.
