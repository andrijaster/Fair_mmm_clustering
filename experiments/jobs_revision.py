"""Job lists for the experiments added after the internal review (run with run_revision.sh).

gpu_rev.txt   mini-batch study without best-iterate selection, ablations, simple baselines,
              tuned TKM, radius-sample seeds
cpu_rev.txt   LSPP for the radius-sample seeds and an LSPP front (anchor/search coefficients)
excl_rev.txt  timing with exclusive access: LSPP and FMM on one CPU thread, FMM on 12 threads
"""

from pathlib import Path

R = "python3 run_exp.py"
J = Path("jobs_rev")
J.mkdir(exist_ok=True)
ALL = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
gpu, cpu, excl = [], [], []

# (1) mini-batch study, returning the last iterate (no full-data selection); curves still logged
for d, n in [("covtype", 0), ("gowalla_full", 1_000_000)]:
    for lam in [10, "inf"]:
        for s in range(5):
            o = f"--data {d} --n {n} --lam {lam} --seed {s} --no_select --log_curve --out results/stochastic_v2.jsonl"
            gpu.append(f"{R} --method fmm {o}")
            for b in [1024, 4096, 16384, 65536]:
                for h in [0, 256, 1024]:
                    gpu.append(f"{R} --method fmm --batch {b} --hard {h} {o}")

# (2) ablations: selection, initialisation, smoothing, unequal temperatures, learning rate
VARIANTS = {
    "default": "",
    "no_select": "--no_select",
    "init_kmeans1": "--init kmeans1",
    "init_random": "--init random",
    "nosmooth": "--nosmooth",
    "beta_low": "--beta0 0.2 --beta1 20",
    "beta_high": "--beta0 20 --beta1 2000",
    "lr_0.005": "--lr 0.005",
    "lr_0.05": "--lr 0.05",
}
for d in ["adult", "bank", "skin", "covtype"]:
    for lam in [10, "inf"]:
        for s in range(5):
            for v, flags in VARIANTS.items():
                gpu.append(f"{R} --method fmm --data {d} --lam {lam} --seed {s} {flags} --out results/ablation_{v}.jsonl")

# (3) simple baselines that target the ratio directly
for d in ALL:
    gpu.append(f"{R} --method ratio_greedy --data {d} --seed 0 --out results/baselines_v2.jsonl")
    for s in range(5):
        gpu.append(f"{R} --method wkm_delta --data {d} --seed {s} --out results/baselines_v2.jsonl")

# (4) tilted k-means with tuned learning rate and best-iterate safeguard
for d in ALL:
    for lr in [0.002, 0.02, 0.2]:
        for t in [0.001, 0.01, 0.1, 1, 10]:
            for s in range(3):
                gpu.append(f"{R} --method tkm --tkm_select --data {d} --tilt {t} --lr {lr} --seed {s} "
                           f"--out results/tkm_tuned.jsonl")

# (5) other random samples for the radius estimator
for d in ["adult", "bank", "diabetes", "shuttle"]:
    for rs in [101, 102, 103]:
        o = f"--data {d} --radius_seed {rs} --out results/radius_seeds.jsonl"
        cpu.append(f"timeout 3h {R} --method lspp --device cpu --seed 0 {o}")
        for s in range(3):
            gpu.append(f"{R} --method kmeans --seed {s} {o}")
            for lam in [1, 10, "inf"]:
                gpu.append(f"{R} --method fmm --lam {lam} --seed {s} {o}")

# (6) an LSPP front: tighter/looser anchor and search coefficients
for d in ["adult", "bank", "diabetes", "gowalla"]:
    for ca in [1.5, 3, 6]:
        for cs in [0.5, 1, 2]:
            for s in range(3):
                cpu.append(f"timeout 3h {R} --method lspp --device cpu --data {d} --seed {s} "
                           f"--coeff_anchor {ca} --coeff_search {cs} --out results/lspp_front.jsonl")

# (7) timing with exclusive access, one run per size
for n in [10_000, 30_000, 100_000, 300_000, 1_000_000]:
    o = f"--data gowalla_full --n {n} --seed 0 --out results/scale_v2.jsonl"
    excl.append(f"OMP_NUM_THREADS=1 timeout 3h {R} --method lspp --device cpu {o}")
    excl.append(f"{R} --method fmm --lam 10 --device cpu --threads 1 {o}")
    excl.append(f"{R} --method fmm --lam 10 --device cpu --threads 12 {o}")

for name, jobs in [("gpu_rev.txt", gpu), ("cpu_rev.txt", cpu), ("excl_rev.txt", excl)]:
    (J / name).write_text("\n".join(jobs) + "\n")
    print(name, len(jobs))
