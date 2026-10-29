"""Write the job lists (one shell command per line) for every experiment.

gpu_*.txt jobs are light and run a few at a time; cpu_*.txt jobs (the pure-Python
baselines of Bateni et al.) run one per core; excl_*.txt must run alone (timing).
"""

from pathlib import Path

import numpy as np

DATASETS = ["adult", "bank", "diabetes", "gowalla", "skin", "shuttle", "covtype"]
SEEDS = range(10)
LAMS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, "inf"]
TILTS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SCALE_N = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 0]  # 0 = all 6.44M

J = Path("jobs")
J.mkdir(exist_ok=True)
R = "python3 run_exp.py"


def write(name, cmds):
    (J / name).write_text("\n".join(cmds) + "\n")
    print(name, len(cmds))


# E1 + E2: main table and Pareto fronts
gpu = []
for d in DATASETS:
    for s in SEEDS:
        gpu.append(f"{R} --data {d} --method kmeans --seed {s} --out results/main_gpu.jsonl")
        for lam in LAMS:
            gpu.append(f"{R} --data {d} --method fmm --lam {lam} --seed {s} --out results/main_gpu.jsonl")
        for t in TILTS:
            gpu.append(f"{R} --data {d} --method tkm --tilt {t} --seed {s} --out results/main_gpu.jsonl")
write("gpu_main.txt", gpu)

cpu = []
for d in DATASETS:
    cpu.append(f"timeout 6h {R} --data {d} --method greedy --seed 0 --device cpu --out results/main.jsonl")
    for s in SEEDS:
        cpu.append(f"timeout 6h {R} --data {d} --method lspp --seed {s} --device cpu --out results/main.jsonl")
for d in DATASETS:  # ICML20 is by far the slowest (hours on a 4000-point sample): last, 3 seeds
    for s in range(3):
        cpu.append(f"timeout 6h {R} --data {d} --method icml20 --seed {s} --device cpu --out results/main.jsonl")
write("cpu_main.txt", cpu)

# E3: scalability on the full gowalla check-in set
scale_gpu, scale_cpu, scale_excl = [], [], []
for n in SCALE_N:
    for s in range(3):
        o = "--out results/scale.jsonl"
        scale_gpu.append(f"{R} --data gowalla_full --n {n} --method fmm --lam 10 --seed {s} {o}")
        scale_gpu.append(f"{R} --data gowalla_full --n {n} --method fmm --lam 10 --batch 65536 --hard 1024 --seed {s} {o}")
        scale_gpu.append(f"{R} --data gowalla_full --n {n} --method kmeans --seed {s} {o}")
        scale_excl.append(f"{R} --data gowalla_full --n {n} --method fmm --lam 10 --device cpu --seed {s} {o}")
        if n and n <= 1_000_000:
            scale_cpu.append(f"timeout 3h {R} --data gowalla_full --n {n} --method lspp --device cpu --seed {s} {o}")
write("cpu_scale.txt", scale_cpu)
write("excl_scale.txt", scale_excl + scale_gpu)  # timing runs alone

# E4: stochastic optimisation (batch size x hard-point memory), with curves
sto = []
for d, n in [("covtype", 0), ("gowalla_full", 1_000_000)]:
    for lam in [10, "inf"]:
        for s in range(5):
            o = f"--out results/stochastic.jsonl --log_curve --data {d} --n {n} --lam {lam} --seed {s}"
            sto.append(f"{R} --method fmm {o}")
            for b in [1024, 4096, 16384, 65536]:
                for h in [0, 256, 1024]:
                    sto.append(f"{R} --method fmm --batch {b} --hard {h} {o}")
write("gpu_stochastic.txt", sto)

# E5: sensitivity to k and to the temperature schedule
abl = []
for d in ["adult", "bank", "diabetes", "skin"]:
    for k in [5, 20, 50]:
        for s in range(5):
            o = f"--data {d} --k {k} --seed {s} --out results/k.jsonl"
            abl.append(f"{R} --method kmeans {o}")
            for lam in [1, 10, "inf"]:
                abl.append(f"{R} --method fmm --lam {lam} {o}")
    for s in range(5):
        o = f"--data {d} --seed {s} --lam 10 --out results/schedule.jsonl"
        for e in [20, 200, 2000]:  # fixed temperature (no annealing)
            abl.append(f"{R} --method fmm --eps0 {e} --eps1 {e} --beta0 {e} --beta1 {e} {o}")
        for e1 in [50, 200, 1000]:  # annealing to different final temperatures
            abl.append(f"{R} --method fmm --eps1 {e1} --beta1 {e1} {o}")
        for it in [500, 1000, 4000]:
            abl.append(f"{R} --method fmm --iters {it} {o}")
write("gpu_ablation.txt", abl)
write("cpu_k.txt", [
    # LSPP is O(n k^2) in pure Python: only the two smaller datasets for the k sweep
    f"timeout 6h {R} --data {d} --k {k} --method lspp --seed {s} --device cpu --out results/k.jsonl"
    for d in ["adult", "bank"] for k in [5, 20, 50] for s in range(5)
])
