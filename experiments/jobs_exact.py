"""Exact-threshold robustness check (appendix): job lists."""
R = "python3 run_exp.py"
gpu, cpu = [], []
for d in ["adult", "bank", "shuttle"]:
    for s in range(5):
        o = f"--data {d} --seed {s} --delta exact --out results/exact.jsonl"
        gpu.append(f"{R} --method kmeans {o}")
        for lam in [0.1, 1, 10, "inf"]:
            gpu.append(f"{R} --method fmm --lam {lam} {o}")
        cpu.append(f"timeout 3h {R} --method lspp --device cpu {o}")
open("jobs_gpu/gpu_exact.txt", "w").write("\n".join(gpu) + "\n")
open("jobs_gpu/cpu_exact.txt", "w").write("\n".join(cpu) + "\n")
