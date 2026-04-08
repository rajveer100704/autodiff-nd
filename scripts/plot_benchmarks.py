import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure the results file exists
if not os.path.exists("bench_results.csv"):
    print("Error: bench_results.csv not found. Please run the benchmark first.")
    exit(1)

df = pd.read_csv("bench_results.csv")

# 1. Performance Plot (Time)
plt.figure(figsize=(10, 6))
plt.errorbar(df["N"], df["eager_mean"], yerr=df["eager_std"], label="Eager Attention (O(N²))", 
             fmt='-o', capsize=5, capthick=2, elinewidth=2, color='#e74c3c')
plt.errorbar(df["N"], df["flash_mean"], yerr=df["flash_std"], label="FlashAttention (Optimized)", 
             fmt='-o', capsize=5, capthick=2, elinewidth=2, color='#2ecc71')

plt.xlabel("Sequence Length (N)", fontsize=12)
plt.ylabel("Execution Time (ms)", fontsize=12)
plt.title("FlashAttention vs Eager Attention: Performance Scaling", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig("benchmark_time.png", dpi=300)
print("Saved benchmark_time.png")

# 2. Memory Scaling Plot
plt.figure(figsize=(10, 6))
plt.plot(df["N"], df["eager_mem"], label="Eager Attention (Estimated O(N²))", 
         marker='o', linestyle='-', linewidth=2, color='#e74c3c')
plt.plot(df["N"], df["flash_mem"], label="FlashAttention (Estimated O(N))", 
         marker='o', linestyle='-', linewidth=2, color='#2ecc71')

plt.xlabel("Sequence Length (N)", fontsize=12)
plt.ylabel("Memory Usage (MB)", fontsize=12)
plt.title("Memory Scaling Comparison (Estimated)", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig("benchmark_memory.png", dpi=300)
print("Saved benchmark_memory.png")
