import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
data = pd.read_csv('performance_data.csv')

# Extract columns
Size = data['Size']
CPU_Time = data['CPU Time']
CUDA_Global_Time = data['CUDA Global Time']
CUDA_Shared_Time = data['CUDA Shared Time']
cuBLAS_Time = data['cuBLAS Time']

# Calculate speedups
Global_Speedup = CPU_Time / CUDA_Global_Time
Shared_Speedup = CPU_Time / CUDA_Shared_Time
cuBLAS_Speedup = CPU_Time / cuBLAS_Time

# Plot 1: Execution Times
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Size, CPU_Time, marker='o', label='CPU')
plt.plot(Size, CUDA_Global_Time, marker='s', label='CUDA Global')
plt.plot(Size, CUDA_Shared_Time, marker='^', label='CUDA Shared')
plt.plot(Size, cuBLAS_Time, marker='*', label='cuBLAS')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)

# Plot 2: Speedups
plt.subplot(1, 2, 2)
plt.plot(Size, Global_Speedup, marker='s', label='CUDA Global')
plt.plot(Size, Shared_Speedup, marker='^', label='CUDA Shared')
plt.plot(Size, cuBLAS_Speedup, marker='*', label='cuBLAS')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Speedup over CPU')
plt.title('GPU Speedup Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()