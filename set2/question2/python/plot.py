import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
file_name = 'benchmark_results.csv'
data = pd.read_csv(file_name)

# Extract columns
N = data['N']
CUDA_time = data['CUDA_time']
CUBLAS_time = data['CUBLAS_time']
CPU_time = data['CPU_time']

Speedup = data['Speedup']

# Plot 1: Execution Times
plt.figure(figsize=(10, 5))
plt.plot(N, CUDA_time, marker='o', label='CUDA Time')
plt.plot(N, CPU_time, marker='o', label='CPU Time')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Problem Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('CUDA vs CPU Execution Time')
plt.legend()
plt.grid(True)

# Plot 2: Speedup
plt.figure(figsize=(10, 5))
plt.plot(N, Speedup, marker='o', label='Speedup', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Problem Size (N)')
plt.ylabel('Speedup')
plt.title('CUDA Speedup over CPU')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
