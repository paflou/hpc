import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
data = pd.read_csv('performance_data.csv')

# Extract columns
Size = data['Size']
CPU_Time = data['CPU Time']
OpenMP_GPU_Time = data['OpenMP GPU Time']

# Calculate speedups
OpenMP_GPU_Speedup = CPU_Time / OpenMP_GPU_Time

# Plot 1: Execution Times
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(Size, CPU_Time, marker='o', label='CPU')
plt.plot(Size, OpenMP_GPU_Time, marker='s', label='OpenMP GPU')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)

# Plot 2: Speedups
plt.subplot(1, 2, 2)
plt.plot(Size, OpenMP_GPU_Speedup, marker='s', label='OpenMP GPU')

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Speedup over CPU')
plt.title('Speedup')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()