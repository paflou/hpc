import numpy as np
import matplotlib.pyplot as plt

# Data from the table
procs = np.array([2, 4, 6, 8])
multiprocessing_pool = np.array([37.76, 23.14, 15.48, 12.00])
mpi_futures = np.array([61.5, 28.14, 24.94, 19.72])
master_worker = np.array([71.07, 27.38, 24.95, 21.19])
serial_time = 76.98

# Compute speedup
speedup_multiprocessing = serial_time / multiprocessing_pool
speedup_mpi_futures = serial_time / mpi_futures
speedup_master_worker = serial_time / master_worker

# Plot execution times
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(procs, multiprocessing_pool, 'o-', label='multiprocessing.Pool')
plt.plot(procs, mpi_futures, 's-', label='MPI Futures')
plt.plot(procs, master_worker, 'd-', label='Master-Worker')
plt.axhline(serial_time, color='k', linestyle='--', label='Serial')
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Number of Processes')
plt.legend()
plt.grid()

# Plot speedup
plt.subplot(1, 2, 2)
plt.plot(procs, speedup_multiprocessing, 'o-', label='multiprocessing.Pool')
plt.plot(procs, speedup_mpi_futures, 's-', label='MPI Futures')
plt.plot(procs, speedup_master_worker, 'd-', label='Master-Worker')
plt.plot(procs, procs, '--', color='gray', label='Ideal Speedup')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Processes')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
