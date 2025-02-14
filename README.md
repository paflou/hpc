# HPC

This repository contains code for high-performance computing (HPC) projects as part of the CEID course CEID_NE5407 (2024-2025).

## Languages and Technologies

- **C**: Main language for performance-critical components.
- **Python**: Used for scripting and automation tasks.
- **Cuda**: Used for GPU computing.

<!--
```sh
git clone https://github.com/paflou/hpc.git
```

# Set 1 - MPI and OpenMP

## Question 1: Hybrid Programming Model and Parallel I/O
This directory contains several MPI and OpenMP implementations for various parallel computing tasks.


### MPI_Exscan_pt2pt.c
    A **point-to-point** communication version of the parallel prefix sum (exclusive scan) using MPI.

### MPI_Exscan_omp.c
    A **parallel** prefix sum (exclusive scan) using **MPI and OpenMP**.

### MPI_Exscan_omp_io.c
    Extends the MPI_Exscan_omp.c example to **include I/O operations**, writing results to a binary file.

### MPI_Exscan_omp_io_compressed.c
    Further extends the MPI_Exscan_omp_io.c example by adding **data compression** before writing to the binary file.


##
## Question 2: Parallel Parametric Search in Machine Learning
    This directory contains various implementations for training and evaluating a neural network model using different parallel processing techniques.

### gs.py
    Uses a grid search method to train an MLP classifier on a synthetic dataset and evaluates its accuracy. It runs sequentially without parallel processing.

### q2a.py
    Parallelizes the grid search using Python's multiprocessing.Pool. It distributes the parameter combinations across multiple processes to speed up the training and evaluation.

### q2b.py
    Uses mpi4py.futures.MPICommExecutor to parallelize the grid search across multiple MPI processes. It maps the evaluation function over the parameter grid using MPI for parallel execution.

### q2c.py
    Employs a master-worker model using MPI to parallelize the grid search. The master process distributes the tasks to worker processes, which perform the training and evaluation.
--->

