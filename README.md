This repository contains code for high-performance computing (HPC) projects as part of the CEID course CEID_NE5407 (2024-2025).
  
# Set 1 - MPI and OpenMP
## Question 1: Homemade implementation of MPI_Exscan

### MPI_Exscan_pt2pt.c
An implementation of MPI_Exscan using point-to-point communication that uses recursive doubling, achieving O(logN) time complexity.

### MPI_Exscan_omp.c
A hybrid implementation of MPI_Exscan, that also utilizes OpenMP parallelism in each process.

### MPI_Exscan_omp_io.c
Extends the MPI_Exscan_omp.c example to **include I/O operations**, writing results to a binary file.

### MPI_Exscan_omp_io_compressed.c
Further extends the MPI_Exscan_omp_io.c example by adding **data compression** provided by the zlib library before writing to the binary file.


##
## Question 2: Parallel Parametric Search in Machine Learning
### gs.py
Uses a grid search method to train an MLP classifier on a synthetic dataset and evaluates its accuracy. It runs sequentially without parallel processing.

### q2a.py
Parallelizes the grid search using Python's multiprocessing.Pool . It distributes the parameter combinations across multiple processes to speed up the training and evaluation.

### q2b.py
Uses mpi4py.futures.MPICommExecutor to parallelize the grid search across multiple MPI processes. It maps the evaluation function over the parameter grid using MPI for parallel execution.

### q2c.py
Employs a master-worker model using MPI to parallelize the grid search. The master process distributes the tasks to worker processes, which perform the training and evaluation.

### Results
![image](https://github.com/user-attachments/assets/30a18f88-a5a5-4583-8ffc-15c7f668e2f3)

Running all three parallel implementations with **9000 samples and 2 features (data.csv)** we can determine **multiprocessing pool** achieves the best results, almost doubling the speedup of the other 2 implementations when ran with 8 processes.


##
## Question 3: Simple example of OpenMP tasks (irrelevant, used for course)

#
# Set 2 - SIMD - CUDA
## Question 1: SIMD and WENO5
Parallelized a WENO5 implementation using different methods and benchmarked the results for different cache usage percentages to accurately guage speedup.

### original
Contains the refernce implementation with no optimizations other than -O1

### first
Only difference between the refernce implmementation are these GCC flags 
```-march=native -ftree-vectorize ``` and a few changes in ```weno_minus_core``` to help GCC understand the data is aligned so as to perform the optimizations.

### second
An OpenMP vectorization implementation. The benefit of OpenMP vectorization is its simplicity, but we lose out on performance.

### third
This is an AVX intrinsics implementation. It utilizes AVX intrinsics to perform 8 calculations per cycle.

## Results
To guage the performance improvements of the 3 parallel implementations compared to the reference one, we used a benchmark that runs the implementation multiple times, with different cache availability.

![image](https://github.com/user-attachments/assets/5fb4481c-b22b-4c07-825f-1e06495dba5d)
![image](https://github.com/user-attachments/assets/60dc341d-5487-455c-aad9-a376e25f32d1)

The AVX intrinsics implementation achieved 8-10x improvement compared to reference, beating out gcc optimizations (6-8x).
OpenMP was pretty slow compared to the other 2, only managing 3-4x speedup.

## Question 2: CUDA and Complex Matrix Multiplication 
Created a CUDA program that performs Complex Matrix Multiplication using 3 different CUDA techniques:

```
- Global Memoery CUDA
- Shared Memory CUDA
- cuBLAS library
```
The folder also contains a reference CPU program to compare against.

## Results
As expected, global CUDA performed the worst, followed by shared CUDA, but never quite reaching the performance of a very well optimized library like cuBLAS, which achieved around 25.000x speedup over CPU.

![image](https://github.com/user-attachments/assets/8769f612-c876-479d-a213-919510a01fe8)


#
# Set 3 - OpenMP and Complex Matrix Multiplication

The objective of this set is to use OpenMP GPU to the same program as the CUDA implementation from before. OpenMP was truly seamless to use, needing just a few commands, but the performance benefits were much less impressive.
The speedup seems to stop increasing at around 100x, while the slowest of the CUDA implementations reached over 1000x.
![image](https://github.com/user-attachments/assets/501606b7-0adb-46c5-b428-97d9179be2f0)

As an example, for matrices of size 4096x4096, cuBLAS needed 0.33s (on a Tesla V100 GPU) while OpenMP took 128s.
