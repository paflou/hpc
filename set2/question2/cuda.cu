#include "headers/def.h"
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

#define N 4096
#include "headers/cuda_shared.h"
#include "headers/cuda_cublas.h"
#include "headers/cpu.h"
#include "headers/cuda_global.h"

double t1, t2, cuda_global_time, cuda_shared_time, cublas_time, cpu_time;
double *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
double *d_temp, *d_temp2;

int main(int argc, char **argv)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    int numDev;
    cudaGetDeviceCount(&numDev);
    if (numDev < 1)
    {
        printf("CUDA device missing! Do you need to use optirun?\n");
        return 1;
    }

    // Use 1D array for N*N matrixes
    // Host
    printf("initializing matrices\n");
    t1 = get_wtime();
    double *A = (double *)malloc(sizeof(double) * N * N);
    double *B = (double *)malloc(sizeof(double) * N * N);
    double *C = (double *)malloc(sizeof(double) * N * N);
    double *D = (double *)malloc(sizeof(double) * N * N);
    double *temp = (double *)malloc(sizeof(double) * N * N);

    double *E_blas = (double *)malloc(sizeof(double) * N * N);
    double *F_blas = (double *)malloc(sizeof(double) * N * N);
    double *E_cuda = (double *)malloc(sizeof(double) * N * N);
    double *F_cuda = (double *)malloc(sizeof(double) * N * N);
    double *E_shared = (double *)malloc(sizeof(double) * N * N);
    double *F_shared = (double *)malloc(sizeof(double) * N * N);
    double *E_CPU = (double *)malloc(sizeof(double) * N * N);
    double *F_CPU = (double *)malloc(sizeof(double) * N * N);

    // random seeds
    int seed[] = {1, 10, 100, 1000};
    initializeMatrix(A, seed[0]);
    initializeMatrix(B, seed[1]);
    initializeMatrix(C, seed[2]);
    initializeMatrix(D, seed[3]);
    t2 = get_wtime();
    printf("Matrices initialized in %.5f seconds\n", t2 - t1);

    // printf("First number of matrix A: %f \n", A[0]);
    // printf("First number of matrix B: %f \n", B[0]);
    // printf("First number of matrix C: %f \n", C[0]);
    // printf("First number of matrix D: %f \n", D[0]);

    printf("\nAllocating memory on device\n");
    t1 = get_wtime();
    initializeCUDA();
    t2 = get_wtime();
    printf("Memory allocated on device in %.5f seconds\n", t2 - t1);

    printf("\nCopying data to device\n");
    t1 = get_wtime();
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, N * N * sizeof(double), cudaMemcpyHostToDevice);
    t2 = get_wtime();
    printf("Data copied to device in %.5f seconds\n", t2 - t1);

    // CUDA kernel (cuBLAS)
    printf("\nRunning cuBLAS\n");
    computeCUBLAS(handle, E_blas, F_blas);
    //    printMatrix(F_blas);

    //set d_E and d_F to 0
    cudaMemset(d_E, 0, N * N * sizeof(double));
    cudaMemset(d_F, 0, N * N * sizeof(double));

    // CUDA kernel (custom)
    printf("\nRunning custom CUDA\n");
    computeCUDAGlobal(E_cuda, F_cuda);

    //set d_E and d_F to 0
    cudaMemset(d_E, 0, N * N * sizeof(double));
    cudaMemset(d_F, 0, N * N * sizeof(double));

    // CUDA kernel (custom with shared memory)
    printf("\nRunning custom CUDA (shared)\n");
    computeCUDAShared(E_shared, F_shared);

    // CPU
    printf("\nRunning CPU\n");
    //computeCPU(A, B, C, D, E_CPU, F_CPU);


    printf("\nE: \n");
    printf("Comparing cuda and blas:\t");
    compareMatrices(E_cuda, E_blas);
    printf("Comparing shared and blas:\t");
    compareMatrices(E_shared, E_blas);
    printf("Comparing cpu and blas:\t\t");
    //compareMatrices(E_CPU, E_blas);

    printf("\nF: \n");
    printf("Comparing cuda and blas:\t");
    compareMatrices(F_cuda, F_blas);
    printf("Comparing shared and blas:\t");
    compareMatrices(F_shared, F_blas);
    printf("Comparing cpu and blas:\t\t");
    //compareMatrices(F_CPU, F_blas);

    float global_speedup = cpu_time / cuda_global_time;
    float shared_speedup = cpu_time / cuda_shared_time;
    float cublas_speedup = cpu_time / cublas_time;


    //printf("\nSpeedup of global: %f\n", global_speedup);
    //printf("Speedup of shared: %f\n", shared_speedup);
    //printf("Speedup of cublas: %f\n", cublas_speedup);

    void cleanupCUDA();
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);
    free(D);
    free(E_blas);
    free(F_blas);
    free(E_cuda);
    free(F_cuda);
    free(E_CPU);
    free(F_CPU);
    free(temp);

    return 0;
}
