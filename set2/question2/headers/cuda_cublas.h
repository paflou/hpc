#include "def.h"
#include "cpu.h"

#ifndef CUDA_CUBLAS_H
#define CUDA_CUBLAS_H

void computeCUBLAS(cublasHandle_t handle, double *E_blas, double *F_blas)
{
    double t1 = get_wtime();
    const double alpha = 1.0, beta = 0.0, reverse = -1.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_C, N, &beta, d_E, N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &reverse, d_B, N, d_D, N, &alpha, d_E, N);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_D, N, &beta, d_F, N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_C, N, &alpha, d_F, N);

    cudaMemcpy(E_blas, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(F_blas, d_F, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    double t2 = get_wtime();
    cublas_time = t2 - t1;
    printf("cuBLAS CUDA: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n", E_blas[N*N-1], F_blas[N*N-1], t2 - t1);
}
#endif // CUDA_CUBLAS_H