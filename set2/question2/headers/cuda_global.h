#include "def.h"
#include "cpu.h"

#ifndef CUDA_GLOBAL_H
#define CUDA_GLOBAL_H


__global__ void matMult(double *a, double *b, double *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        double sum = 0.0;
        // Column-major indexing like cuBLAS
        for (int k = 0; k < N; k++)
        {
            // a[i,k] * b[k,j] in column-major: a[i + k*N] * b[k + j*N]
            sum += a[row + k * N] * b[k + col * N];
        }
        // Store result in column-major: c[i,j] = c[i + j*N]
        c[row + col * N] = sum;
    }
}

__global__ void vecSub(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        c[j * N + i] = b[j * N + i] - a[j * N + i];
    }
}

__global__ void vecAdd(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        c[j * N + i] = b[j * N + i] + a[j * N + i];
    }
}

void computeCUDAGlobal(double *E_cuda, double *F_cuda)
{
    double t1 = get_wtime();
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMult<<<numBlocks, threadsPerBlock, 0>>>(d_A, d_C, d_E);
    matMult<<<numBlocks, threadsPerBlock, 0>>>(d_B, d_D, d_temp);
    vecSub<<<numBlocks, threadsPerBlock, 0>>>(d_temp, d_E, d_E);

    matMult<<<numBlocks, threadsPerBlock, 0>>>(d_A, d_D, d_temp2);
    matMult<<<numBlocks, threadsPerBlock, 0>>>(d_B, d_C, d_F);
    vecAdd<<<numBlocks, threadsPerBlock, 0>>>(d_temp2, d_F, d_F);

    cudaMemcpy(E_cuda, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(F_cuda, d_F, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    double t2 = get_wtime();
    cuda_global_time = t2 - t1;
    printf("Global CUDA: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n", E_cuda[N*N-1], F_cuda[N*N-1], t2 - t1);
}

#endif // CUDA_GLOBAL_H