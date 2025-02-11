#include "def.h"
#include "cpu.h"
#include "cuda_global.h"


#ifndef CUDA_SHARED_H
#define CUDA_SHARED_H
extern double cuda_shared_time;
extern double *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_temp, *d_temp2;

__global__ void matMultShared(double *a, double *b, double *c)
{
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; m++)
    {
        // Load A tile: column-major [i + k*N]
        if (i < N && (m * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = a[i + (m * TILE_SIZE + threadIdx.x) * N];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        // Load B tile: column-major [k + j*N]
        if ((m * TILE_SIZE + threadIdx.y) < N && j < N)
            Bs[threadIdx.y][threadIdx.x] = b[(m * TILE_SIZE + threadIdx.y) + j * N];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        if (i < N && j < N)
        {
            for (int k = 0; k < TILE_SIZE && (m * TILE_SIZE + k) < N; k++)
            {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (i < N && j < N)
        c[i + j * N] = sum; // Store result in column-major
}


void computeCUDAShared(double *E_shared, double *F_shared)
{
    double t1 = get_wtime();
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMultShared<<<numBlocks, threadsPerBlock, 0>>>(d_A, d_C, d_E);
    matMultShared<<<numBlocks, threadsPerBlock, 0>>>(d_B, d_D, d_temp);
    vecSub<<<numBlocks, threadsPerBlock, 0>>>(d_temp, d_E, d_E);

    matMultShared<<<numBlocks, threadsPerBlock, 0>>>(d_A, d_D, d_temp2);
    matMultShared<<<numBlocks, threadsPerBlock, 0>>>(d_B, d_C, d_F);
    vecAdd<<<numBlocks, threadsPerBlock, 0>>>(d_temp2, d_F, d_F);

    cudaMemcpy(E_shared, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(F_shared, d_F, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    double t2 = get_wtime();
    cuda_shared_time = t2 - t1;
    printf("Shared CUDA: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n",
           E_shared[N*N-1], F_shared[N*N-1], t2 - t1);
}

#endif // CUDA_SHARED_H
