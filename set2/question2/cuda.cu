#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>

#define N 16384
#define INDEX(i, j) (i * N + j)

double t1, t2;

double get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void CPUvecAdd(double *a, double *b, double *c)
{
    for (int i = 0; i < N * N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void CPUvecSub(double *a, double *b, double *c)
{
    for (int i = 0; i < N * N; i++)
    {
        c[i] = a[i] - b[i];
    }
}

void CPUmatMult(double *result, double *a, double *b)
{
    // Zero the result matrix first
    //memset(result, 0, N * N * sizeof(double));

    // Column-major ordering (j,i,k)
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                result[j * N + i] += a[k * N + i] * b[j * N + k];
            }
        }
    }
}

void initializeMatrix(double *matrix, unsigned int seed)
{
#pragma omp for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = INDEX(i, j);
            matrix[index] = (double)rand_r(&seed) / RAND_MAX;
            // matrix[index] = 5; // testing
        }
    }
}

void printMatrix(double *matrix)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("\t%f", matrix[INDEX(i, j)]);
        }
        printf("\n");
    }
}

int compareMatrices(double *a, double *b)
{
    for (int i = 0; i < N * N; i++)
    {
        if (fabs(a[i] - b[i]) > 1e-5)
        {
            printf("Matrices are not equal at index %d: %f != %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    printf("Matrices are equal\n");
    return 1;
}

__global__ void matMult(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        double sum = 0.0;
        for (int k = 0; k < N; k++)
        {
            sum += a[k * N + i] * b[j * N + k];
        }
        c[j * N + i] = sum;
    }
}
// ...existing code...
__global__ void vecSub(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        c[j * N + i] = b[j * N + i] - a[j * N + i];
    }
}

// ...existing code...
__global__ void vecAdd(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        c[j * N + i] = b[j * N + i] + a[j * N + i];
    }
}

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
    double *A = (double *)malloc(sizeof(double) * N * N);
    double *B = (double *)malloc(sizeof(double) * N * N);
    double *C = (double *)malloc(sizeof(double) * N * N);
    double *D = (double *)malloc(sizeof(double) * N * N);
    double *temp = (double *)malloc(sizeof(double) * N * N);

    double *E_blas = (double *)malloc(sizeof(double) * N * N);
    double *F_blas = (double *)malloc(sizeof(double) * N * N);
    double *E_cuda = (double *)malloc(sizeof(double) * N * N);
    double *F_cuda = (double *)malloc(sizeof(double) * N * N);
    double *E_CPU = (double *)malloc(sizeof(double) * N * N);
    double *F_CPU = (double *)malloc(sizeof(double) * N * N);

    // random seeds
    int seed[] = {1, 10, 100, 1000};
    initializeMatrix(A, seed[0]);
    initializeMatrix(B, seed[1]);
    initializeMatrix(C, seed[2]);
    initializeMatrix(D, seed[3]);

    // printf("First number of matrix A: %f \n", A[0]);
    // printf("First number of matrix B: %f \n", B[0]);
    // printf("First number of matrix C: %f \n", C[0]);
    // printf("First number of matrix D: %f \n", D[0]);

    // Device
    double *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;

    double *d_temp;

    cudaMalloc((void **)&d_temp, sizeof(double) * N * N);
    cudaMalloc((void **)&d_A, sizeof(double) * N * N);
    cudaMalloc(&d_B, sizeof(double) * N * N);
    cudaMalloc((void **)&d_C, sizeof(double) * N * N);
    cudaMalloc(&d_D, sizeof(double) * N * N);
    cudaMalloc((void **)&d_E, sizeof(double) * N * N);
    cudaMalloc(&d_F, sizeof(double) * N * N);

    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA kernel (cuBLAS)
    t1 = get_wtime();
    const double alpha = 1.0;
    const double beta = 0.0;
    const double reverse = -1.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_C, N, &beta, d_E, N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &reverse, d_B, N, d_D, N, &alpha, d_E, N);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_D, N, &beta, d_F, N);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_C, N, &alpha, d_F, N);

    cudaMemcpy(E_blas, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(F_blas, d_F, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    t2 = get_wtime();

    float cublas_time = t2 - t1;
    printf("cuBLAS CUDA : E[0]= %f | F[0] = %f, Took %.5f seconds\n", E_blas[0], F_blas[0], cublas_time);

    // CUDA kernel (custom)
    t1 = get_wtime();

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(4, 4);

    matMult<<<numBlocks, threadsPerBlock>>>(d_A, d_C, d_E);
    matMult<<<numBlocks, threadsPerBlock>>>(d_B, d_D, d_temp);
    vecSub<<<numBlocks, threadsPerBlock>>>(d_temp, d_E, d_E);

    matMult<<<numBlocks, threadsPerBlock>>>(d_A, d_D, d_temp);
    matMult<<<numBlocks, threadsPerBlock>>>(d_B, d_C, d_F);
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_temp, d_F, d_F);

    cudaDeviceSynchronize();

    cudaMemcpy(E_cuda, d_E, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(F_cuda, d_F, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    t2 = get_wtime();

    float cuda_time = t2 - t1;
    printf("Custom CUDA: E[0]= %f | F[0] = %f, Took %.5f seconds\n", E_cuda[0], F_cuda[0], cuda_time);

    printf("E: ");
    compareMatrices(E_cuda, E_blas);
    printf("F: ");
    compareMatrices(F_cuda, F_blas);

    // CPU

    t1 = get_wtime();
    // E = BD - AC
    //CPUmatMult(temp, A, C);        // temp = AC
    //CPUmatMult(E_CPU, B, D);       // E_CPU = BD
    //CPUvecSub(temp, E_CPU, E_CPU); // E_CPU = BD - AC
//
    //// F = AD + BC
    //CPUmatMult(temp, A, D);        // temp = AD
    //CPUmatMult(F_CPU, B, C);       // F_CPU = BC
    //CPUvecAdd(F_CPU, F_CPU, temp); // F_CPU = AD + BC
    t2 = get_wtime();

    float cpu_time = t2 - t1;
    printf("CPU: First val = %f, Took %.5f seconds\n", E_CPU[0], cpu_time);

    float speedup = cpu_time / cuda_time;
    printf("\nSpeedup: %f\n", speedup);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

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
