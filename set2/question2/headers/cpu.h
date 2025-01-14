#include "def.h"

#ifndef CPU_H
#define CPU_H


void initializeCUDA()
{
    cudaMalloc((void **)&d_A, sizeof(double) * N * N);
    cudaMalloc((void **)&d_B, sizeof(double) * N * N);
    cudaMalloc((void **)&d_C, sizeof(double) * N * N);
    cudaMalloc((void **)&d_D, sizeof(double) * N * N);
    cudaMalloc((void **)&d_E, sizeof(double) * N * N);
    cudaMalloc((void **)&d_F, sizeof(double) * N * N);
    cudaMalloc((void **)&d_temp, sizeof(double) * N * N);
    cudaMalloc((void **)&d_temp2, sizeof(double) * N * N);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
}

void cleanupCUDA()
{
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_F);
    cudaFree(d_temp);
    cudaFree(d_temp2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

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
        c[i] = b[i] - a[i];
    }
}

void CPUmatMult(double *result, double *a, double *b)
{
    // Zero the result matrix first
    // memset(result, 0, N * N * sizeof(double));
    
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
        //printf("Index %d: a=%f, b=%f\n", i, a[i], b[i]);

        if (fabs(a[i] - b[i]) > 1e-5)
        {
            printf("Matrices are not equal at index %d: %f != %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    printf("Matrices are equal\n");
    return 1;
}

void computeCPU(double *A, double *B, double *C, double *D, double *E_cpu, double *F_cpu)
{
    double t1 = get_wtime();

    double *temp = (double *)malloc(N * N * sizeof(double));
    double *temp2 = (double *)malloc(N * N * sizeof(double));

    //avoid garbage values
    memset(temp, 0, N * N * sizeof(double));
    memset(temp2, 0, N * N * sizeof(double));
    memset(E_cpu, 0, N * N * sizeof(double));
    memset(F_cpu, 0, N * N * sizeof(double));

    CPUmatMult(E_cpu, A, C);
    CPUmatMult(temp, B, D);
    CPUvecSub(temp, E_cpu, E_cpu);

    CPUmatMult(temp2, A, D);
    CPUmatMult(F_cpu, B, C);
    CPUvecAdd(temp2, F_cpu, F_cpu);

    free(temp);
    free(temp2);

    double t2 = get_wtime();
    cpu_time = t2 - t1;
    printf("CPU: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n", E_cpu[N*N-1], F_cpu[N*N-1], t2 - t1);
}

#endif // CPU_H
