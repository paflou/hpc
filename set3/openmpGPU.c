#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N 512
double t1, t2;

double get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
void initializeMatrix(double *matrix, unsigned int seed)
{
#pragma omp for collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int index = j * N + i;
            matrix[index] = (double)rand_r(&seed) / RAND_MAX;
            // matrix[index] = 5; // testing
        }
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
    // Column-major ordering (j,i,k)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[k * N + i] * b[j * N + k];
            }
            result[j * N + i] = sum;
        }
    }
}
void GPUvecAdd(double *a, double *b, double *c)
{
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < N * N; i++)
    {
        c[i] = a[i] + b[i];
    }
}
void GPUvecSub(double *a, double *b, double *c)
{
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < N * N; i++)
    {
        c[i] = b[i] - a[i];
    }
}
void GPUmatMult(double *result, double *a, double *b)
{    
    // Column-major ordering (j,i,k)
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[k * N + i] * b[j * N + k];
            }
            result[j * N + i] = sum;
        }
    }
    
}

int main(int argc, char **argv)
{

	double *A = (double *)malloc(N * N * sizeof(double));
	double *B = (double *)malloc(N * N * sizeof(double));
	double *C = (double *)malloc(N * N * sizeof(double));
	double *D = (double *)malloc(N * N * sizeof(double));
	double *E = (double *)malloc(N * N * sizeof(double));
	double *F = (double *)malloc(N * N * sizeof(double));
	double *E_gpu = (double *)malloc(N * N * sizeof(double));
	double *F_gpu = (double *)malloc(N * N * sizeof(double));

    int seed[] = {1, 10, 100, 1000};
    initializeMatrix(A, seed[0]);
    initializeMatrix(B, seed[1]);
    initializeMatrix(C, seed[2]);
    initializeMatrix(D, seed[3]);


    double *temp = (double *)malloc(N * N * sizeof(double));

	t1 = get_wtime();
    #pragma omp target data map(to: A, B, C, D) map(alloc: temp) map(from: E_gpu, F_gpu)
    {
        GPUmatMult(E_gpu, A, C);
        GPUmatMult(temp, B, D);
        GPUvecSub(temp, E_gpu, E_gpu);

        GPUmatMult(temp, A, D);
        GPUmatMult(F_gpu, B, C);
        GPUvecAdd(temp, F_gpu, F_gpu);
    }
	t2 = get_wtime();

	printf("openMP GPU: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n", E_gpu[N*N-1], F_gpu[N*N-1], t2 - t1);

    //avoid garbage values
    memset(temp, 0, N * N * sizeof(double));
    memset(E, 0, N * N * sizeof(double));
    memset(F, 0, N * N * sizeof(double));

	double t1 = get_wtime();
    CPUmatMult(E, A, C);
    CPUmatMult(temp, B, D);
    CPUvecSub(temp, E, E);

    CPUmatMult(temp, A, D);
    CPUmatMult(F, B, C);
    CPUvecAdd(temp, F, F);
    double t2 = get_wtime();

    printf("CPU: E[N*N-1]= %f | F[N*N-1] = %f \n Took %.5f seconds\n", E[N*N-1], F[N*N-1], t2 - t1);
    

	compareMatrices(E_gpu, E);
    compareMatrices(F_gpu, F);
	
	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(temp);
    return 0;
}
