#include <stdio.h>

#define N 2

__global__ void add(int *a, int *b, int *c){
    printf("Hello World from block %d thread %d!\n", blockIdx.x, threadIdx.x);

    a[blockIdx.x] = b[blockIdx.x] + c[blockIdx.x];
}

int main() {
    int A[N][N];
    int B[N][N];
    int C[N][N];
    int D[N][N];
    
    double **A = malloc(sizeof(double) * N);
    double **B = malloc(sizeof(double) * N);
    double **C = malloc(sizeof(double) * N);
    double **D = malloc(sizeof(double) * N);

    for(int i=0;i<N;i++) {
        
    }
    // E[N][N] = (A*C - B*D);
    // F[N][N] = (A*D + B*C);

    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    for(int i = 0; i < N; i++) {
        b[i] = i;
        c[i] = i;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    add<<<N,1>>>(d_a, d_b, d_c); 
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++) {
        printf("%d = %d + %d\n", a[i],b[i],c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
