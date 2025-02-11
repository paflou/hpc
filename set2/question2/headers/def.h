#ifndef DEF_H
#define DEF_H

extern const int N;
#define INDEX(i, j) (i * N + j)
#define TILE_SIZE 8 // Add at top with other defines


extern double cpu_time, cublas_time, cuda_global_time, cuda_shared_time;
extern double *d_A, *d_B, *d_C, *d_D, *d_E, *d_F, *d_temp, *d_temp2;

#endif // DEF_H