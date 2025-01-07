#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <x86intrin.h>
//Compile gcc -msse4 -O3

double gettime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}


#define N 1024*1024
#define REPETITIONS 1024

// a saxpy version without explicit SSE
void saxpy_aligned(int n, float a, float* __restrict__ x, float* __restrict__ y)
{
  __builtin_assume_aligned(x,32);
  __builtin_assume_aligned(y,32);

  for (int i=0; i<n; ++i)
    y[i] += a*x[i];
}

// a saxpy version requiring alignment
void saxpy(int n, float a, float* x, float* y)
{
  // load the scale factor four times into a regsiter
  __m256 x0 = _mm256_broadcast_ss(&a);

  // we assume alignment
  assert(((size_t)x) % 32 == 0 && ((size_t)y) % 32 == 0);

  // loop over chunks of 4 values
  int ndiv8 = n/8;
  for (int i=0; i<ndiv8; ++i) {
    __m256 x1 = _mm256_load_ps(x+8*i); // aligned (fast) load
    __m256 x2 = _mm256_load_ps(y+8*i); // aligned (fast) load
    __m256 x3 = _mm256_mul_ps(x0,x1);  // multiply
    __m256 x4 = _mm256_add_ps(x2,x3);  // add
    _mm256_store_ps(y+8*i,x4);         // store back aligned
  }

  // do the remaining entries
  for (int i=ndiv8*8 ; i< n ; ++i)
    y[i] += a*x[i];
}


__attribute__((aligned(64))) float x[N];
__attribute__((aligned(64))) float y[N];


int main()
{
  // initialize two vectors
  for (int i=0; i<N; ++i)
    x[i] = 1.;

  for (int i=0; i<N; ++i)
    y[i] = 2.;

  // call saxpy and time it
  double start, end;
  start = gettime();
  for (int it=0; it<REPETITIONS; ++it )
    saxpy(N, 4.0/REPETITIONS, &x[0], &y[0]);
  end = gettime();
  double elapsed_time = 1e6*(end-start);

  // calculate error
  float d=0.0;
  for (int i=0; i<N; ++i)
    d += fabs(y[i]-6.0);

  printf("y[0]=%lf\n", y[0]);

  printf("elapsed time: %lf mus\n", elapsed_time/REPETITIONS);
  printf("l1-norm of error: %lf\n", d);
}

