#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <immintrin.h>

// a dscal assuming alignment
void dscal(int n, double a, double* x)
{
  // broadcast the scale factor into a register
  __m256d x0 = _mm256_broadcast_sd(&a);

  // we assume alignment
  size_t xv = (size_t)(&x);
  assert(xv % 16 == 0);

  int ndiv4 = n/4;

  // loop over chunks of 4 values
  for (int i=0; i<ndiv4; ++i) {
    __m256d x1 = _mm256_load_pd(x+4*i);  // aligned (fast) load
    __m256d x2 = _mm256_mul_pd(x0,x1);   // multiply
    _mm256_store_pd(x+4*i,x2);          // store back aligned
  }

  // do the remaining entries
  for (int i=ndiv4*4 ; i< n ; ++i)
    x[i] *= a;
}


#define N 1024

__attribute__((aligned(32))) double x[N];

int main()
{
  // initialize a vector
  for (int i=0; i<N; ++i)
    x[i] = i;

  // call sscal
  printf("The address is %p\n", &x[0]);
  dscal(N, 4.0, &x[0]);

  // calculate error
  double d=0.;
  for (int i=0; i<N; ++i)
    d += fabs(x[i]-4.*i);
  printf("l1-norm of error: %lf\n", d);

  return 0;
}

