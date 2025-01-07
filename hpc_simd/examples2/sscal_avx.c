#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <x86intrin.h>
#include <immintrin.h>

// an sscal assuming alignment
void sscal(int n, float a, float* x)
{
  // broadcast the scale factor into a register
  __m256 x0 = _mm256_broadcast_ss(&a);

  // we assume alignment to 32 bytes
  assert(((size_t) x) % 32 == 0);

  int ndiv8 = n/8;

  // loop over chunks of 8 values
  for (int i=0; i<ndiv8; ++i) {
    __m256 x1 = _mm256_load_ps(x+8*i);  // aligned (fast) load
    __m256 x2 = _mm256_mul_ps(x0,x1);   // multiply
    _mm256_store_ps(x+8*i,x2);          // store back aligned
  }

  // do the remaining entries
  for (int i=ndiv8*8 ; i< n ; ++i)
    x[i] *= a;
}

#define N 1024

__attribute__((aligned(32))) float x[N];

int main()
{
  // initialize a vector
  for (int i=0; i<N; ++i)
    x[i] = i;

  // call sscal
  printf("The address is %p\n", &x[0]);
  sscal(N, 4.f, &x[0]);

  // calculate error
  float d=0.;
  for (int i=0; i<N; ++i)
    d += fabs(x[i]-4.*i);
  printf("l1-norm of error: %lf\n", d);

  return 0;
}

