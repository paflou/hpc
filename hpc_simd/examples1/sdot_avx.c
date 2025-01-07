#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <smmintrin.h> //SSE4
#include <immintrin.h>
#include <math.h>


double gettime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void mat_fill(int N, float *mat, float val) {
  int i;

  for(i = 0; i < N; i ++)
    mat[i] = val;
}


float dot_vec(int N, float *a, float *b) {
  float arr[8];
  float total;
  int i;
  __m256 num1, num2, num3, num4;
  num4= _mm256_setzero_ps();  //sets sum to zero

  for(i=0; i<N; i+=8) {
    num1 = _mm256_loadu_ps(a+i); //loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
    num2 = _mm256_loadu_ps(b+i); //loads unaligned array b into num2  num2= b[3] b[2] b[1]  b[0]
#if 1
    num3 = _mm256_mul_ps(num1, num2); //performs multiplication num3 = a[3]*b[3]  a[2]*b[2]  a[1]*b[1]  a[0]*b[0]
    num4 = _mm256_add_ps(num4, num3);  //performs vertical addition
#else
    num4 = _mm256_fmadd_ps(num1, num2, num4);
#endif
  }
  //num4 = _mm256_hadd_ps(num4,num4);
  _mm256_store_ps(arr,num4);

  total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
  //total = arr[0] + arr[1] + arr[4] + arr[5];
  return total;
}


float dot_product(int N, float *a, float *b)
{
  int i;
  float prod=0.0;

  for(i=0; i<N; i++) {
    prod += a[i] * b[i];
  }
  return prod;
}

int main(int argc, char * argv[])
{
  int i;
  double time1 = 0;
  double time2 = 0;
  double t;
  float *A, *B;
  float prod1,prod2;
  int N = 32*1024;
  int Niter = 10000;

  if (argc > 1) N = atoi(argv[1]);

  //malloc aligned memory
  posix_memalign((void**)&A, 32, N * sizeof(float));
  posix_memalign((void**)&B, 32, N * sizeof(float));

  mat_fill(N, A, 1.0);
  mat_fill(N, B, 2.0);

  prod1 = 0;
  t = gettime();
  for (int i = 0; i < Niter; i++)
    prod1 += dot_product(N,A,B);
  t = gettime() - t;
  time1 = t;

  printf("Serial Dot Product\t %f sec\n",time1);

  prod2 = 0;
  t = gettime();
  for (int i = 0; i < Niter; i++)
    prod2 += dot_vec(N,A,B);
  t = gettime() - t;
  time2 = t;

  printf("SIMD Dot Product\t %f sec\n",time2);

  printf("%f, %f\n", prod1, prod2);

  free(A);
  free(B);

  return 0;
}
