#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <smmintrin.h> //SSE4

//Compile gcc -msse4 -O3

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
  __attribute__ ((aligned(32))) float arr[4];
  float total;
  int i;
  __m128 num1, num2, num3, num4;
  //num4= _mm_setzero_ps();  //sets sum to zero
  num4 = _mm_set1_ps(0.);

  for(i=0; i<N; i+=4) {
    num1 = _mm_load_ps(a+i); //loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
    num2 = _mm_load_ps(b+i); //loads unaligned array b into num2  num2= b[3] b[2] b[1]  b[0]
    num3 = _mm_mul_ps(num1, num2); //performs multiplication num3 = a[3]*b[3]  a[2]*b[2]  a[1]*b[1]  a[0]*b[0]
    num4 = _mm_add_ps(num4, num3);  //performs vertical addition
  }

  _mm_store_ps(arr, num4);

  //printf("%f %f %f %f\n", arr[0], arr[1], arr[2], arr[3]);
  total = arr[0] + arr[1] + arr[2] + arr[3];
  return total;
}

float dot_vec_v1(int N, float *a, float *b)
{
  //float arr[4];
  //__attribute__ ((aligned(32))) 
  float total;
  int i;
  __m128 num1, num2, num3, num4;
  num4 = _mm_setzero_ps();  //sets sum to zero
  for(i=0; i<N; i+=4) {
    num1 = _mm_loadu_ps(a+i); //loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
    num2 = _mm_loadu_ps(b+i); //loads unaligned array b into num2  num2= b[3] b[2] b[1]  b[0]
    num3 = _mm_mul_ps(num1, num2); //performs multiplication num3 = a[3]*b[3]  a[2]*b[2]  a[1]*b[1]  a[0]*b[0]

    num3 = _mm_hadd_ps(num3, num3); //performs horizontal addition 
          //num3=  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]
    num4 = _mm_add_ps(num4, num3);  //performs vertical addition
  }

  num4 = _mm_hadd_ps(num4, num4);
	//printf("%f\n", num4[0]);
  _mm_store_ss(&total,num4);
	//total = num4[0];
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
  double time3 = 0;
  double t;
  float *A, *B;
  float prod1,prod2,prod3;
  int N = 32*1024;
  int Niter = 10000;

  if (argc > 1) N = atoi(argv[1]);

  //malloc aligned memory
  posix_memalign((void**)&A, 16, N * sizeof(float));
  posix_memalign((void**)&B, 16, N * sizeof(float));

  mat_fill(N, A, 1.0);
  mat_fill(N, B, 2.0);

  prod1 = 0;
  t = gettime();
  for (int t = 0; t < Niter; t++)
    prod1 = dot_product(N,A,B);
  t = gettime() - t;
  time1 = t;

  printf("Serial Dot Product\t %f sec\n",time1);

  prod2 = 0;
  t = gettime();
  for (int t = 0; t < Niter; t++)
    prod2 += dot_vec(N,A,B);
  t = gettime() - t;
  time2 = t;

  printf("SIMD Dot Product\t %f sec\n",time2);

  prod3 = 0;
  t = gettime();
  for (int t = 0; t < Niter; t++)
    prod3 += dot_vec_v1(N,A,B);
  t = gettime() - t;
  time3 = t;

  printf("SIMD Dot Product v1\t %f sec\n", time3);
  printf("%f, %f, %f\n", prod1, prod2, prod3);

  free(A);
  free(B);

  return 0;
}
