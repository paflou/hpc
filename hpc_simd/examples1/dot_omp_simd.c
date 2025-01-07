#include <stdio.h>
#include <omp.h>

#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

real sprod(real *a, real *b, int n)
{
	real sum = 0.0;
	for (int k=0; k<n; k++)
		sum += a[k] * b[k];

	return sum;
}


real sprod_simd(real *a, real *b, int n)
{
	real sum = 0.0;
#pragma omp simd reduction(+:sum)
	for (int k=0; k<n; k++)
		sum += a[k] * b[k];

	return sum;
}


#define N (32*1024)
real a[N] __attribute__ ((aligned(256)));
real b[N] __attribute__ ((aligned(256)));

int main(int argc, char *argv[])
{
	for (int i = 0; i < N; i++) a[i] = 1.0;
	for (int i = 0; i < N; i++) b[i] = 2.0;

	double t0, t1;

	real s0 = 0.0;
	for (int t = 0; t < 10000; t++)
	{
		s0 += sprod(a, b, N);
	}
  s0 = 0.0;
	t0 = omp_get_wtime();
	for (int t = 0; t < 10000; t++)
	{
		s0 += sprod(a, b, N);
	}
	t1 = omp_get_wtime();
	real t_elapsed = t1-t0;
	printf("s0 = %f, elapsed = %lf ms\n", s0, 1000.0*(t_elapsed));


	real s1 = 0.0;
	for (int t = 0; t < 10000; t++)
	{
		s1 += sprod_simd(a, b, N);
	}
	s1 = 0.0;
	t0 = omp_get_wtime();
	for (int t = 0; t < 10000; t++)
	{
		s1 += sprod_simd(a, b, N);
	}
	t1 = omp_get_wtime();
	real t_elapsed_simd = t1-t0;

	printf("s1 = %f, elapsed = %lf ms\n", s1, 1000.0*(t_elapsed_simd));

	printf("t / t_simd = %.2f\n", t_elapsed / t_elapsed_simd);
	return 0;
}
