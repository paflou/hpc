#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (128*1024*1024)

struct {
	double x[N];
	double y[N];
	double z[N];
} points;

int main()
{
	#pragma omp parallel
	#pragma	omp master
	printf("running	with %d threads\n", omp_get_num_threads());

	double result = 0;

	srand48(10);
	for (int i = 0; i < N ; ++ i) {
		points.x[i] = drand48();
		points.y[i] = drand48();
	}

	#pragma omp parallel for reduction(+:result)
	for (int i = 0; i < N ; ++i) {
		result += (points.x[i]);
	}

	result = 0;
	double t0 = omp_get_wtime();
	#pragma omp parallel for reduction(+:result)
	for (int i = 0; i < N ; ++i) {
		result += (points.x[i]);
	}
	double t1 = omp_get_wtime();

	printf("result = %f, elapsed time %lf\n", result/N, t1-t0);
	return 0;
}
