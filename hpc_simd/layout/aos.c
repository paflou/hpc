#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (128*1024*1024)

typedef struct {
	double x;
	double y;
	double z;
} point_t ;

point_t points[N];

int main()
{
	#pragma omp parallel
	#pragma omp master
	printf("running with %d threads\n", omp_get_num_threads());

	double result = 0;

	srand48(10);
	for (int i = 0; i < N ; ++ i) {
		points[i].x = drand48();
		points[i].y = drand48();
	}

	#pragma omp parallel for reduction(+:result)
	for (int i = 0; i < N; ++i) {
		result += (points[i].x);
	}

	result = 0;
	double t0 = omp_get_wtime();
	#pragma omp parallel for reduction(+:result)
	for (int i = 0; i < N; ++i) {
		result += (points[i].x);
	}
	double t1 = omp_get_wtime();

	printf("result = %f, elapsed time %lf\n", result/N, t1-t0);


	return 0;
}
