#include <stdio.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6
#endif

#include "weno.h"

double	t1, t2;

float * myalloc(const int NENTRIES, const int verbose )
{
	const int initialize = 1;
	enum { alignment_bytes = 32 } ;
	float * tmp = NULL;

	const int result = posix_memalign((void **)&tmp, alignment_bytes, sizeof(float) * NENTRIES);
	assert(result == 0);

	if (initialize)
	{
		for(int i=0; i<NENTRIES; ++i)
			tmp[i] = drand48();

		if (verbose)
		{
			for(int i=0; i<NENTRIES; ++i)
				printf("tmp[%d] = %f\n", i, tmp[i]);
			printf("==============\n");
		}
	}
	return tmp;
}

double get_wtime()
{
	struct timeval t;
	gettimeofday(&t,  NULL);
	return t.tv_sec + t.tv_usec*1e-6;
}

void check_error(const double tol, float ref[], float val[], const int N)
{
	static const int verbose = 0;

	for(int i=0; i<N; ++i)
	{
		assert(!isnan(ref[i]));
		assert(!isnan(val[i]));

		const double err = ref[i] - val[i];
		const double relerr = err/fmaxf(FLT_EPSILON, fmaxf(fabs(val[i]), fabs(ref[i])));

		if (verbose) printf("+%1.1e,", relerr);

		if (fabs(relerr) >= tol && fabs(err) >= tol)
			printf("\n%d: %e %e -> %e %e\n", i, ref[i], val[i], err, relerr);

		assert(fabs(relerr) < tol || fabs(err) < tol);
	}

	if (verbose) printf("\t");
}


void benchmark(int argc, char *argv[], const int NENTRIES_, const int NTIMES, const int verbose, char *benchmark_name)
{
	const int NENTRIES = 8 * (NENTRIES_ / 4);

	printf("nentries set to %e\n", (float)NENTRIES);

	float * const a = myalloc(NENTRIES, verbose);
	float * const b = myalloc(NENTRIES, verbose);
	float * const c = myalloc(NENTRIES, verbose);
	float * const d = myalloc(NENTRIES, verbose);
	float * const e = myalloc(NENTRIES, verbose);
	float * const f = myalloc(NENTRIES, verbose);
	float * const gold = myalloc(NENTRIES, verbose);
	float * const result = myalloc(NENTRIES, verbose);

	t1 = get_wtime();
	weno_minus_reference(a, b, c, d, e, gold, NENTRIES);
	weno_minus_reference(a, b, c, d, e, result, NENTRIES);
	t2 = get_wtime();

	printf("\n\n\nWeno time: %f\n\n\n", t2-t1);

	const double tol = 1e-5;
	printf("minus: verifying accuracy with tolerance %.5e...", tol);
	check_error(tol, gold, result, NENTRIES);
	printf("passed!\n");

	free(a);
	free(b);
	free(c);
	free(d);
	free(e);
	free(gold);
	free(result);
}

int main (int argc, char *  argv[])
{
	printf("Hello, weno benchmark!\n");
	const int debug = 0;
	int times = 4;

	if(argc > 1) times = atoi(argv[1]);

	if (debug)
	{
		benchmark(argc, argv, 4, 1, 1, "debug");
		return 0;
	}

	{
        const int nentries = 128;  // Small number of entries
        const int ntimes = 100000; // High repetitions to observe performance

        for(int i = 0; i < times; ++i)
        {
            printf("*************** 0%% CACHE UTILIZATION (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache-0%");
        }
    }

    /* 25% cache utilization */
    {
        const int nentries = 16 * (int)(pow(32 + 6, 2) * 4 * 0.25); // 25% of 50%
        const int ntimes = (int)floor(2. / (1e-7 * nentries));

        for(int i = 0; i < times; ++i)
        {
            printf("*************** 25%% CACHE UTILIZATION (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache-25%");
        }
    }

    /* 50% cache utilization (original benchmark) */
    {
        const int nentries = 16 * (int)(pow(32 + 6, 2) * 4 * 0.5);
        const int ntimes = (int)floor(2. / (1e-7 * nentries));

        for(int i = 0; i < times; ++i)
        {
            printf("*************** 50%% CACHE UTILIZATION (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache-50%");
        }
    }

    /* 75% cache utilization */
    {
        const int nentries = 16 * (int)(pow(32 + 6, 2) * 4 * .75); // 75% of 50%
        const int ntimes = (int)floor(2. / (1e-7 * nentries));

        for(int i = 0; i < times; ++i)
        {
            printf("*************** 75%% CACHE UTILIZATION (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache-75%");
        }
    }

    /* 100% cache utilization (original peak-like benchmark) */
    {
        const int nentries = 16 * (int)(pow(32 + 6, 2) * 4);
        const int ntimes = (int)floor(2. / (1e-7 * nentries));

        for(int i = 0; i < times; ++i)
        {
            printf("*************** PEAK-LIKE BENCHMARK (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, ntimes, 0, "cache-peak");
        }
    }

    /* STREAM-LIKE BENCHMARK (Original) */
    {
        const double desired_mb = 128 * 4;
        const int nentries = (int)floor(desired_mb * 1024. * 1024. / 7 / sizeof(float));

        for(int i = 0; i < times; ++i)
        {
            printf("*************** STREAM-LIKE BENCHMARK (RUN %d) **************************\n", i);
            benchmark(argc, argv, nentries, 1, 0, "stream");
        }
    }


    return 0;
}
