#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//Compile gcc -msse4 -O3

double gettime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}


#define N 1024*1024
#define REPETITIONS 1024

// a saxpy version without explicit SSE
void saxpy(int n, float a, float* x, float* y)
{
  for (int i=0; i<n; ++i)
    y[i] += a*x[i];
}

// initialize two vectors
float x[N];
float y[N];


int main()
{
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

