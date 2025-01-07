extern double work(int i);

void initialize(double *A, int N)
{
    #pragma omp parallel
    {
        #pragma omp single 
        {
            //unrolling with 2 iterations per task
            for (int i = 0; i < N-1; i+=2) {
                #pragma omp task firstprivate(i)
                {
                    A[i] = work(i);
                    A[i+1] = work(i+1);
                }
            }
            //Handle remaining element if N is odd
            if (N % 2) {
                #pragma omp task
                {
                    A[N-1] = work(N-1);
                }
            }
        }
    }
}