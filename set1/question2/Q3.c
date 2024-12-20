extern double work(int i);

void initialize(double *A, int N) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task
                {
                    A[i] = work(i);
                }
            }
        }
    }
}
