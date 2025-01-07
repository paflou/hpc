#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>

int T;

int for_loop(int unique_num, int *prev) {
    int sum = 0;
    //ensure that the threads are ordered
    #pragma omp for ordered nowait
    for(int i=0; i < T; i++) {
        #pragma omp ordered
        {
            sum = *prev;
            *prev += unique_num;
        }
    }
    return sum;
}

int MPI_Exscan_omp(int size, int rank, int unique_num, int *prev) {
    int sum;
    int next = rank + 1;
    int thread_num = omp_get_thread_num();
        
    if(rank==0){
        sum = for_loop(unique_num, prev);
        if(thread_num == T - 1)
            MPI_Send(prev, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, unique_num, sum);

        } else if(rank==size-1) {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum = for_loop(unique_num, prev);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, unique_num, sum);

        } else {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum = for_loop(unique_num, prev);

            if(thread_num == T - 1)
                MPI_Send(prev, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, unique_num, sum);

        }
        return sum;
        //printf("rank %d computes %d\n",rank, sum);
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0) {
        printf("Usage: %s <number of threads>\n", argv[0]);
        return 1;
    }
    T = atoi(argv[1]);

    int prev = 0;
    int sum[T];
    #pragma omp parallel num_threads(T) shared(prev)
    {
        int thread_num = omp_get_thread_num();
        int unique_num = thread_num + rank * T;

        //simutaneously start all threads
        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
        sum[thread_num] = MPI_Exscan_omp(size, rank, unique_num, &prev);
    }

    printf("rank %d sum: \t", rank);
    for(int i = 0; i < T; i++) {
        printf("%d \t", sum[i]);
    }
    printf("\n");

    MPI_Finalize();
}
