#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>

#define T 8

void for_loop(int global_num, int *prev, int *sum, int rank) {
    #pragma omp for ordered schedule(static, 1)
    for(int i=0; i < T; i++) {
        #pragma omp ordered
        {
        *sum += global_num - 1  + *prev;
        #pragma omp critical
        *prev = *sum;
        printf("thread %d of %d (global num %d) computes %d\n", omp_get_thread_num(),rank, global_num, *sum);
        }
    }
}

void MPI_Exscan_pt2pt(int size, int rank, int global_num, int *prev) {
    int sum = 0;
        int next = rank + 1;
        int thread_num = omp_get_thread_num();

        if(rank==0){
            for_loop(global_num, prev, &sum, rank);
            if(thread_num == T - 1)
                MPI_Send(&sum, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, global_num, sum);

        } else if(rank==size-1) {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for_loop(global_num, prev, &sum, rank);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, global_num, sum);

        } else {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for_loop(global_num, prev, &sum, rank);

            if(thread_num == T - 1)
                MPI_Send(&sum, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
            //printf("thread %d of %d (global num %d) computes %d\n", thread_num, rank, global_num, sum);

        }

        //printf("rank %d computes %d\n",rank, sum);
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //offset for first thread
    int prev = 1;

    #pragma omp parallel num_threads(T) shared(prev)
    {
        int global_num = omp_get_thread_num() + rank * T;
        MPI_Exscan_pt2pt(size, rank, global_num, &prev);
    }
    MPI_Finalize();
}
