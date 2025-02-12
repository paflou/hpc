#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>

#define CACHE_LINE_SIZE 64

int T;

void MPI_Exscan_omp(int size, int rank, int values[][CACHE_LINE_SIZE], int sum[][CACHE_LINE_SIZE])
{
    int lsum = 0;
    int thread_num = omp_get_thread_num();

    // each thread computes their local sum serially
    for (int i = 0; i < thread_num; i++)
    {
        lsum += values[i][0];
    }
    sum[thread_num][0] = lsum;

    usleep(thread_num * 1000 * rank);
    //printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num][0]);
    #pragma omp barrier
    //     #pragma omp single
    //         printf("rank %d, largest thread val = %d\n", rank, sum[T - 1]);

    // compute the prefix sum in parallel
    for (int step = 1; step < size; step++)
    {

        int partial = 0;
        if (rank - step >= 0)
        {
            MPI_Recv(&partial, 1, MPI_INT, rank - step, rank + thread_num, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           // printf("thread %d of rank %d recieved %d from %d\n", thread_num, rank, partial, rank - step);

            sum[thread_num][0] += partial;
        }

        if (rank + step < size)
        {
            int send_val = sum[T - 1][0] + values[T - 1][0];
            
            printf("thread %d of rank %d sends %d to %d\n", thread_num, rank, send_val, rank + step);
            MPI_Bsend(&send_val, 1, MPI_INT, rank + step, rank + step + thread_num, MPI_COMM_WORLD);
        }
    }
    //printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num]);
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0)
    {
        printf("Usage: %s <number of threads>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    T = atoi(argv[1]);

    int sum[T][CACHE_LINE_SIZE];
    int values[T][CACHE_LINE_SIZE];

#pragma omp parallel num_threads(T) shared(sum, values)
    {
        int thread_num = omp_get_thread_num();
        int unique_num = thread_num + rank * T;
        values[thread_num][0] = unique_num;

// simutaneously start all threads
#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Exscan_omp(size, rank, values, sum);
    }

    // print the prefix sum in order
    usleep(rank * 1000);
    printf("\n rank %d sum: \t", rank);
    for (int i = 0; i < T; i++)
    {
        printf("%d \t", sum[i][0]);
    }
    printf("\n");

    MPI_Finalize();
}
