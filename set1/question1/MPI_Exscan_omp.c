#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>

#define CACHE_LINE_SIZE 64

int T;
int recv_val = 0;

void MPI_Exscan_omp(int size, int rank, int values[][CACHE_LINE_SIZE], int sum[][CACHE_LINE_SIZE], MPI_Op op)
{
    int lsum = 0;
    int thread_num = omp_get_thread_num();

    // each thread computes their local sum serially
    if (op == MPI_SUM)
    {
        for (int i = 0; i < thread_num; i++)
            lsum += values[i][0];
    }
    else if (op == MPI_PROD)
    {
        for (int i = 0; i < thread_num; i++)
            lsum *= values[i][0];
    }
    else if (op == MPI_MAX)
    {
        for (int i = 0; i < thread_num; i++)
            lsum = lsum > values[i][0] ? lsum : values[i][0];
    }
    else if (op == MPI_MIN)
    {
        for (int i = 0; i < thread_num; i++)
            lsum = lsum < values[i][0] ? lsum : values[i][0];
    }
    else
    {
        printf("Invalid MPI_Op\n");
        return;
    }

    sum[thread_num][0] = lsum;

// printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num][0]);
#pragma omp barrier
    // compute the prefix sum in parallel
    for (int step = 1; step < size; step *= 2)
    {
        int send_partner = rank + step;
        int recv_partner = rank - step;

#pragma omp single
        {
            if (send_partner < size)
            {
                int send_val = -1;
                if (op == MPI_SUM)
                    send_val = sum[T - 1][0] + values[T - 1][0];
                else if (op == MPI_PROD)
                    send_val = sum[T - 1][0] * values[T - 1][0];
                else if (op == MPI_MAX)
                    send_val = sum[T - 1][0] > values[T - 1][0] ? sum[T - 1][0] : values[T - 1][0];
                else
                    send_val = sum[T - 1][0] < values[T - 1][0] ? sum[T - 1][0] : values[T - 1][0];

                MPI_Bsend(&send_val, 1, MPI_INT, send_partner, 0, MPI_COMM_WORLD);
            }
        }
        if (recv_partner >= 0)
        {
#pragma omp single
            MPI_Recv(&recv_val, 1, MPI_INT, recv_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Add received value to all threads' sums

            if (op == MPI_SUM)
            {
#pragma omp barrier
#pragma omp for
                for (int i = 0; i < T; i++)
                    sum[i][0] += recv_val;
            }
            else if (op == MPI_PROD)
            {
#pragma omp barrier
#pragma omp for
                for (int i = 0; i < T; i++)
                    sum[i][0] *= recv_val;
            }
            else if (op == MPI_MAX)
            {
#pragma omp barrier
#pragma omp for
                for (int i = 0; i < T; i++)
                    sum[i][0] = sum[i][0] > recv_val ? sum[i][0] : recv_val;
            }
            else
            {
#pragma omp barrier
#pragma omp for
                for (int i = 0; i < T; i++)
                    sum[i][0] = sum[i][0] < recv_val ? sum[i][0] : recv_val;
            }
        }
    }

    // printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num]);
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (provided < MPI_THREAD_SINGLE)
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

    int buffer_size = sizeof(double) + MPI_BSEND_OVERHEAD;
    char *buffer = malloc(buffer_size * sizeof(char));
    MPI_Buffer_attach(buffer, buffer_size);

#pragma omp parallel num_threads(T) shared(sum, values)
    {
        int thread_num = omp_get_thread_num();
        int unique_num = thread_num + rank * T;
        values[thread_num][0] = unique_num;

// simutaneously start all threads
#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Exscan_omp(size, rank, values, sum, MPI_SUM);
    }

    // print the prefix sum in order
    usleep(rank * 1000);
    printf("\n rank %d sum: \t", rank);
    for (int i = 0; i < T; i++)
    {
        printf("%d \t", sum[i][0]);
    }
    printf("\n");
    MPI_Buffer_detach(&buffer, &buffer_size);
    MPI_Finalize();
}
