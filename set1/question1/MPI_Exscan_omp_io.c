#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>
#include <math.h>

#define CACHE_LINE_SIZE 64

int T, N;

#define INDEX(i, j, k) ((i) * N * N + (j) * N + (k))

void MPI_Exscan_omp_io(int size, int rank, int matrix_size, int *sum)
{
    int lsum = 0;
    int thread_num = omp_get_thread_num();

    // each thread computes their local sum serially
    for (int i = 0; i < thread_num; i++)
    {
        lsum += matrix_size;
    }
    sum[thread_num] = lsum;

#pragma omp barrier // wait until all threads are finished
    //     #pragma omp single
    //         printf("rank %d, largest thread val = %d\n", rank, sum[T - 1]);

    // compute the prefix sum in parallel
    for (int step = 1; step < size; step++)
    {
        int partial = 0;
        if (rank - step >= 0)
        {
            MPI_Recv(&partial, 1, MPI_INT, rank - step, rank + thread_num, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("thread %d of rank %d recieved %d\n", thread_num, rank, partial);
            sum[thread_num] += partial;
        }
        if (rank + step < size)
        {
            // send your value along with the value of thread T-1
            // e.g  P0T0 -> P1T0    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //      P0T1 -> P1T1    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //      P0T2 -> P1T2    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //      P0T3 -> P1T3    matrix_size + sum[3] (values of 0, 1, 2 at once)

            int send_val = sum[T - 1] + matrix_size;
            MPI_Send(&send_val, 1, MPI_INT, rank + step, rank + step + thread_num, MPI_COMM_WORLD);
        }
    }
    // printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num] * (int)sizeof(double));
}

void initializeMatrix(double *matrix, unsigned int seed)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                int index = INDEX(i, j, k);
                matrix[index] = (double)rand_r(&seed) / RAND_MAX;

                // testing
                // printf("index: %d", index);
                // matrix[index] =   100;
                // printf("matrix[%d] = %f\n",index, matrix[index]);
            }
        }
    }
}

int checkMatrix(MPI_File file, unsigned int *seed, int offset)
{
    MPI_Status status;
    double *values = (double *)malloc(N * N * N * sizeof(double));
    int error_found = 0;

    MPI_File_read_at_all(file, offset, values, N * N * N, MPI_DOUBLE, &status);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                if (error_found)
                    continue;

                double val = (double)rand_r(seed) / RAND_MAX;
                // val = 100;

                int index = INDEX(i, j, k);

                if (fabs(values[index] - val) > 0.0001)
                {
#pragma omp atomic write
                    error_found = 1;

                    printf("ERROR DETECTED: values[%d] = %.5f, expected %.5f [ thread %d ]\n", index, values[index], val, omp_get_thread_num());
                }
            }
        }
    }
    free(values);
    return error_found;
}

int main(int argc, char *argv[])
{
    MPI_File file;
    const char *filename = "output.bin";
    MPI_Status status;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3)
    {
        printf("Usage: %s <number of threads> <N (matrix = N*N*N)>\n", argv[0]);
        return 1;
    }
    T = atoi(argv[1]);
    N = atoi(argv[2]);

    int local_flag = 0;
    int global_flag;

    // deletes any previous file
    if (rank == 0)
    {
        MPI_File_delete("output.bin", MPI_INFO_NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);

    int sum[T];
#pragma omp parallel num_threads(T) shared(sum)
    {
        int unique_num = omp_get_thread_num() + rank * T;
        unsigned int seed = unique_num;
        int thread_num = omp_get_thread_num();

        // use 1d array for the matrix (easier + better memory locality)
        double *matrix = (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

        int matrixSize = N * N * N;

// simutaneously start all threads
#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Exscan_omp_io(size, rank, matrixSize, sum);

        int start = sum[thread_num];
        int end = start + matrixSize;
        start *= sizeof(double);
        end *= sizeof(double);

        usleep(unique_num * 1000);
        printf("thread %d begins writing at %d and ends at %d. 1st val = %f\n", unique_num, start, end, matrix[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_write_at_all(file, start, matrix, matrixSize, MPI_DOUBLE, &status);
        // printf("thread %d finished.\n", unique_num);

#pragma omp single
        if (checkMatrix(file, &seed, start))
        {
#pragma omp critical
            local_flag = 1;
        }
        free(matrix);
    }

    MPI_File_close(&file);
    MPI_Reduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        global_flag ? printf("\nThe binary file is wrong\n") : printf("\nThe binary file is correct\n");
    }
    MPI_Finalize();
    return 0;
}
