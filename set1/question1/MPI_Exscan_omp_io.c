#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <math.h>
#include <unistd.h>

#define CACHE_LINE_SIZE 64

int T, N;
int recv_val = 0;

#define INDEX(i, j, k) ((i) * N * N + (j) * N + (k))

void MPI_Exscan_omp_io(int size, int rank, int values, int sum[][CACHE_LINE_SIZE])
{
    int thread_num = omp_get_thread_num();
    int lsum = values * thread_num;

    sum[thread_num][0] = lsum;

// printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num][0]);
#pragma omp barrier
    for (int step = 1; step < size; step *= 2)
    {
        int send_partner = rank + step;
        int recv_partner = rank - step;

        if (send_partner < size)
        {
            int send_val = sum[T - 1][0] + values;

            // printf("thread %d of rank %d sends %d to %d\n", thread_num, rank, send_val, rank + step);
            if (omp_get_thread_num() == 0)
                MPI_Bsend(&send_val, 1, MPI_INT, send_partner, 0, MPI_COMM_WORLD);
        }
        if (recv_partner >= 0)
        {
            if (omp_get_thread_num() == 0)
                MPI_Recv(&recv_val, 1, MPI_INT, recv_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("thread %d of rank %d recieved %d from %d\n", thread_num, rank, partial, rank - step);

#pragma omp barrier
#pragma omp for
            for (int i = 0; i < T; i++)
            {
                sum[i][0] += recv_val;
            }
#pragma omp barrier
        }
    }
    // printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num][0]);
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

    MPI_File_read_at(file, offset, values, N * N * N, MPI_DOUBLE, &status);
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

    int buffer_size = sizeof(double) + MPI_BSEND_OVERHEAD;
    char *buffer = malloc(buffer_size * sizeof(char));
    MPI_Buffer_attach(buffer, buffer_size);

    int matrixSize = N * N * N;
    int sum[T][CACHE_LINE_SIZE];
#pragma omp parallel num_threads(T) shared(sum)
    {
        int unique_num = omp_get_thread_num() + rank * T;
        unsigned int seed = unique_num;
        int thread_num = omp_get_thread_num();

        // use 1d array for the matrix (easier + better memory locality)
        double *matrix = (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

// simutaneously start all threads
#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Exscan_omp_io(size, rank, matrixSize, sum);

        int start = sum[thread_num][0];
        int end = start + matrixSize;
        start *= sizeof(double);
        end *= sizeof(double);
        end--;
        
        //usleep(unique_num * 1000);
        //printf("thread %d begins writing at %d and ends at %d. 1st val = %f\n", unique_num, start, end, matrix[0]);

        #pragma omp single 
        {
            MPI_Barrier(MPI_COMM_WORLD);
            printf("Writing to file...\n");
        }
        MPI_File_write_at(file, start, matrix, matrixSize, MPI_DOUBLE, &status);
        // printf("thread %d finished.\n", unique_num);

        if (checkMatrix(file, &seed, start))
        {
#pragma omp critical
            local_flag = 1;
        }
        free(matrix);
    }
    MPI_File_close(&file);
    MPI_Reduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Buffer_detach(&buffer, &buffer_size);
    printf("Checking binary file...\n");
    if (rank == 0)
    {
        global_flag ? printf("\nThe binary file is wrong\n") : printf("\nThe binary file is correct\n");
    }
    MPI_Finalize();
    return 0;
}
