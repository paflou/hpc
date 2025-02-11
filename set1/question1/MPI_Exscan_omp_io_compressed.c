#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>
#include <zlib.h>
#include <math.h>

#define BUFFER 7000000
#define INDEX(i, j, k) ((i) * N * N + (j) * N + (k))

int T, N;

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
            //  e.g  P0T0 -> P1T0    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //       P0T1 -> P1T1    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //       P0T2 -> P1T2    matrix_size + sum[3] (values of 0, 1, 2 at once)
            //       P0T3 -> P1T3    matrix_size + sum[3] (values of 0, 1, 2 at once)

            int send_val = sum[T - 1] + matrix_size;
            MPI_Send(&send_val, 1, MPI_INT, rank + step, rank + step + thread_num, MPI_COMM_WORLD);
        }
    }
    printf("thread %d of rank %d starts at %d\n", thread_num, rank, sum[thread_num] * (int)sizeof(double));
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

int checkMatrix(MPI_File file, unsigned int *seed, unsigned long compressed_len, int offset)
{
    MPI_Status status;

    unsigned char *compressed_buffer = malloc(compressed_len);
    double *decompressed_values = malloc(N * N * N * sizeof(double));
    uLong decompressed_len = compressBound(N * N * N * sizeof(double));


    int error_found = 0;
    MPI_File_read_at(file, offset, compressed_buffer, compressed_len, MPI_BYTE, &status);

    // Decompress the data
    if (uncompress((unsigned char *)decompressed_values, &decompressed_len, compressed_buffer, compressed_len) != Z_OK)
    {
        fprintf(stderr, "Decompression failed\n");
        return 1;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                if (error_found)
                    continue;

                double val = (double)rand_r(seed) / RAND_MAX;

                int index = INDEX(i, j, k);

                if (fabs(decompressed_values[index] - val) > 0.0001)
                {
                    printf("ERROR DETECTED: values[%d] = %.5f, expected %.5f [ thread %d ]\n", index, decompressed_values[index], val, omp_get_thread_num());
#pragma omp atomic write
                    error_found = 1;
                }
                // testing
                // printf("index: %d", index);
                // decompressed_values[index] =   100;
                // printf("matrix[%d] = %f\n",index, decompressed_values[index]);
            }
        }
    }

    free(compressed_buffer);
    free(decompressed_values);
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
        int thread_num = omp_get_thread_num();
        int unique_num = thread_num + rank * T;
        unsigned int seed = unique_num;

        double *matrix = (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

        uLong compressed_len = compressBound(N * N * N * sizeof(double));
        unsigned char *compressed = malloc(compressed_len);

        // size_t data_len = sizeof(matrix); // Total size of the double array in bytes

        // #pragma omp critical
        if (compress(compressed, &compressed_len, (const unsigned char *)matrix, N * N * N * sizeof(double)) != Z_OK)
        {
            fprintf(stderr, "Compression failed\n");
        }
// printf("Original size: %ld, Compressed size: %ld\n", N*N*N*sizeof(double), compressed_len);

// simutaneously start all threads
#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        // Calculate the correct offsets based on the compressed data size
        MPI_Exscan_omp_io(size, rank, compressed_len, sum);

        int start = sum[thread_num];
        int end = start + compressed_len;

        start *= sizeof(unsigned char); // Use sizeof(unsigned char) for bytes
        end *= sizeof(unsigned char);   // Use sizeof(unsigned char) for bytes
        end--;

        // printf("thread %d begins writing at %d and ends at %d.\n", unique_num, start, end);

#pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_write_at(file, start, compressed, compressed_len, MPI_BYTE, &status);
        // printf("thread %d finished.\n", unique_num);

        if (checkMatrix(file, &seed, compressed_len, start))
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
