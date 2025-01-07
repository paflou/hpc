#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>
#include <zlib.h>
#include <math.h>

#define BUFFER 5000000

int T, N;

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

void initializeMatrix(double *matrix, unsigned int seed) {
    for (int i = 0; i < N*N*N; i++) {
                matrix[i] = (double)rand_r(&seed) / RAND_MAX;

                //testing
                //matrix[i] =   100.0;
    }
}

int checkMatrix(MPI_File file, unsigned int *seed, unsigned long compressed_len, int offset) {
    MPI_Status status;

    unsigned char *compressed_buffer = malloc(compressed_len);
    double *decompressed_values = malloc(N*N*N * sizeof(double));
    uLong decompressed_len = N*N*N*sizeof(double);

    MPI_File_read_at_all(file, offset, compressed_buffer, compressed_len, MPI_BYTE, &status);

    // Decompress the data
    if (uncompress((unsigned char *)decompressed_values, &decompressed_len, compressed_buffer, compressed_len) != Z_OK) {
        fprintf(stderr, "Decompression failed\n");
        return 1;
    }

    for (int i = 0; i < N*N*N; i++) {
        double val = (double)rand_r(seed) / RAND_MAX;

        //testing purposes
        //double val = 100.0;
        //printf("values[%d] = %.5f, expected %.5f [ thread %d ]\n", i, decompressed_values[i], val, omp_get_thread_num());
        if(fabs(decompressed_values[i] - val) > 0.0001) {
            printf("ERROR DETECTED: values[%d] = %.5f, expected %.5f [ thread %d ]\n", i, decompressed_values[i], val, omp_get_thread_num());
            free(compressed_buffer);
            free(decompressed_values);
            return 1;
        }
    }
    free(compressed_buffer);
    free(decompressed_values);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_File file;
    MPI_Status status;
    const char *filename = "output.bin";
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        printf("Usage: %s <number of threads> <N (matrix = N*N*N)>\n", argv[0]);
        return 1;
    }

    T = atoi(argv[1]);
    N = atoi(argv[2]);

    int prev = 0;
    int local_flag = 0;
    int global_flag;

    //deletes any previous file
    if (rank == 0) {
        MPI_File_delete("output.bin", MPI_INFO_NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);

    #pragma omp parallel num_threads(T) shared(prev, local_flag)
    {
        int unique_num = omp_get_thread_num() + rank * T;
        unsigned int seed = unique_num;

        double *matrix= (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

        unsigned char compressed[BUFFER];
        uLong compressed_len = BUFFER;

        //size_t data_len = sizeof(matrix); // Total size of the double array in bytes

        //#pragma omp critical
        if (compress(compressed, &compressed_len, (const unsigned char *)matrix, N*N*N*sizeof(double)) != Z_OK) {
            fprintf(stderr, "Compression failed\n");
        }
        //printf("Original size: %ld, Compressed size: %ld\n", N*N*N*sizeof(double), compressed_len);

        //simutaneously start all threads
        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);

        // Calculate the correct offsets based on the compressed data size
        int start = MPI_Exscan_omp(size, rank, compressed_len, &prev);

        int end = start + compressed_len;

        start *= sizeof(unsigned char); // Use sizeof(unsigned char) for bytes
        end *= sizeof(unsigned char); // Use sizeof(unsigned char) for bytes
        end--;
        
        //printf("thread %d begins writing at %d and ends at %d.\n", unique_num, start, end);

        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_write_at(file, start, compressed, compressed_len, MPI_BYTE, &status);
        //printf("thread %d finished.\n", unique_num);

        if(checkMatrix(file, &seed, compressed_len, start)) {
            #pragma omp critical
            local_flag = 1;
        }

        free(matrix);
    }

    MPI_File_close(&file);
    MPI_Reduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank ==0) {
        global_flag ? printf("\nThe binary file is wrong\n") : printf("\nThe binary file is correct\n");
    }
    MPI_Finalize();
    return 0;
}
