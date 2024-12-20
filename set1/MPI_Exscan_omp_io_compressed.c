#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>
#include <zlib.h>

#define T 4
#define N 16

#define BUFFER 5000000

void for_loop(int step, int *prev, int *sum, int rank) {
    #pragma omp for ordered schedule(static, 1)
    for(int i=0; i < T; i++) {
        #pragma omp ordered
        {
        *sum += step - 1  + *prev;
        #pragma omp critical
        *prev = *sum;
        //printf("thread %d of %d computes %d\n", omp_get_thread_num(),rank, *sum);
        }
    }
}

int MPI_Exscan_pt2pt(int size, int rank, int step, int *prev) {
    int sum = 0;
        int next = rank + 1;
        int thread_num = omp_get_thread_num();

        if(rank==0){
            for_loop(step, prev, &sum, rank);
            if(thread_num == T - 1)
                MPI_Send(&sum, 1, MPI_INT, next, rank, MPI_COMM_WORLD);

        } else if(rank==size-1) {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for_loop(step, prev, &sum, rank);

        } else {
            if(thread_num == 0)
                MPI_Recv(prev, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for_loop(step, prev, &sum, rank);

            if(thread_num == T - 1)
                MPI_Send(&sum, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
        }
        return sum;
}

void initializeMatrix(double *matrix, unsigned int seed) {
    for (int i = 0; i < N*N*N; i++) {
                matrix[i] = (double)rand_r(&seed) / RAND_MAX;

                //testing
                //matrix[i] =   0x0001020304050607;
    }
}

int checkMatrix(MPI_File file, unsigned int *seed, unsigned long compressed_len, int offset) {
    MPI_Request request;
    MPI_Status status;
    // Allocate memory for decompressed data
    //
    unsigned char *compressed_buffer = malloc(compressed_len);
    double *decompressed_values = malloc(N*N*N * sizeof(double));
    uLong decompressed_len = N*N*N*sizeof(double);

    MPI_File_iread_at(file, offset, compressed_buffer, compressed_len, MPI_BYTE, &request);
    MPI_Wait(&request, &status);

    // Decompress the data
    if (uncompress((unsigned char *)decompressed_values, &decompressed_len, compressed_buffer, compressed_len) != Z_OK) {
        fprintf(stderr, "Decompression failed\n");
        return 1;
    }

    for (int i = 0; i < N*N*N; i++) {
        //printf("values[%d] = %.5f [ thread %d ]\n", i, decompressed_values[i], omp_get_thread_num());
        double val = (double)rand_r(seed) / RAND_MAX;


        //testing purposes
        //double val = 0x0001020304050607;
        if(decompressed_values[i] != val) {
            printf("values[%d] = %f, expected %f [ thread %d ]\n", i, decompressed_values[i], val, omp_get_thread_num());
            //return 1;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_File file;
    MPI_Request request;
    const char *filename = "output.bin";
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Warning: The requested threading level is not available.\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int prev = 0;
    int local_flag = 0;
    int global_flag;

    //deletes any previous file
    if (rank == 0) {
        MPI_File_delete("output.bin", MPI_INFO_NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes sync before proceeding
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);

    #pragma omp parallel num_threads(T) shared(prev, local_flag)
    {
        int global_num = omp_get_thread_num() + rank * T;
        unsigned int seed = global_num;

        double *matrix= (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

        unsigned char compressed[BUFFER];
        uLong compressed_len = BUFFER;

        //size_t data_len = sizeof(matrix); // Total size of the double array in bytes

        #pragma omp critical
        if (compress(compressed, &compressed_len, (const unsigned char *)matrix, N*N*N*sizeof(double)) != Z_OK) {
            fprintf(stderr, "Compression failed\n");
        }
        //printf("Original size: %ld, Compressed size: %ld\n", N*N*N*sizeof(double), compressed_len);

        #pragma omp barrier
        // Calculate the correct offsets based on the compressed data size
        int end = MPI_Exscan_pt2pt(size, rank, compressed_len, &prev);

        int start = end - compressed_len + 1;

        start += global_num;
        start *= sizeof(unsigned char); // Use sizeof(unsigned char) for bytes
        end += global_num + 1;
        end *= sizeof(unsigned char); // Use sizeof(unsigned char) for bytes

        //if(global_num== 8)
        //printf("thread %d begins writing at %d and ends at %d.\n", global_num, start, end);

        #pragma omp barrier
        MPI_File_iwrite_at(file, start, compressed, compressed_len, MPI_BYTE, &request);
        //MPI_File_write_at(file, start, compressed, compressed_len, MPI_BYTE, MPI_STATUS_IGNORE);
        #pragma omp barrier
        //printf("thread %d finished.\n", global_num);

        if(checkMatrix(file, &seed, compressed_len, start)) {
            #pragma omp critical
            local_flag = 1;
        }

        free(matrix);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank ==0) {
        global_flag ? printf("The binary file is wrong") : printf("The binary file is correct");
    }
    MPI_Finalize();
    return 0;
}
