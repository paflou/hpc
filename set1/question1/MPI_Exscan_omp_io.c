#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <threads.h>
#include <unistd.h>
#include <math.h>

int T, N;

#define INDEX(i, j, k) ((i) * N * N + (j) * N + (k))

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
    #pragma omp for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                int index = INDEX(i, j, k);
                matrix[index] = (double)rand_r(&seed) / RAND_MAX;

                //testing
                //printf("index: %d", index);
                //matrix[index] =   100;
                //printf("matrix[%d] = %f\n",index, matrix[index]);
            }
        }
    }
}


int checkMatrix(MPI_File file, unsigned int *seed, int offset) {
    MPI_Status status;
    double *values = (double *)malloc(N * N * N * sizeof(double));
    int error_found = 0;

    MPI_File_read_at_all(file, offset, values, N*N*N, MPI_DOUBLE, &status);
    #pragma omp for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (error_found) continue;

                double val = (double)rand_r(seed) / RAND_MAX;

                int index = INDEX(i, j, k);

                if(fabs(values[index] - val) > 0.0001) {
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

int main(int argc, char *argv[]) {
    MPI_File file;
    const char *filename = "output.bin";
    MPI_Status status;
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

        //use 1d array for the matrix (easier + better memory locality)
        double *matrix= (double *)malloc(N * N * N * sizeof(double));

        initializeMatrix(matrix, seed);

        int matrixSize = N * N * N;

        //simutaneously start all threads
        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
        int start = MPI_Exscan_omp(size, rank, matrixSize, &prev);

        int end = start + matrixSize;
        start *= sizeof(double);
        end *= sizeof(double);
        end--;

        //printf("thread %d begins writing at %d and ends at %d. 1st val = %f\n", unique_num, start, end, matrix[0]);
        MPI_File_write_at_all(file, start, matrix, matrixSize, MPI_DOUBLE, &status);
        //printf("thread %d finished.\n", unique_num);
        
        #pragma omp barrier
        if(checkMatrix(file, &seed, start)) {
            #pragma omp critical
            local_flag = 1;
        }
        free(matrix);
    }

    MPI_File_close(&file);
    MPI_Reduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        global_flag ? printf("\nThe binary file is wrong\n") : printf("\nThe binary file is correct\n");
    }
    MPI_Finalize();
    return 0;
}
