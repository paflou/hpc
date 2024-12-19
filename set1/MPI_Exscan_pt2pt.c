#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void MPI_Exscan_pt2pt(int size, int rank) {
    int sum = 0;
        int next = rank + 1;

        //printf(" rank %d\n", rank);
        if(rank==0){
            //printf("rank %d sending to %d\n",rank, rank);
            MPI_Send(&rank, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
        }else if(rank==size-1) {
            int prev = rank - 1;
            //printf("rank %d waiting on %d\n",rank, prev);
            MPI_Recv(&sum, 1, MPI_INT, prev, prev, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += prev;
        }else {
            int prev = rank - 1;
            //printf("rank %d waiting on %d\n",rank, prev);
            MPI_Recv(&sum, 1, MPI_INT, prev, prev, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += prev;
            //printf("rank %d sending to %d\n",rank, next);
            MPI_Send(&sum, 1, MPI_INT, next, rank, MPI_COMM_WORLD);
        }
        printf("rank %d computes %d\n",rank, sum);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Exscan_pt2pt(size, rank);
    MPI_Finalize();
}
