#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>


int MPI_Exscan_pt2pt(int size, int rank, int value)
{
    int result = 0;
    for (int step = 1; step < size; step++)
    {
        int partial = 0;
        if (rank - step >= 0)
        {
            MPI_Recv(&partial, 1, MPI_INT, rank - step, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += partial;
            //printf("rank %d computes %d iteration %d\n", rank, result, step);
        }
        if (rank + step < size)
            MPI_Bsend(&value, 1, MPI_INT, rank + step, rank + step, MPI_COMM_WORLD);
    }
    //printf("rank %d computes %d\n", rank, result);
    return result;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int result = MPI_Exscan_pt2pt(size, rank, rank);

    usleep(rank * 1000);        // Sleep for rank milliseconds to print in order
    printf("rank %d result %d\n", rank, result);
    MPI_Finalize();
}
