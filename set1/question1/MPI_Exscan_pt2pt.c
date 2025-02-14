#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

int MPI_Exscan_pt2pt(int size, int rank, int value, MPI_Op op)
{
    int result = op == MPI_SUM ? 0 : 1;
    int recv_val = 0;

    for (int step = 1; step < size; step *= 2)
    {
        int send_partner = rank + step;
        int recv_partner = rank - step;
        int send_val = -1;
        if (send_partner < size)
        {
            if(op == MPI_SUM)
                send_val = result + value;
            else if(op == MPI_PROD)
                send_val = result * value;
            else if(op == MPI_MAX)
                send_val = result > value ? result : value;
            else if(op == MPI_MIN)
                send_val = result < value ? result : value;
            else
                {printf("Invalid MPI_Op\n"); return -1;}
            MPI_Send(&send_val, 1, MPI_INT, send_partner, 0, MPI_COMM_WORLD);
        }
        if (recv_partner >= 0)
        {
            MPI_Recv(&recv_val, 1, MPI_INT, recv_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Add received value to all threads' sums
            if(op == MPI_SUM)
                result += recv_val;
            else if(op == MPI_PROD)
                result *= recv_val;
            else if(op == MPI_MAX)
                result = result > recv_val ? result : recv_val;
            else if(op == MPI_MIN)
                result = result < recv_val ? result : recv_val;
            else
                {printf("Invalid MPI_Op\n"); return -1;}
        }
        // printf("rank %d computes %d\n", rank, result);
    }
    return rank == 0 ? -1 : result;

}

    int main(int argc, char *argv[])
    {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        MPI_Op op = MPI_SUM;
        int result = MPI_Exscan_pt2pt(size, rank, rank, op);

        usleep(rank * 1000); // Sleep for rank milliseconds to print in order
        printf("rank %d result %d\n", rank, result);
        MPI_Finalize();
    }
