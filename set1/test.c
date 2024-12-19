#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_File file;
    MPI_Status status;
    unsigned short data[10];
    int i;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize data array with values
    for (i = 0; i < 10; i++) {
        data[i] = rank * 10 + i;
    }

    // Open the binary file for writing (create if doesn't exist)
    MPI_File_open(MPI_COMM_WORLD, "output.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Each process writes to a different part of the file
    MPI_File_write_at(file, rank * 10 * sizeof(int), data, 10, MPI_UNSIGNED_SHORT, &status);

    // Close the file
    MPI_File_close(&file);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
