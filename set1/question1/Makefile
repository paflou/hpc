CC=gcc
MPICC=mpicc
CFLAGS=-Wall -Ofast -fopenmp
LDLIBS=-lm -lz

all: MPI_Exscan_pt2pt MPI_Exscan_omp MPI_Exscan_omp_io MPI_Exscan_omp_io_compressed

MPI_Exscan_pt2pt: MPI_Exscan_pt2pt.c Makefile
	$(MPICC) $(CFLAGS) -o MPI_Exscan_pt2pt MPI_Exscan_pt2pt.c $(LDLIBS)

MPI_Exscan_omp: MPI_Exscan_omp.c Makefile
	$(MPICC) $(CFLAGS) -o MPI_Exscan_omp MPI_Exscan_omp.c $(LDLIBS)

MPI_Exscan_omp_io: MPI_Exscan_omp_io.c Makefile
	$(MPICC) $(CFLAGS) -o MPI_Exscan_omp_io MPI_Exscan_omp_io.c $(LDLIBS)


MPI_Exscan_omp_io_compressed: MPI_Exscan_omp_io_compressed.c Makefile
	$(MPICC) $(CFLAGS) -o MPI_Exscan_omp_io_compressed MPI_Exscan_omp_io_compressed.c $(LDLIBS)


clean:
	rm -f MPI_Exscan_pt2pt MPI_Exscan_omp MPI_Exscan_omp_io MPI_Exscan_omp_io_compressed output.bin
