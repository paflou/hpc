CC=gcc
CFLAGS=-O3 -fopenmp -fopenmp-simd -march=native -ftree-vectorize -fopt-info-vec-missed=vec.txt
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench
