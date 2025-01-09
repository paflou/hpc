CC=gcc
CFLAGS=-O3 -fopenmp -ffast-math -march=native -fopt-info-vec-optimized
LIBS=-lm

all: clean bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench perf*
