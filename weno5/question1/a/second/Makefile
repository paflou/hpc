CC=gcc
CFLAGS=-O1 -fopenmp-simd -fopenmp -ffast-math -fopt-info-vec-optimized
LIBS=-lm

all: clean bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench perf*
