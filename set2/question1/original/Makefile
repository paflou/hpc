CC=gcc
CFLAGS=-fopenmp -O3 -fopt-info-vec-optimized
LIBS=-lm

all: bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench
