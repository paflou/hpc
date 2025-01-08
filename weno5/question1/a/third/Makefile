CC=gcc
CFLAGS=-O -fopenmp -ffast-math -march=native -fopt-info-vec-optimized=vec.txt
LIBS=-lm

all: clean bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench vec.txt perf*
