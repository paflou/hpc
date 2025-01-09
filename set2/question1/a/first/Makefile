CC=gcc
CFLAGS=-O3 -mavx2 -mavx -ftree-vectorize -march=native -ffast-math -fopt-info-optimized
LIBS=-lm

all: clean bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench perf*
