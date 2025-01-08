CC=gcc
CFLAGS=-O -ftree-vectorize -march=native -ffast-math -fopt-info-optimized=vec.txt
LIBS=-lm

all: clean bench

bench: bench.c weno.h
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS)

clean:
	rm -f bench vec.txt perf*
