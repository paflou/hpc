CC=gcc
CFLAGS=-fopenmp -O3 -mavx -msse4 -mfma -lm -Wno-unused-result #  -march=native -O2 -fno-tree-vectorize -fopt-info # --save-temps -fopt-info
LDFLAGS=-lm
all: avx_to_double ddot_avx sdot_sse sdot_avx dot_omp_simd


clean:
	rm -f avx_to_double ddot_avx sdot_sse sdot_avx dot_omp_simd
