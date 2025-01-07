CC=gcc
CFLAGS=-Wall -march=native -Wno-unused-value # -fopenmp -O2 -fno-tree-vectorize -fopt-info # --save-temps -fopt-info

all: saxpy_gcc_novec saxpy_gcc_O2 saxpy_gcc_O3 saxpy_restrict saxpy_aligned saxpy_simple.o saxpy_sse saxpy_avx sdot_sse sscal_sse sscal_avx dscal_avx isamax_sse

saxpy_gcc_novec: saxpy.c
	$(CC) $(CFLAGS) -O2 saxpy.c -o saxpy_gcc_novec

saxpy_gcc_O2: saxpy.c
	$(CC) $(CFLAGS) -O2 -ftree-vectorize saxpy.c -o saxpy_gcc_O2

saxpy_gcc_O3: saxpy.c
	$(CC) $(CFLAGS) -O3 saxpy.c -o saxpy_gcc_O3

saxpy_restrict: saxpy_restrict.c
	$(CC) $(CFLAGS) -O2 -ftree-vectorize saxpy_restrict.c -o saxpy_restrict

saxpy_aligned: saxpy_aligned.c
	$(CC) $(CFLAGS) -O3 -ftree-vectorize saxpy_aligned.c -o saxpy_aligned

saxpy_simple.o: saxpy_simple.c
	$(CC) $(CFLAGS) -O3 -save-temps -ftree-vectorize -c saxpy_simple.c -o saxpy_simple.o

saxpy_sse: saxpy_sse.c
	$(CC) $(CFLAGS) -O3 -msse4 saxpy_sse.c -o saxpy_sse

saxpy_avx: saxpy_avx.c
	$(CC) $(CFLAGS) -O3 -mavx saxpy_avx.c -o saxpy_avx

sdot_sse: sdot_sse.c
	$(CC) $(CFLAGS) -O3 -msse4 sdot_sse.c -o sdot_sse

sscal_sse: sscal_sse.c
	$(CC) $(CFLAGS) -O3 -msse4 sscal_sse.c -o sscal_sse

sscal_avx: sscal_avx.c
	$(CC) $(CFLAGS) -O3 -mavx sscal_avx.c -o sscal_avx

dscal_avx: dscal_avx.c
	$(CC) $(CFLAGS) -O3 -mavx dscal_avx.c -o dscal_avx

isamax_sse: isamax_sse.c
	$(CC) $(CFLAGS) -O3 -msse4 isamax_sse.c -o isamax_sse

clean:
	rm -f saxpy_gcc_novec saxpy_gcc_O2 saxpy_gcc_O3 saxpy_restrict saxpy_aligned saxpy_simple.[ois] saxpy_sse saxpy_avx sdot_sse sscal_sse sscal_avx dscal_avx isamax_sse
