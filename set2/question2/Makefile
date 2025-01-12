NVCC = nvcc
NVCCFLAGS = -O3 -lm -lcublas

TARGET = cuda
SRC = cuda.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

.PHONY: clean
clean:
	rm -f $(TARGET)