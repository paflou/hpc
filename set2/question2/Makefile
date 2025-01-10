NVCC = nvcc
NVCCFLAGS = -O2

TARGET = cuda
SRC = cuda.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

.PHONY: clean
clean:
	rm -f $(TARGET)