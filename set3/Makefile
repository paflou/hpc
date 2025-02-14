CC = gcc
CFLAGS = -fopenmp -O3 -lm -fno-lto
TARGET = openmpGPU
SRC = openmpGPU.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)