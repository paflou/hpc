CC = gcc
CFLAGS = -fopenmp -O3
OMPFLAGS = -foffload="-lm" -foffload=nvptx-none
TARGET = openmpGPU
SRC = openmpGPU.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)