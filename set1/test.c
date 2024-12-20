#include <stdio.h>
#include <string.h>
#include <zlib.h>

#define CHUNK_SIZE 256

int main() {
    const char *input = "This is the data to be compressed using zlib.";
    size_t input_len = strlen(input) + 1; // Include the null terminator

    // Allocate memory for compressed data
    unsigned char compressed[CHUNK_SIZE];
    uLong compressed_len = CHUNK_SIZE;

    // Compress the data
    if (compress(compressed, &compressed_len, (const unsigned char *)input, input_len) != Z_OK) {
        fprintf(stderr, "Compression failed\n");
        return 1;
    }

    printf("Original size: %lu, Compressed size: %lu\n", input_len, compressed_len);

    // Allocate memory for decompressed data
    unsigned char decompressed[CHUNK_SIZE];
    uLong decompressed_len = CHUNK_SIZE;

    // Decompress the data
    if (uncompress(decompressed, &decompressed_len, compressed, compressed_len) != Z_OK) {
        fprintf(stderr, "Decompression failed\n");
        return 1;
    }

    printf("Decompressed size: %lu\n", decompressed_len);
    printf("Decompressed data: %s\n", decompressed);

    return 0;
}
