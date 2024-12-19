#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <zlib.h>

void compressString(const char *str, unsigned char **compressed, size_t *compressedSize) {
    uLong srcLen = strlen(str) + 1; // +1 to include the null terminator
    uLong destLen = compressBound(srcLen); // Get the maximum size of the compressed data

    *compressed = (unsigned char *)malloc(destLen);

    if (compress(*compressed, &destLen, (const Bytef *)str, srcLen) != Z_OK) {
        printf("Compression failed!\n");
        free(*compressed);
        *compressed = NULL;
        *compressedSize = 0;
        return;
    }

    *compressedSize = destLen;
}

void decompressString(const unsigned char *compressed, size_t compressedSize, char **decompressed) {
    uLong destLen = compressedSize * 2; // Initial guess for the decompressed size
    *decompressed = (char *)malloc(destLen);

    while (uncompress((Bytef *)*decompressed, &destLen, compressed, compressedSize) == Z_BUF_ERROR) {
        // If the buffer was too small, increase its size and try again
        destLen *= 2;
        *decompressed = (char *)realloc(*decompressed, destLen);
    }
}

int main() {
    const char *originalString = "Hello, this is a test string for zlib compression!";
    unsigned char *compressedString = NULL;
    size_t compressedSize = 0;

    // Compress the string
    compressString(originalString, &compressedString, &compressedSize);

    if (compressedString != NULL) {
        printf("Original size: %lu\n", strlen(originalString) + 1);
        printf("Compressed size: %lu\n", compressedSize);

        // Decompress the string
        char *decompressedString = NULL;
        decompressString(compressedString, compressedSize, &decompressedString);

        if (decompressedString != NULL) {
            printf("Decompressed string: %s\n", decompressedString);
            free(decompressedString);
        }

        free(compressedString);
    }

    return 0;
}
