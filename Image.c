#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Image.h"

#pragma pack(push,1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

// Загрузить 24/32-bit BMP и конвертировать в оттенки серого
int load_bmp_grayscale(const char* filename, unsigned char** out_pixels, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    BITMAPFILEHEADER bf;
    BITMAPINFOHEADER bi;
    if (fread(&bf, sizeof(bf), 1, f) != 1) { fclose(f); return -2; }
    if (bf.bfType != 0x4D42) { fclose(f); return -3; }
    if (fread(&bi, sizeof(bi), 1, f) != 1) { fclose(f); return -4; }
    int bitCount = bi.biBitCount;
    if (bitCount != 24 && bitCount != 32) { fclose(f); return -5; }
    int w = bi.biWidth;
    int h = abs(bi.biHeight);

    // обработка возможных масок BITFIELDS для 32-bit изображений
    uint32_t redMask = 0, greenMask = 0, blueMask = 0;
    int hasMasks = 0;
    if (bitCount == 32 && bi.biCompression == 3) {
        // read three DWORD masks (file pointer is right after BITMAPINFOHEADER)
        if (fread(&redMask, 4, 1, f) == 1 && fread(&greenMask, 4, 1, f) == 1 && fread(&blueMask, 4, 1, f) == 1) {
            hasMasks = 1;
        } else {
            // failed to read masks, but continue treating as simple BGRA
            redMask = greenMask = blueMask = 0;
            hasMasks = 0;
        }
    }

    int bytesPerPixel = bitCount / 8;
    int row_padded = (bytesPerPixel * w + 3) & (~3);
    unsigned char* data = (unsigned char*)malloc(w*h);
    if (!data) { fclose(f); return -6; }
    unsigned char* row = (unsigned char*)malloc(row_padded);
    fseek(f, bf.bfOffBits, SEEK_SET);
    for (int y = 0; y < h; ++y) {
        if (fread(row, 1, row_padded, f) != (size_t)row_padded) { free(row); free(data); fclose(f); return -7; }
        for (int x = 0; x < w; ++x) {
            unsigned char r = 0, g = 0, b = 0;
            if (bitCount == 24) {
                b = row[x*3 + 0];
                g = row[x*3 + 1];
                r = row[x*3 + 2];
            } else {
                // 32-bit
                uint32_t px = (uint32_t)row[x*4 + 0] | ((uint32_t)row[x*4 + 1] << 8) | ((uint32_t)row[x*4 + 2] << 16) | ((uint32_t)row[x*4 + 3] << 24);
                if (hasMasks && redMask && greenMask && blueMask) {
                    // extract component by mask
                    uint32_t v;
                    // red
                    v = px & redMask;
                    int rs = 0; uint32_t rm = redMask; while (rm && ((rm & 1) == 0)) { rm >>= 1; rs++; }
                    int rbits = 0; rm = redMask; while (rm) { rbits += (rm & 1); rm >>= 1; }
                    uint32_t rval = (rbits > 0) ? (v >> rs) : 0;
                    if (rbits > 0) r = (unsigned char)((rval * 255) / ((1u << rbits) - 1));
                    // green
                    v = px & greenMask;
                    int gs = 0; uint32_t gm = greenMask; while (gm && ((gm & 1) == 0)) { gm >>= 1; gs++; }
                    int gbits = 0; gm = greenMask; while (gm) { gbits += (gm & 1); gm >>= 1; }
                    uint32_t gval = (gbits > 0) ? (v >> gs) : 0;
                    if (gbits > 0) g = (unsigned char)((gval * 255) / ((1u << gbits) - 1));
                    // blue
                    v = px & blueMask;
                    int bs = 0; uint32_t bm = blueMask; while (bm && ((bm & 1) == 0)) { bm >>= 1; bs++; }
                    int bbits = 0; bm = blueMask; while (bm) { bbits += (bm & 1); bm >>= 1; }
                    uint32_t bval = (bbits > 0) ? (v >> bs) : 0;
                    if (bbits > 0) b = (unsigned char)((bval * 255) / ((1u << bbits) - 1));
                } else {
                    // assume BGRA byte order
                    b = (unsigned char)row[x*4 + 0];
                    g = (unsigned char)row[x*4 + 1];
                    r = (unsigned char)row[x*4 + 2];
                }
            }
            // преобразовать в оттенки серого
            unsigned char gray = (unsigned char)((0.299*r + 0.587*g + 0.114*b));
            int yy = (bi.biHeight > 0) ? (h-1-y) : y; // BMP may be bottom-up
            data[yy*w + x] = gray;
        }
    }
    free(row);
    fclose(f);
    *out_pixels = data;
    *width = w;
    *height = h;
    return 0;
}

// Напечатать основную информацию о BMP
int print_bmp_info(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot open %s\n", filename); return -1; }
    BITMAPFILEHEADER bf;
    BITMAPINFOHEADER bi;
    if (fread(&bf, sizeof(bf), 1, f) != 1) { fclose(f); printf("Read error\n"); return -2; }
    if (bf.bfType != 0x4D42) { fclose(f); printf("Not a BMP (bfType mismatch)\n"); return -3; }
    if (fread(&bi, sizeof(bi), 1, f) != 1) { fclose(f); printf("Read error\n"); return -4; }
    printf("BMP Info: %s\n", filename);
    printf("  File size: %u\n", bf.bfSize);
    printf("  Offset to pixel data: %u\n", bf.bfOffBits);
    printf("  Header size: %u\n", bi.biSize);
    printf("  Width: %d\n", bi.biWidth);
    printf("  Height: %d\n", bi.biHeight);
    printf("  Planes: %u\n", bi.biPlanes);
    printf("  BitCount: %u\n", bi.biBitCount);
    printf("  Compression: %u\n", bi.biCompression);
    printf("  ImageSize: %u\n", bi.biSizeImage);
    printf("  X Pels per meter: %d\n", bi.biXPelsPerMeter);
    printf("  Y Pels per meter: %d\n", bi.biYPelsPerMeter);
    printf("  Colors used: %u\n", bi.biClrUsed);
    fclose(f);
    return 0;
}

int save_bmp_grayscale(const char* filename, const unsigned char* pixels, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    int row_padded = (width*3 + 3) & (~3);
    int image_size = row_padded * height;
    BITMAPFILEHEADER bf;
    BITMAPINFOHEADER bi;
    bf.bfType = 0x4D42;
    bf.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bf.bfSize = bf.bfOffBits + image_size;
    bf.bfReserved1 = bf.bfReserved2 = 0;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = height; // bottom-up
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = 0;
    bi.biSizeImage = image_size;
    bi.biXPelsPerMeter = bi.biYPelsPerMeter = 0;
    bi.biClrUsed = bi.biClrImportant = 0;
    fwrite(&bf, sizeof(bf), 1, f);
    fwrite(&bi, sizeof(bi), 1, f);
    unsigned char* row = (unsigned char*)malloc(row_padded);
    if (!row) { fclose(f); return -2; }
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_padded);
        int yy = height-1-y;
        for (int x = 0; x < width; ++x) {
            unsigned char gray = pixels[yy*width + x];
            row[x*3 + 0] = gray;
            row[x*3 + 1] = gray;
            row[x*3 + 2] = gray;
        }
        fwrite(row, 1, row_padded, f);
    }
    free(row);
    fclose(f);
    return 0;
}
