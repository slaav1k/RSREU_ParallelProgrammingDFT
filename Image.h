// Image.h - простая загрузка/сохранение BMP и конвертация в оттенки серого
#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

// Load 24-bit BMP, convert to grayscale. Returns 0 on success.
int load_bmp_grayscale(const char* filename, unsigned char** out_pixels, int* width, int* height);

// Save grayscale pixels as 24-bit BMP. Returns 0 on success.
int save_bmp_grayscale(const char* filename, const unsigned char* pixels, int width, int height);

// Print basic BMP header information to stdout. Returns 0 on success.
int print_bmp_info(const char* filename);

#endif // IMAGE_H
