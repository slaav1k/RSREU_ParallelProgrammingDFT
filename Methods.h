// Methods.h - объявления для DFT/IDFT
#ifndef METHODS_H
#define METHODS_H

#include <math.h>
#include "Complex.h"

#define PI 3.141592653589793

void dft(const cplx in[], cplx out[], int n);
void idft(const cplx in[], cplx out[], int n);

// 2D-преобразования (width x height). Вход и выход — в строковом порядке (row-major), размер width*height
typedef void (*progress_cb_t)(const char* stage, int current, int total);

// set a progress callback (may be NULL to disable)
void set_progress_callback(progress_cb_t cb);

void dft2d(const cplx* in, cplx* out, int width, int height);
void idft2d(const cplx* in, cplx* out, int width, int height);

#endif // METHODS_H
