// transform.h - helpers to convert between pixels and complex arrays
#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Complex.h"

// Convert grayscale pixels [0..255] to complex array with real in [0,1]
void pixels_to_cplx(const unsigned char* pixels, int w, int h, cplx* out);

// Create a visualization image (grayscale) from complex spectrum: log(1+|X|) normalized
void cplx_to_spectrum_image(const cplx* data, int w, int h, unsigned char* out);

// Convert complex image (take real part) to grayscale pixels [0..255]
void cplx_to_pixels(const cplx* data, int w, int h, unsigned char* out);

#endif // TRANSFORM_H
