#include <stdlib.h>
#include <math.h>
#include "Transform.h"

// Преобразовать пиксели (0..255) в комплексный массив (реальная часть в [0,1])
void pixels_to_cplx(const unsigned char* pixels, int w, int h, cplx* out) {
    int n = w * h;
    for (int i = 0; i < n; ++i) {
        out[i].re = pixels[i] / 255.0;
        out[i].im = 0.0;
    }
}

// Создать изображение спектра (градации серого) из комплексных данных: log(1+|X|) нормализованное
void cplx_to_spectrum_image(const cplx* data, int w, int h, unsigned char* out) {
    int n = w * h;
    double maxm = 1e-12;
    for (int i = 0; i < n; ++i) {
        double mag = sqrt(data[i].re*data[i].re + data[i].im*data[i].im);
        if (mag > maxm) maxm = mag;
    }
    for (int i = 0; i < n; ++i) {
        double mag = sqrt(data[i].re*data[i].re + data[i].im*data[i].im);
        double v = log(1 + mag) / log(1 + maxm);
        int iv = (int)(v * 255.0);
        if (iv < 0) iv = 0; if (iv > 255) iv = 255;
        out[i] = (unsigned char)iv;
    }
}

// Преобразовать комплексный массив в пиксели (взять реальную часть и нормализовать в 0..255)
void cplx_to_pixels(const cplx* data, int w, int h, unsigned char* out) {
    int n = w * h;
    for (int i = 0; i < n; ++i) {
        double v = data[i].re;
        if (v < 0) v = 0; if (v > 1) v = 1;
        out[i] = (unsigned char)(v * 255.0);
    }
}
