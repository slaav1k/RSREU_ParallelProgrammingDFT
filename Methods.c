#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Methods.h"

static progress_cb_t g_progress_cb = NULL;

void set_progress_callback(progress_cb_t cb) {
    g_progress_cb = cb;
}

// Наивное прямое DFT
void dft(const cplx in[], cplx out[], int n) {
    for (int k = 0; k < n; k++) {
        cplx sum;
        sum.re = 0.0; sum.im = 0.0;
        for (int m = 0; m < n; m++) {
            double angle = -2 * PI * k * m / (double)n;
            cplx w = cplx_from_polar(1.0, angle);
            cplx prod = cplx_mul(in[m], w);
            sum = cplx_add(sum, prod);
        }
        out[k] = sum;
    }
}

// Обратное DFT (почти то же, только знак + и деление на n)
void idft(const cplx in[], cplx out[], int n) {
    for (int k = 0; k < n; k++) {
        cplx sum;
        sum.re = 0.0; sum.im = 0.0;
        for (int m = 0; m < n; m++) {
            double angle = +2 * PI * k * m / (double)n;
            cplx w = cplx_from_polar(1.0, angle);
            cplx prod = cplx_mul(in[m], w);
            sum = cplx_add(sum, prod);
        }
        out[k] = cplx_scale(sum, 1.0 / (double)n);
    }
}

// 2D DFT: выполнить 1D DFT по строкам, затем по столбцам
void dft2d(const cplx* in, cplx* out, int width, int height) {
    cplx* temp = (cplx*)malloc(sizeof(cplx) * width * height);
    if (!temp) return;
    // rows
    for (int y = 0; y < height; ++y) {
        dft(&in[y*width], &temp[y*width], width);
        if (g_progress_cb) g_progress_cb("Rows (forward)", y+1, height);
    }
    // columns
    cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
    cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
    if (!col_in || !col_out) { free(temp); free(col_in); free(col_out); return; }
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) col_in[y] = temp[y*width + x];
        dft(col_in, col_out, height);
        for (int y = 0; y < height; ++y) out[y*width + x] = col_out[y];
        if (g_progress_cb) g_progress_cb("Cols (forward)", x+1, width);
    }
    free(col_in); free(col_out); free(temp);
}

void idft2d(const cplx* in, cplx* out, int width, int height) {
    cplx* temp = (cplx*)malloc(sizeof(cplx) * width * height);
    if (!temp) return;
    // columns (inverse)
    cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
    cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
    if (!col_in || !col_out) { free(temp); free(col_in); free(col_out); return; }
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) col_in[y] = in[y*width + x];
        idft(col_in, col_out, height);
        for (int y = 0; y < height; ++y) temp[y*width + x] = col_out[y];
        if (g_progress_cb) g_progress_cb("Cols (inverse)", x+1, width);
    }
    // rows (inverse)
    for (int y = 0; y < height; ++y) {
        idft(&temp[y*width], &out[y*width], width);
        if (g_progress_cb) g_progress_cb("Rows (inverse)", y+1, height);
    }
    free(col_in); free(col_out); free(temp);
}
