#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Methods.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static progress_cb_t g_progress_cb = NULL;

void set_progress_callback(progress_cb_t cb) {
    g_progress_cb = cb;
}

static int g_use_omp = 0;
void set_use_omp(int use) { g_use_omp = use; }

void set_num_threads(int n) {
#if defined(_OPENMP)
    if (n > 0) omp_set_num_threads(n);
#endif
}

// Naive forward DFT
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

// Inverse DFT (almost the same, plus sign and divide by n)
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

// 2D DFT: perform 1D DFT on rows, then on columns
void dft2d(const cplx* in, cplx* out, int width, int height) {
    cplx* temp = (cplx*)malloc(sizeof(cplx) * width * height);
    if (!temp) return;
    // rows
#if defined(_OPENMP)
    if (g_use_omp) {
        int progress = 0;
        int y;
        #pragma omp parallel for schedule(static) private(y)
        for (y = 0; y < height; ++y) {
            dft(&in[y*width], &temp[y*width], width);
            if (g_progress_cb) {
                #pragma omp atomic
                progress++;
                #pragma omp critical
                {
                    g_progress_cb("Rows (forward)", progress, height);
                }
            }
        }
    } else
#endif
    {
        for (int y = 0; y < height; ++y) {
            dft(&in[y*width], &temp[y*width], width);
            if (g_progress_cb) g_progress_cb("Rows (forward)", y+1, height);
        }
    }
    // columns
    cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
    cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
    if (!col_in || !col_out) { free(temp); free(col_in); free(col_out); return; }
#if defined(_OPENMP)
    if (g_use_omp) {
        int progress = 0;
        int x;
        #pragma omp parallel for schedule(static) private(x)
        for (x = 0; x < width; ++x) {
            cplx* local_in = (cplx*)malloc(sizeof(cplx) * height);
            cplx* local_out = (cplx*)malloc(sizeof(cplx) * height);
            if (!local_in || !local_out) { free(local_in); free(local_out); continue; }
            for (int y = 0; y < height; ++y) local_in[y] = temp[y*width + x];
            dft(local_in, local_out, height);
            for (int y = 0; y < height; ++y) out[y*width + x] = local_out[y];
            free(local_in); free(local_out);
            if (g_progress_cb) {
                #pragma omp atomic
                progress++;
                #pragma omp critical
                {
                    g_progress_cb("Cols (forward)", progress, width);
                }
            }
        }
    } else
#endif
    {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) col_in[y] = temp[y*width + x];
            dft(col_in, col_out, height);
            for (int y = 0; y < height; ++y) out[y*width + x] = col_out[y];
            if (g_progress_cb) g_progress_cb("Cols (forward)", x+1, width);
        }
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
#if defined(_OPENMP)
    if (g_use_omp) {
        int progress = 0;
        int x;
        #pragma omp parallel for schedule(static) private(x)
        for (x = 0; x < width; ++x) {
            cplx* local_in = (cplx*)malloc(sizeof(cplx) * height);
            cplx* local_out = (cplx*)malloc(sizeof(cplx) * height);
            if (!local_in || !local_out) { free(local_in); free(local_out); continue; }
            for (int y = 0; y < height; ++y) local_in[y] = in[y*width + x];
            idft(local_in, local_out, height);
            for (int y = 0; y < height; ++y) temp[y*width + x] = local_out[y];
            free(local_in); free(local_out);
            if (g_progress_cb) {
                #pragma omp atomic
                progress++;
                #pragma omp critical
                {
                    g_progress_cb("Cols (inverse)", progress, width);
                }
            }
        }
    } else
#endif
    {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) col_in[y] = in[y*width + x];
            idft(col_in, col_out, height);
            for (int y = 0; y < height; ++y) temp[y*width + x] = col_out[y];
            if (g_progress_cb) g_progress_cb("Cols (inverse)", x+1, width);
        }
    }
    // rows (inverse)
#if defined(_OPENMP)
    if (g_use_omp) {
        int progress = 0;
        int y;
        #pragma omp parallel for schedule(static) private(y)
        for (y = 0; y < height; ++y) {
            idft(&temp[y*width], &out[y*width], width);
            if (g_progress_cb) {
                #pragma omp atomic
                progress++;
                #pragma omp critical
                {
                    g_progress_cb("Rows (inverse)", progress, height);
                }
            }
        }
    } else
#endif
    {
        for (int y = 0; y < height; ++y) {
            idft(&temp[y*width], &out[y*width], width);
            if (g_progress_cb) g_progress_cb("Rows (inverse)", y+1, height);
        }
    }
    free(col_in); free(col_out); free(temp);
}
