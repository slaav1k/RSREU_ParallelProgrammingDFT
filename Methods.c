#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Methods.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static progress_cb_t g_progress_cb = NULL;

void set_progress_callback(progress_cb_t cb) {
    g_progress_cb = cb;
}

static int g_use_omp = 0;
void set_use_omp(int use) { g_use_omp = use; }

static int g_use_mpi = 0;
void set_use_mpi(int use) { g_use_mpi = use; }

void set_num_threads(int n) {
#if defined(_OPENMP)
    if (n > 0) omp_set_num_threads(n);
#endif
}

// Naive forward DFT
void dft(const cplx in[], cplx out[], int n) {
    for (int k = 0; k < n; k++) {
        cplx sum = { 0.0, 0.0 };
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
        cplx sum = { 0.0, 0.0 };
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
#if defined(USE_MPI)
    if (g_use_mpi) {
        int rank = 0, size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Распределение строк
        int base = height / size;
        int rem = height % size;

        int* rows = (int*)malloc(sizeof(int) * size);
        int* displs_rows = (int*)malloc(sizeof(int) * size);

        int offset = 0;
        for (int i = 0; i < size; ++i) {
            rows[i] = base + (i < rem ? 1 : 0);
            displs_rows[i] = offset;
            offset += rows[i];
        }

        int local_rows = rows[rank];

        // sendcounts в double (2 double на cplx)
        int* sendcounts_dbl = (int*)malloc(sizeof(int) * size);
        int* senddispls_dbl = (int*)malloc(sizeof(int) * size);

        for (int i = 0; i < size; ++i) {
            sendcounts_dbl[i] = rows[i] * width * 2;
            senddispls_dbl[i] = displs_rows[i] * width * 2;
        }

        // Подготовка данных на root
        double* sendbuf_dbl = NULL;
        if (rank == 0) {
            sendbuf_dbl = (double*)malloc(height * width * 2 * sizeof(double));
            for (int i = 0; i < height * width; i++) {
                sendbuf_dbl[i * 2] = in[i].re;
                sendbuf_dbl[i * 2 + 1] = in[i].im;
            }
        }

        double* recvbuf_dbl = (double*)malloc(local_rows * width * 2 * sizeof(double));

        MPI_Barrier(MPI_COMM_WORLD);

        // Scatterv
        MPI_Scatterv(sendbuf_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE,
            recvbuf_dbl, sendcounts_dbl[rank], MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Конвертация обратно в cplx
        cplx* local_in = (cplx*)malloc(sizeof(cplx) * local_rows * width);
        for (int i = 0; i < local_rows * width; i++) {
            local_in[i].re = recvbuf_dbl[i * 2];
            local_in[i].im = recvbuf_dbl[i * 2 + 1];
        }

        // Локальное DFT по строкам
        cplx* local_temp = (cplx*)malloc(sizeof(cplx) * local_rows * width);
        for (int r = 0; r < local_rows; ++r) {
            dft(&local_in[r * width], &local_temp[r * width], width);
        }

        // Конвертируем в double для передачи
        double* local_temp_dbl = (double*)malloc(sizeof(double) * local_rows * width * 2);
        for (int i = 0; i < local_rows * width; i++) {
            local_temp_dbl[i * 2] = local_temp[i].re;
            local_temp_dbl[i * 2 + 1] = local_temp[i].im;
        }

        // Собираем все результаты
        double* temp_all_dbl = (double*)malloc(sizeof(double) * width * height * 2);
        MPI_Allgatherv(local_temp_dbl, sendcounts_dbl[rank], MPI_DOUBLE,
            temp_all_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE, MPI_COMM_WORLD);

        // Конвертируем обратно в cplx
        cplx* temp_all = (cplx*)malloc(sizeof(cplx) * width * height);
        for (int i = 0; i < width * height; i++) {
            temp_all[i].re = temp_all_dbl[i * 2];
            temp_all[i].im = temp_all_dbl[i * 2 + 1];
        }

        // DFT по столбцам
        cplx* out_full = (cplx*)malloc(sizeof(cplx) * width * height);
        for (int x = 0; x < width; ++x) {
            cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
            cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
            for (int y = 0; y < height; ++y) {
                col_in[y] = temp_all[y * width + x];
            }
            dft(col_in, col_out, height);
            for (int y = 0; y < height; ++y) {
                out_full[y * width + x] = col_out[y];
            }
            free(col_in);
            free(col_out);
        }

        // Конвертируем результат в double
        double* out_full_dbl = (double*)malloc(sizeof(double) * width * height * 2);
        for (int i = 0; i < width * height; i++) {
            out_full_dbl[i * 2] = out_full[i].re;
            out_full_dbl[i * 2 + 1] = out_full[i].im;
        }

        // Собираем на root
        double* out_dbl = (rank == 0) ? (double*)malloc(sizeof(double) * width * height * 2) : NULL;
        MPI_Gatherv(out_full_dbl + displs_rows[rank] * width * 2, sendcounts_dbl[rank], MPI_DOUBLE,
            out_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Конвертируем финальный результат
        if (rank == 0) {
            for (int i = 0; i < width * height; i++) {
                out[i].re = out_dbl[i * 2];
                out[i].im = out_dbl[i * 2 + 1];
            }
            free(out_dbl);
        }

        // Очистка
        free(rows);
        free(displs_rows);
        free(sendcounts_dbl);
        free(senddispls_dbl);
        free(sendbuf_dbl);
        free(recvbuf_dbl);
        free(local_in);
        free(local_temp);
        free(local_temp_dbl);
        free(temp_all_dbl);
        free(temp_all);
        free(out_full);
        free(out_full_dbl);

        return;
    }
#endif

    // Не-MPI версия
    cplx* temp = (cplx*)malloc(sizeof(cplx) * width * height);
    if (!temp) return;

    // rows
    for (int y = 0; y < height; ++y) {
        dft(&in[y * width], &temp[y * width], width);
        if (g_progress_cb) g_progress_cb("Rows (forward)", y + 1, height);
    }

    // columns
    for (int x = 0; x < width; ++x) {
        cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
        cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
        for (int y = 0; y < height; ++y) col_in[y] = temp[y * width + x];
        dft(col_in, col_out, height);
        for (int y = 0; y < height; ++y) out[y * width + x] = col_out[y];
        free(col_in);
        free(col_out);
        if (g_progress_cb) g_progress_cb("Cols (forward)", x + 1, width);
    }
    free(temp);
}

void idft2d(const cplx* in, cplx* out, int width, int height) {
#if defined(USE_MPI)
    if (g_use_mpi) {
        int rank = 0, size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Распределение строк
        int base = height / size;
        int rem = height % size;

        int* rows = (int*)malloc(sizeof(int) * size);
        int* displs_rows = (int*)malloc(sizeof(int) * size);

        int offset = 0;
        for (int i = 0; i < size; ++i) {
            rows[i] = base + (i < rem ? 1 : 0);
            displs_rows[i] = offset;
            offset += rows[i];
        }

        int local_rows = rows[rank];

        // sendcounts в double (2 double на cplx)
        int* sendcounts_dbl = (int*)malloc(sizeof(int) * size);
        int* senddispls_dbl = (int*)malloc(sizeof(int) * size);

        for (int i = 0; i < size; ++i) {
            sendcounts_dbl[i] = rows[i] * width * 2;
            senddispls_dbl[i] = displs_rows[i] * width * 2;
        }

        // Подготовка данных на root
        double* sendbuf_dbl = NULL;
        if (rank == 0) {
            sendbuf_dbl = (double*)malloc(height * width * 2 * sizeof(double));
            for (int i = 0; i < height * width; i++) {
                sendbuf_dbl[i * 2] = in[i].re;
                sendbuf_dbl[i * 2 + 1] = in[i].im;
            }
        }

        double* recvbuf_dbl = (double*)malloc(local_rows * width * 2 * sizeof(double));

        MPI_Barrier(MPI_COMM_WORLD);

        // Scatterv
        MPI_Scatterv(sendbuf_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE,
            recvbuf_dbl, sendcounts_dbl[rank], MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Конвертация обратно в cplx
        cplx* local_in = (cplx*)malloc(sizeof(cplx) * local_rows * width);
        for (int i = 0; i < local_rows * width; i++) {
            local_in[i].re = recvbuf_dbl[i * 2];
            local_in[i].im = recvbuf_dbl[i * 2 + 1];
        }

        // Локальное IDFT по строкам
        cplx* local_temp = (cplx*)malloc(sizeof(cplx) * local_rows * width);
        for (int r = 0; r < local_rows; ++r) {
            idft(&local_in[r * width], &local_temp[r * width], width);
        }

        // Конвертируем в double для передачи
        double* local_temp_dbl = (double*)malloc(sizeof(double) * local_rows * width * 2);
        for (int i = 0; i < local_rows * width; i++) {
            local_temp_dbl[i * 2] = local_temp[i].re;
            local_temp_dbl[i * 2 + 1] = local_temp[i].im;
        }

        // Собираем все результаты
        double* temp_all_dbl = (double*)malloc(sizeof(double) * width * height * 2);
        MPI_Allgatherv(local_temp_dbl, sendcounts_dbl[rank], MPI_DOUBLE,
            temp_all_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE, MPI_COMM_WORLD);

        // Конвертируем обратно в cplx
        cplx* temp_all = (cplx*)malloc(sizeof(cplx) * width * height);
        for (int i = 0; i < width * height; i++) {
            temp_all[i].re = temp_all_dbl[i * 2];
            temp_all[i].im = temp_all_dbl[i * 2 + 1];
        }

        // IDFT по столбцам
        cplx* out_full = (cplx*)malloc(sizeof(cplx) * width * height);
        for (int x = 0; x < width; ++x) {
            cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
            cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
            for (int y = 0; y < height; ++y) {
                col_in[y] = temp_all[y * width + x];
            }
            idft(col_in, col_out, height);
            for (int y = 0; y < height; ++y) {
                out_full[y * width + x] = col_out[y];
            }
            free(col_in);
            free(col_out);
        }

        // Конвертируем результат в double
        double* out_full_dbl = (double*)malloc(sizeof(double) * width * height * 2);
        for (int i = 0; i < width * height; i++) {
            out_full_dbl[i * 2] = out_full[i].re;
            out_full_dbl[i * 2 + 1] = out_full[i].im;
        }

        // Собираем на root
        double* out_dbl = (rank == 0) ? (double*)malloc(sizeof(double) * width * height * 2) : NULL;
        MPI_Gatherv(out_full_dbl + displs_rows[rank] * width * 2, sendcounts_dbl[rank], MPI_DOUBLE,
            out_dbl, sendcounts_dbl, senddispls_dbl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Конвертируем финальный результат
        if (rank == 0) {
            for (int i = 0; i < width * height; i++) {
                out[i].re = out_dbl[i * 2];
                out[i].im = out_dbl[i * 2 + 1];
            }
            free(out_dbl);
        }

        // Очистка
        free(rows);
        free(displs_rows);
        free(sendcounts_dbl);
        free(senddispls_dbl);
        free(sendbuf_dbl);
        free(recvbuf_dbl);
        free(local_in);
        free(local_temp);
        free(local_temp_dbl);
        free(temp_all_dbl);
        free(temp_all);
        free(out_full);
        free(out_full_dbl);

        return;
    }
#endif

    // Не-MPI версия
    cplx* temp = (cplx*)malloc(sizeof(cplx) * width * height);
    if (!temp) return;

    // columns (inverse)
    for (int x = 0; x < width; ++x) {
        cplx* col_in = (cplx*)malloc(sizeof(cplx) * height);
        cplx* col_out = (cplx*)malloc(sizeof(cplx) * height);
        for (int y = 0; y < height; ++y) col_in[y] = in[y * width + x];
        idft(col_in, col_out, height);
        for (int y = 0; y < height; ++y) temp[y * width + x] = col_out[y];
        free(col_in);
        free(col_out);
        if (g_progress_cb) g_progress_cb("Cols (inverse)", x + 1, width);
    }

    // rows (inverse)
    for (int y = 0; y < height; ++y) {
        idft(&temp[y * width], &out[y * width], width);
        if (g_progress_cb) g_progress_cb("Rows (inverse)", y + 1, height);
    }
    free(temp);
}