#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Methods.h"
#ifdef _OPENMP
#include <omp.h>
#include <CL/cl.h>
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

// Вспомогательная функция для чтения файла кернела
char* load_kernel_source(const char* filename) {
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    char* src = (char*)malloc(size + 1);
    fread(src, 1, size, f);
    src[size] = '\0';
    fclose(f);
    return src;
}

void dft2d_opencl(const cplx* in, cplx* out, int width, int height, int is_inverse) {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_rows, kernel_cols;

    // 1. Инициализация OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("OpenCL platform error\n"); return; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) { printf("OpenCL device error\n"); return; }
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    // 2. Загрузка и сборка кернела
    char* src = load_kernel_source("dft_kernel.cl");
    if (!src) { printf("Failed to load kernel\n"); return; }

    program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Вывод ошибок компиляции
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error: %s\n", log);
        free(log);
        free(src);
        return;
    }

    kernel_rows = clCreateKernel(program, "dft_1d_kernel", &err);
    kernel_cols = clCreateKernel(program, "dft_1d_kernel", &err);

    // 3. Подготовка буферов
    size_t img_size = sizeof(cplx) * width * height;
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, img_size, (void*)in, &err);
    cl_mem buf_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, img_size, NULL, &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, NULL, &err);

    // 4. Обработка строк (по горизонтали)
    int inv = is_inverse;
    int n = width;
    int stride = width;

    clSetKernelArg(kernel_rows, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel_rows, 1, sizeof(cl_mem), &buf_temp);
    clSetKernelArg(kernel_rows, 2, sizeof(int), &n);
    clSetKernelArg(kernel_rows, 3, sizeof(int), &stride);
    clSetKernelArg(kernel_rows, 4, sizeof(int), &inv);

    size_t global_rows[2] = { width, height };
    err = clEnqueueNDRangeKernel(queue, kernel_rows, 2, NULL, global_rows, NULL, 0, NULL, NULL);
    clFinish(queue);

    // 5. Обработка столбцов (по вертикали)
    // Для этого используем stride = height, чтобы кернел работал по столбцам
    n = height;
    stride = height;  // шаг между элементами одного столбца

    clSetKernelArg(kernel_cols, 0, sizeof(cl_mem), &buf_temp);
    clSetKernelArg(kernel_cols, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(kernel_cols, 2, sizeof(int), &n);
    clSetKernelArg(kernel_cols, 3, sizeof(int), &stride);
    clSetKernelArg(kernel_cols, 4, sizeof(int), &inv);

    // Для столбцов: global_id[0] = строка (0..height-1), global_id[1] = столбец (0..width-1)
    // Но нам нужно обрабатывать каждый столбец как отдельный 1D массив
    size_t global_cols[2] = { height, width };
    err = clEnqueueNDRangeKernel(queue, kernel_cols, 2, NULL, global_cols, NULL, 0, NULL, NULL);
    clFinish(queue);

    // 6. Чтение результата
    err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, img_size, out, 0, NULL, NULL);

    // 7. Очистка
    free(src);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_temp);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel_rows);
    clReleaseKernel(kernel_cols);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
