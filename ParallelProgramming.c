#include <stdio.h>
#include "Methods.h"
#include "Image.h"
#include "Transform.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

// Прогресс бар
static void progress_cb(const char* stage, int current, int total) {
    int pct = (int)((current * 100) / (double)total);
    int bar_width = 30;
    int filled = (pct * bar_width) / 100;
    printf("\r%16s: %4d/%4d [", stage, current, total);
    for (int i = 0; i < filled; ++i) putchar('=');
    for (int i = filled; i < bar_width; ++i) putchar(' ');
    printf("] %3d%%", pct);
    if (current >= total) putchar('\n');
    fflush(stdout);
}

// Масштабирование изображения (ближайший сосед) и сохранение
static int scale_and_save_from_pixels(unsigned char* pixels, int width, int height, const char* out_path, int scale) {
    if (!pixels || width <= 0 || height <= 0 || scale <= 0) return -1;
    int w2 = width * scale;
    int h2 = height * scale;
    unsigned char* out = (unsigned char*)malloc((size_t)w2 * h2);
    if (!out) return -1;
    for (int y = 0; y < h2; ++y) {
        int sy = y / scale;
        for (int x = 0; x < w2; ++x) {
            int sx = x / scale;
            out[y * w2 + x] = pixels[sy * width + sx];
        }
    }
    int res = save_bmp_grayscale(out_path, out, w2, h2);
    if (res == 0) printf("Scaled image saved: %s\n", out_path);
    else printf("Failed to save scaled image: %s\n", out_path);
    free(out);
    return res;
}

int main(int argc, char** argv) {
    // Включить/выключить OMP / MPI здесь в коде
    int flagOMP = 0;
    int flagMPI = 1; // set to 1 to enable MPI mode (requires build with USE_MPI and mpiexec)
    int use_omp = flagOMP;
    int use_mpi = flagMPI;
    const char* in_file = "resources/input.bmp";
    const char* spectrum_file = "resources/spectrum.bmp";
    const char* recon_file = "resources/recovered.bmp";

    int mpi_rank = 0, mpi_size = 1;
    unsigned char* pixels = NULL;
    int width = 0, height = 0;

    if (use_omp) {
        set_use_omp(1);
        int omp_threads = 12;
        set_num_threads(omp_threads);
        printf("OpenMP mode: enabled (threads=%d) (requires build with OpenMP support)\n", omp_threads);
    }
    else {
        set_use_omp(0);
        printf("OpenMP mode: disabled\n");
    }

    if (use_mpi) {
#if defined(USE_MPI)
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        set_use_mpi(1);
        if (mpi_rank == 0) printf("MPI mode: enabled (USE_MPI), ranks=%d\n", mpi_size);

        // load image only on MPI root
        if (mpi_rank == 0) {
            print_bmp_info(in_file);
            if (load_bmp_grayscale(in_file, &pixels, &width, &height) != 0) {
                printf("Failed to load %s\n", in_file);
                MPI_Finalize();
                return 1;
            }
        }

        // рассылаем размеры всем процессам
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Если не rank 0, выделяем память под pixels
        if (mpi_rank != 0) {
            pixels = (unsigned char*)malloc(width * height);
            if (!pixels) {
                printf("Rank %d: Failed to allocate pixels\n", mpi_rank);
                MPI_Finalize();
                return 1;
            }
        }

        // Рассылаем сами пиксели всем процессам
        MPI_Bcast(pixels, width * height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
#else
        printf("MPI mode requested but binary not built with USE_MPI\n");
        return 1;
#endif
    }
    else {
        // Non-MPI version
        print_bmp_info(in_file);
        if (load_bmp_grayscale(in_file, &pixels, &width, &height) != 0) {
            printf("Failed to load %s\n", in_file);
            return 1;
        }
    }

    int w = width, h = height;
    cplx* in = (cplx*)malloc(sizeof(cplx) * w * h);
    cplx* out = (cplx*)malloc(sizeof(cplx) * w * h);
    cplx* rec = (cplx*)malloc(sizeof(cplx) * w * h);
    if (!in || !out || !rec) { printf("Memory alloc failed\n"); return 1; }

    // Преобразование в комплексный формат
    pixels_to_cplx(pixels, w, h, in);

    if (mpi_rank == 0) printf("\nForward DFT start\n");

    set_progress_callback(progress_cb);

    // Замер времени для прямого DFT
    double t0 = MPI_Wtime();
    dft2d(in, out, w, h);
    double t1 = MPI_Wtime();

    if (mpi_rank == 0) {
        double forward_ms = (t1 - t0) * 1000.0;
        printf("Forward DFT is done for %.1f ms\n", forward_ms);
    }

    // Отключить прогресс коллбек для сохранения изображений
    set_progress_callback(NULL);

    // Сохранить спектр в виде изображения (только на rank 0)
    if (mpi_rank == 0) {
        unsigned char* spec_img = (unsigned char*)malloc(w * h);
        cplx_to_spectrum_image(out, w, h, spec_img);
        if (save_bmp_grayscale(spectrum_file, spec_img, w, h) == 0) printf("Spectrum saved: %s\n", spectrum_file);
        free(spec_img);
    }

    if (mpi_rank == 0) printf("\nInverse DFT start\n");

    // Замер времени для обратного DFT
    set_progress_callback(progress_cb);
    double t2 = MPI_Wtime();
    idft2d(out, rec, w, h);
    double t3 = MPI_Wtime();

    if (mpi_rank == 0) {
        double inverse_ms = (t3 - t2) * 1000.0;
        printf("Inverse DFT is done for %.1f ms\n", inverse_ms);
    }

    // Сохранить восстановленное изображение (только на rank 0)
    if (mpi_rank == 0) {
        unsigned char* rec_img = (unsigned char*)malloc(w * h);
        cplx_to_pixels(rec, w, h, rec_img);
        if (save_bmp_grayscale(recon_file, rec_img, w, h) == 0) printf("Recovered image saved: %s\n", recon_file);
        free(rec_img);
    }

    free(pixels); free(in); free(out); free(rec);

    if (use_mpi) {
#if defined(USE_MPI)
        MPI_Finalize();
#endif
    }

    return 0;
}