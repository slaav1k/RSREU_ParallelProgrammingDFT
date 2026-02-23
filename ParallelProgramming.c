#include <stdio.h>
#include "Methods.h"
#include "Image.h"
#include "Transform.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

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

int main(int argc, char** argv) {
    // Включить/выключить OMP здесь в коде
    int flagOMP = 1;
    int use_omp = flagOMP;
    const char* in_file = "resources/input.bmp";
    const char* spectrum_file = "resources/spectrum.bmp";
    const char* recon_file = "resources/recovered.bmp";

	// Информация о BMP
    print_bmp_info(in_file);

    if (use_omp) {
        set_use_omp(1);
        int omp_threads = 16;
        set_num_threads(omp_threads);
        printf("OpenMP mode: enabled (threads=%d) (requires build with OpenMP support)\n", omp_threads);
    } else {
        set_use_omp(0);
        printf("OpenMP mode: disabled\n");
    }

    unsigned char* pixels = NULL;
    int width = 0, height = 0;
    if (load_bmp_grayscale(in_file, &pixels, &width, &height) != 0) {
        printf("Failed to load %s\n", in_file);
        return 1;
    }

    int w = width, h = height;
    cplx* in = (cplx*)malloc(sizeof(cplx) * w * h);
    cplx* out = (cplx*)malloc(sizeof(cplx) * w * h);
    cplx* rec = (cplx*)malloc(sizeof(cplx) * w * h);
    if (!in || !out || !rec) { printf("Memory alloc failed\n"); return 1; }

	// Преобразование в комплексный формат
    pixels_to_cplx(pixels, w, h, in);

    printf("\nForward DFT start\n");
    
    set_progress_callback(progress_cb);

	// Замер времени для прямого DFT
    clock_t t0 = clock();
    dft2d(in, out, w, h);
    clock_t t1 = clock();
    double forward_ms = (double)(t1 - t0) * 1000.0 / (double)CLOCKS_PER_SEC;
    printf("Forward DFT is done for %.1f ms\n", forward_ms);

	// Отключить прогресс коллбек для сохранения изображений
    set_progress_callback(NULL);

	// Сохранить спектр в виде изображения
    unsigned char* spec_img = (unsigned char*)malloc(w*h);
    cplx_to_spectrum_image(out, w, h, spec_img);
    if (save_bmp_grayscale(spectrum_file, spec_img, w, h) == 0) printf("Spectrum saved: %s\n", spectrum_file);


    printf("\nInverse DFT start\n");
	// Замер времени для обратного DFT
    set_progress_callback(progress_cb);
    clock_t t2 = clock();
    idft2d(out, rec, w, h);
    clock_t t3 = clock();
    double inverse_ms = (double)(t3 - t2) * 1000.0 / (double)CLOCKS_PER_SEC;
    printf("Inverse DFT is done for %.1f ms\n", inverse_ms);

    unsigned char* rec_img = (unsigned char*)malloc(w*h);
    cplx_to_pixels(rec, w, h, rec_img);
    if (save_bmp_grayscale(recon_file, rec_img, w, h) == 0) printf("Recovered image saved: %s\n", recon_file);

    free(pixels); free(in); free(out); free(rec); free(spec_img); free(rec_img);

    return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
