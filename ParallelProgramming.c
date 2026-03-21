#include <stdio.h>
#include "Methods.h"
#include "Image.h"
#include "Transform.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

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


void check_openCL() {
    cl_int err;
    cl_uint num_platforms = 0;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs return %d\n", err);
        // можно return 1; если хочешь сразу выйти
    }
    else if (num_platforms == 0) {
        printf("OpenCL dont find\n");
    }
    else {
        printf("Find OpenCL count: %u\n", num_platforms);

        cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
        clGetPlatformIDs(num_platforms, platforms, NULL);

        for (cl_uint i = 0; i < num_platforms; i++) {
            char name[256] = { 0 };
            char vendor[256] = { 0 };
            char version[256] = { 0 };

            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);

            printf("  Model %u: %s  |  %s  |  %s\n", i, name, vendor, version);

            // Показываем устройства этой платформы
            cl_uint num_devices = 0;
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

            if (num_devices > 0) {
                cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

                for (cl_uint d = 0; d < num_devices; d++) {
                    char dev_name[256] = { 0 };
                    char dev_vendor[256] = { 0 };
                    cl_device_type dev_type;
                    cl_uint compute_units;

                    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
                    clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(dev_vendor), dev_vendor, NULL);
                    clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
                    clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

                    printf("      GPU:  %u: %s  |  %s  |  ", d, dev_name, dev_vendor);

                    if (dev_type & CL_DEVICE_TYPE_GPU)   printf("GPU ");
                    if (dev_type & CL_DEVICE_TYPE_CPU)   printf("CPU ");
                    if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) printf("Accelerator ");
                    printf("  |  Compute units: %u\n", compute_units);
                }
                free(devices);
            }
            else {
                printf("      NO GPU\n\n");
            }
        }
        free(platforms);
    }
}

int main(int argc, char** argv) {
    check_openCL();





    // Включить/выключить OMP здесь в коде
    int flagOMP = 0;
    int use_omp = flagOMP;
    const char* in_file = "resources/input.bmp";
    const char* spectrum_file = "resources/spectrum.bmp";
    const char* recon_file = "resources/recovered.bmp";

	// Информация о BMP
    print_bmp_info(in_file);

    if (use_omp) {
        set_use_omp(1);
        int omp_threads = 12;
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

    //// Создать увеличенные версии исходного изображения (2x и 4x)
    //scale_and_save_from_pixels(pixels, width, height, "resources/input_2.bmp", 2);
    //scale_and_save_from_pixels(pixels, width, height, "resources/input_4.bmp", 4);

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
    dft2d_opencl(in, out, w, h);
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
    dft2d_opencl(out, rec, w, h, 1);
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
