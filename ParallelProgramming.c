#include <stdio.h>
#include "Methods.h"
#include "Image.h"
#include "Transform.h"
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv)
{

    int rank = 0;
    int size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    set_use_mpi(1);

    if (rank == 0)
        printf("MPI ranks = %d\n", size);
#endif

    unsigned char* pixels = NULL;
    int width = 0;
    int height = 0;

    if (rank == 0)
    {
        print_bmp_info("resources/input_4.bmp");
        load_bmp_grayscale("resources/input_4.bmp", &pixels, &width, &height);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        pixels = malloc(width * height);

    MPI_Bcast(pixels, width * height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    cplx* in = malloc(sizeof(cplx) * width * height);
    cplx* out = malloc(sizeof(cplx) * width * height);
    cplx* rec = malloc(sizeof(cplx) * width * height);

    pixels_to_cplx(pixels, width, height, in);

    if (rank == 0)
        printf("Forward DFT start\n");

    double t0 = MPI_Wtime();

    dft2d(in, out, width, height);

    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Forward time %.3f sec\n", t1 - t0);

    if (rank == 0)
        printf("Inverse DFT start\n");

    double t2 = MPI_Wtime();

    idft2d(out, rec, width, height);

    double t3 = MPI_Wtime();

    if (rank == 0)
        printf("Inverse time %.3f sec\n", t3 - t2);

    if (rank == 0)
    {
        unsigned char* rec_img = malloc(width * height);

        cplx_to_pixels(rec, width, height, rec_img);

        save_bmp_grayscale("resources/recovered.bmp",
            rec_img,
            width,
            height);

        free(rec_img);
    }

    free(pixels);
    free(in);
    free(out);
    free(rec);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}