#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Methods.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

static int g_use_mpi = 0;

void set_use_mpi(int use) { g_use_mpi = use; }

void dft(const cplx in[], cplx out[], int n)
{
    for (int k = 0; k < n; k++)
    {
        cplx sum = { 0,0 };

        for (int m = 0; m < n; m++)
        {
            double angle = -2 * PI * k * m / (double)n;
            cplx w = cplx_from_polar(1.0, angle);

            sum = cplx_add(sum, cplx_mul(in[m], w));
        }

        out[k] = sum;
    }
}

void idft(const cplx in[], cplx out[], int n)
{
    for (int k = 0; k < n; k++)
    {
        cplx sum = { 0,0 };

        for (int m = 0; m < n; m++)
        {
            double angle = 2 * PI * k * m / (double)n;
            cplx w = cplx_from_polar(1.0, angle);

            sum = cplx_add(sum, cplx_mul(in[m], w));
        }

        out[k] = cplx_scale(sum, 1.0 / n);
    }
}

void dft2d(const cplx* in, cplx* out, int width, int height)
{

#ifdef USE_MPI
    if (g_use_mpi)
    {
        int rank, size;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rows_per_proc = height / size;
        int remainder = height % size;

        int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

        int* sendcounts = malloc(sizeof(int) * size);
        int* displs = malloc(sizeof(int) * size);

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * width;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        cplx* local_in = malloc(sizeof(cplx) * local_rows * width);

        MPI_Scatterv(in,
            sendcounts,
            displs,
            MPI_DOUBLE_COMPLEX,
            local_in,
            sendcounts[rank],
            MPI_DOUBLE_COMPLEX,
            0,
            MPI_COMM_WORLD);

        cplx* local_temp = malloc(sizeof(cplx) * local_rows * width);

        for (int r = 0; r < local_rows; r++)
            dft(&local_in[r * width], &local_temp[r * width], width);

        cplx* temp_full = malloc(sizeof(cplx) * width * height);

        MPI_Gatherv(local_temp,
            sendcounts[rank],
            MPI_DOUBLE_COMPLEX,
            temp_full,
            sendcounts,
            displs,
            MPI_DOUBLE_COMPLEX,
            0,
            MPI_COMM_WORLD);

        MPI_Bcast(temp_full, width * height, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        int cols_per_proc = width / size;
        int col_rem = width % size;

        int start = rank * cols_per_proc + (rank < col_rem ? rank : col_rem);
        int count = cols_per_proc + (rank < col_rem ? 1 : 0);

        cplx* col_in = malloc(sizeof(cplx) * height);
        cplx* col_out = malloc(sizeof(cplx) * height);

        cplx* partial = calloc(width * height, sizeof(cplx));

        for (int x = start; x < start + count; x++)
        {
            for (int y = 0; y < height; y++)
                col_in[y] = temp_full[y * width + x];

            dft(col_in, col_out, height);

            for (int y = 0; y < height; y++)
                partial[y * width + x] = col_out[y];
        }

        MPI_Reduce(partial,
            out,
            width * height,
            MPI_DOUBLE_COMPLEX,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);

        free(local_in);
        free(local_temp);
        free(temp_full);
        free(col_in);
        free(col_out);
        free(partial);
        free(sendcounts);
        free(displs);

        return;
    }
#endif

    cplx* temp = malloc(sizeof(cplx) * width * height);

    for (int y = 0; y < height; y++)
        dft(&in[y * width], &temp[y * width], width);

    cplx* col_in = malloc(sizeof(cplx) * height);
    cplx* col_out = malloc(sizeof(cplx) * height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
            col_in[y] = temp[y * width + x];

        dft(col_in, col_out, height);

        for (int y = 0; y < height; y++)
            out[y * width + x] = col_out[y];
    }

    free(col_in);
    free(col_out);
    free(temp);
}

void idft2d(const cplx* in, cplx* out, int width, int height)
{

#ifdef USE_MPI
    if (g_use_mpi)
    {
        int rank, size;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rows_per_proc = height / size;
        int remainder = height % size;

        int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

        int* sendcounts = malloc(sizeof(int) * size);
        int* displs = malloc(sizeof(int) * size);

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * width;
            displs[i] = offset;
            offset += sendcounts[i];
        }

        cplx* local_in = malloc(sizeof(cplx) * local_rows * width);

        MPI_Scatterv(in,
            sendcounts,
            displs,
            MPI_DOUBLE_COMPLEX,
            local_in,
            sendcounts[rank],
            MPI_DOUBLE_COMPLEX,
            0,
            MPI_COMM_WORLD);

        cplx* local_temp = malloc(sizeof(cplx) * local_rows * width);

        // IDFT по строкам
        for (int r = 0; r < local_rows; r++)
            idft(&local_in[r * width], &local_temp[r * width], width);

        cplx* temp_full = malloc(sizeof(cplx) * width * height);

        MPI_Gatherv(local_temp,
            sendcounts[rank],
            MPI_DOUBLE_COMPLEX,
            temp_full,
            sendcounts,
            displs,
            MPI_DOUBLE_COMPLEX,
            0,
            MPI_COMM_WORLD);

        MPI_Bcast(temp_full, width * height, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        int cols_per_proc = width / size;
        int col_rem = width % size;

        int start = rank * cols_per_proc + (rank < col_rem ? rank : col_rem);
        int count = cols_per_proc + (rank < col_rem ? 1 : 0);

        cplx* col_in = malloc(sizeof(cplx) * height);
        cplx* col_out = malloc(sizeof(cplx) * height);

        cplx* partial = calloc(width * height, sizeof(cplx));

        // IDFT по столбцам
        for (int x = start; x < start + count; x++)
        {
            for (int y = 0; y < height; y++)
                col_in[y] = temp_full[y * width + x];

            idft(col_in, col_out, height);

            for (int y = 0; y < height; y++)
                partial[y * width + x] = col_out[y];
        }

        MPI_Reduce(partial,
            out,
            width * height,
            MPI_DOUBLE_COMPLEX,
            MPI_SUM,
            0,
            MPI_COMM_WORLD);

        free(local_in);
        free(local_temp);
        free(temp_full);
        free(col_in);
        free(col_out);
        free(partial);
        free(sendcounts);
        free(displs);

        return;
    }
#endif

    // обычная версия без MPI
    cplx* temp = malloc(sizeof(cplx) * width * height);

    cplx* col_in = malloc(sizeof(cplx) * height);
    cplx* col_out = malloc(sizeof(cplx) * height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
            col_in[y] = in[y * width + x];

        idft(col_in, col_out, height);

        for (int y = 0; y < height; y++)
            temp[y * width + x] = col_out[y];
    }

    for (int y = 0; y < height; y++)
        idft(&temp[y * width], &out[y * width], width);

    free(col_in);
    free(col_out);
    free(temp);
}