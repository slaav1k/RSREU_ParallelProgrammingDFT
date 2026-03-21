// Структура должна совпадать с cplx из Complex.h
typedef struct {
    double re;
    double im;
} cplx_cl;

__kernel void dft_1d_kernel(__global const cplx_cl* in, 
                            __global cplx_cl* out, 
                            int n, 
                            int stride, 
                            int is_inverse) {
    int line_idx = get_global_id(1); // Индекс строки или столбца
    int k = get_global_id(0);        // Индекс элемента внутри 1D массива

    if (k >= n) return;

    const __global cplx_cl* line_in = &in[line_idx * stride];
    __global cplx_cl* line_out = &out[line_idx * stride];

    double sum_re = 0;
    double sum_im = 0;
    double sign = is_inverse ? 1.0 : -1.0;
    double PI = 3.141592653589793;

    for (int m = 0; m < n; m++) {
        double angle = sign * 2.0 * PI * k * m / (double)n;
        double cos_a = cos(angle);
        double sin_a = sin(angle);
        
        // Комплексное умножение: (a.re + i*a.im) * (cos_a + i*sin_a)
        sum_re += line_in[m].re * cos_a - line_in[m].im * sin_a;
        sum_im += line_in[m].re * sin_a + line_in[m].im * cos_a;
    }

    if (is_inverse) {
        line_out[k].re = sum_re / (double)n;
        line_out[k].im = sum_im / (double)n;
    } else {
        line_out[k].re = sum_re;
        line_out[k].im = sum_im;
    }
}