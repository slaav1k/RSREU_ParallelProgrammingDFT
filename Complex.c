// Complex.c - реализация операций с комплексными числами
#include <math.h>
#include "Complex.h"

cplx cplx_add(cplx a, cplx b) {
    cplx r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

cplx cplx_mul(cplx a, cplx b) {
    cplx r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

cplx cplx_scale(cplx a, double s) {
    cplx r;
    r.re = a.re * s;
    r.im = a.im * s;
    return r;
}

cplx cplx_from_polar(double r, double theta) {
    cplx out;
    out.re = r * cos(theta);
    out.im = r * sin(theta);
    return out;
}
