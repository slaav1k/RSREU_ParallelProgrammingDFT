// cplx.h - portable complex type and helpers
#ifndef CPLX_H
#define CPLX_H

typedef struct { double re; double im; } cplx;

// basic complex helpers
cplx cplx_add(cplx a, cplx b);
cplx cplx_mul(cplx a, cplx b);
cplx cplx_scale(cplx a, double s);
cplx cplx_from_polar(double r, double theta);

#endif // CPLX_H
