#include <math.h>
#pragma GCC target "sse2"
#define FUN sse2_add
#include "fma-base.c"
#undef FUN
#define FUN sse2_sub
#define SUB
#include "fma-base.c"

void 
libc_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

void 
libc_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        out[i] = fma(a[i], b[i], -c[i]);
    }
}
