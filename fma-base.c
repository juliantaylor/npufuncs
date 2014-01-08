void 
FUN(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
#ifdef SUB
        out[i] = a[i] * b[i] - c[i];
#else
        out[i] = a[i] * b[i] + c[i];
#endif
    }
}

