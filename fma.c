#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_logit.c
 * This is the C code for creating your own
 * Numpy ufunc for a logit function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different funciton are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef LogitMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

#define OPTATTR(v) \
__attribute__((target(v))) __attribute__((optimize("fast-math"))) __attribute__((optimize("tree-vectorize")))

void 
fma4(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);// OPTATTR("arch=bdver1");
void 
_fma(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);// OPTATTR("fma");
void 
avx(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n) OPTATTR("avx");
void 
sse2(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n) OPTATTR("sse2");
void
sse2_2(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n) OPTATTR("sse2");


void 
fma4(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

void 
_fma(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

void 
avx(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i] + c[i];
    }
}

void 
sse2(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        //out[i] = fma(a[i], b[i], c[i]);
        out[i] = a[i] * b[i] + c[i];
    }
}

void 
sse2_2(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n)
{
    double * restrict a = __builtin_assume_aligned(a_, 32);
    double * restrict b = __builtin_assume_aligned(b_, 32);
    double * restrict c = __builtin_assume_aligned(c_, 32);
    double * restrict out = __builtin_assume_aligned(out_, 32);
    int i;
    for (i = 0; i < n; i++) {
        //out[i] = fma(a[i], b[i], c[i]);
        out[i] = a[i] * b[i] - c[i];
    }
}

static void double_logit(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    double * in1 = __builtin_assume_aligned(args[0], 16);
    double * in2 = __builtin_assume_aligned(args[1], 16);
    double * in3 = __builtin_assume_aligned(args[2], 16);
    double * out = __builtin_assume_aligned(args[3], 16);
    npy_intp in1s = steps[0] / 8;
    npy_intp in2s = steps[1] / 8;
    npy_intp in3s = steps[2] / 8;
    npy_intp outs = steps[3] / 8;
    if (in1s == 1 && in2s == 1 && in3s == 1 && outs == 1) {
        if (__builtin_cpu_supports("avx")) {
            puts("avx");
            avx(out, in1, in2, in3, n);
        }
        else if (__builtin_cpu_supports("sse2")) {
            //puts("sse2");
            sse2(out, in1, in2, in3, n);
        }
        else if (__builtin_cpu_supports("avx2")) {
            puts("fma");
            _fma(out, in1, in2, in3, n);
        }
        /* no xop ... */
        else if (__builtin_cpu_is("bdver1")) {
            puts("fma4");
            fma4(out, in1, in2, in3, n);
        }
        else {
            puts("norm");
            for (i = 0; i < n; i++) {
                out[i] = in1[i] * in2[i] + in3[i];
            }
        }
    }
    else {
        for (i = 0; i < n; i++) {
            out[i * outs] = in1[i * in1s] * in2[i * in2s] + in3[i * in3s];
        }
    }
}

static void double_logit2(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    double * in1 = __builtin_assume_aligned(args[0], 16);
    double * in2 = __builtin_assume_aligned(args[1], 16);
    double * in3 = __builtin_assume_aligned(args[2], 16);
    double * out = __builtin_assume_aligned(args[3], 16);
    npy_intp in1s = steps[0] / 8;
    npy_intp in2s = steps[1] / 8;
    npy_intp in3s = steps[2] / 8;
    npy_intp outs = steps[3] / 8;
    if (in1s == 1 && in2s == 1 && in3s == 1 && outs == 1) {
        if (__builtin_cpu_supports("avx")) {
            puts("avx");
            avx(out, in1, in2, in3, n);
        }
        else if (__builtin_cpu_supports("sse2")) {
            //puts("sse2");
            sse2_2(out, in1, in2, in3, n);
        }
        else if (__builtin_cpu_supports("avx2")) {
            puts("fma");
            _fma(out, in1, in2, in3, n);
        }
        /* no xop ... */
        else if (__builtin_cpu_is("bdver1")) {
            puts("fma4");
            fma4(out, in1, in2, in3, n);
        }
        else {
            puts("norm");
            for (i = 0; i < n; i++) {
                out[i] = in1[i] * in2[i] + in3[i];
            }
        }
    }
    else {
        for (i = 0; i < n; i++) {
            out[i * outs] = in1[i * in1s] * in2[i * in2s] + in3[i * in3s];
        }
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs_fma[1] = {&double_logit};
PyUFuncGenericFunction funcs_fms[1] = {&double_logit2};

/* These are the input and return dtypes of logit.*/
static char types_fma[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static char types_fms[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void *data_fma[1] = {NULL};
static void *data_fms[1] = {NULL};

PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *fma, *fms, *d;


    m = Py_InitModule("npufunc", LogitMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    fma = PyUFunc_FromFuncAndData(funcs_fma, data_fma, types_fma, 4, 3, 1,
                                    PyUFunc_Zero, "fma",
                                    "logit_docstring", 0);
    fms = PyUFunc_FromFuncAndData(funcs_fms, data_fms, types_fms, 4, 3, 1,
                                    PyUFunc_Zero, "fms",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "fma", fma);
    PyDict_SetItemString(d, "fms", fms);
    Py_DECREF(fma);
    Py_DECREF(fms);
}
