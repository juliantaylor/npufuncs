#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <stdio.h>
#include <string.h>

#define SSE2 1
#define AVX 2
#define FMA3 3
#define FMA4 4
#define LIBC 5
int type = SSE2;
void sse2_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void avx_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void fma3_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void fma4_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void libc_add(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void sse2_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void avx_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void fma3_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void fma4_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);
void libc_sub(double *restrict out_, double * restrict a_, double * restrict b_, double * restrict c_, int n);

static PyObject *
set_type(PyObject *self, PyObject *args)
{
    const char * tmp;
    if (!PyArg_ParseTuple(args, "s", &tmp))
        return NULL;
    if (strcmp(tmp, "sse2") == 0) {
        printf("using sse2");
        type = SSE2;
    }
    else if (strcmp(tmp, "avx") == 0) {
        printf("using avx");
        type = AVX;
    }
    else if (strcmp(tmp, "fma3") == 0) {
        printf("using fma3");
        type = FMA3;
    }
    else if (strcmp(tmp, "fma4") == 0) {
        printf("using fma4");
        type = FMA4;
    }
    else if (strcmp(tmp, "libc") == 0) {
        printf("using libc");
        type = LIBC;
    }
    else {
        printf("invalid type");
    }
    return PyLong_FromLong(type);
}

static PyMethodDef LogitMethods[] = {
    {"set_type",  set_type, METH_VARARGS},
    {NULL, NULL, 0, NULL}
};

static void fma_add(char **args, npy_intp *dimensions,
                    npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    double * in1 = __builtin_assume_aligned(args[0], 32);
    double * in2 = __builtin_assume_aligned(args[1], 32);
    double * in3 = __builtin_assume_aligned(args[2], 32);
    double * out = __builtin_assume_aligned(args[3], 32);
    npy_intp in1s = steps[0] / sizeof(double);
    npy_intp in2s = steps[1] / sizeof(double);
    npy_intp in3s = steps[2] / sizeof(double);
    npy_intp outs = steps[3] / sizeof(double);
    if (in1s == 1 && in2s == 1 && in3s == 1 && outs == 1) {
        if (type == SSE2) {
            sse2_add(out, in1, in2, in3, n);
        }
        else if (type == AVX) {
            avx_add(out, in1, in2, in3, n);
        }
        else if (type == FMA3) {
            fma3_add(out, in1, in2, in3, n);
        }
        else if (type == FMA4) {
            fma4_add(out, in1, in2, in3, n);
        }
        else if (type == LIBC) {
            libc_add(out, in1, in2, in3, n);
        }
    }
    else {
        for (i = 0; i < n; i++) {
            out[i * outs] = in1[i * in1s] * in2[i * in2s] + in3[i * in3s];
        }
    }
}

static void fma_sub(char **args, npy_intp *dimensions,
                    npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    double * in1 = __builtin_assume_aligned(args[0], 32);
    double * in2 = __builtin_assume_aligned(args[1], 32);
    double * in3 = __builtin_assume_aligned(args[2], 32);
    double * out = __builtin_assume_aligned(args[3], 32);
    npy_intp in1s = steps[0] / sizeof(double);
    npy_intp in2s = steps[1] / sizeof(double);
    npy_intp in3s = steps[2] / sizeof(double);
    npy_intp outs = steps[3] / sizeof(double);
    if (in1s == 1 && in2s == 1 && in3s == 1 && outs == 1) {
        if (type == SSE2) {
            sse2_sub(out, in1, in2, in3, n);
        }
        else if (type == AVX) {
            avx_sub(out, in1, in2, in3, n);
        }
        else if (type == FMA3) {
            fma3_sub(out, in1, in2, in3, n);
        }
        else if (type == FMA4) {
            fma4_sub(out, in1, in2, in3, n);
        }
        else if (type == LIBC) {
            libc_sub(out, in1, in2, in3, n);
        }
    }
    else {
        for (i = 0; i < n; i++) {
            out[i * outs] = in1[i * in1s] * in2[i * in2s] - in3[i * in3s];
        }
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs_fma[1] = {&fma_add};
PyUFuncGenericFunction funcs_fms[1] = {&fma_sub};

/* These are the input and return dtypes of logit.*/
static char types_fma[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static char types_fms[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void *data_fma[1] = {NULL};
static void *data_fms[1] = {NULL};

PyMODINIT_FUNC initnpfma(void)
{
    PyObject *m, *fma, *fms, *d;


    m = Py_InitModule("npfma", LogitMethods);
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
