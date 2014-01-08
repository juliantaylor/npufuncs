#pragma GCC target "avx"
#define FUN avx_add
#include "fma-base.c"
#undef FUN
#define FUN avx_sub
#define SUB
#include "fma-base.c"
