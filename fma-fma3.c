#pragma GCC target "fma"
#define FUN fma3_add
#include "fma-base.c"
#undef FUN
#define FUN fma3_sub
#define SUB
#include "fma-base.c"
