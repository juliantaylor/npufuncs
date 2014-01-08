#pragma GCC target "arch=bdver1"
#define FUN fma4_add
#include "fma-base.c"
#undef FUN
#define FUN fma4_sub
#define SUB
#include "fma-base.c"
