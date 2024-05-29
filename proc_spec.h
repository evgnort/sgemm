#ifndef _PROC_SPEC_H
#define _PROC_SPEC_H

#define MMX_REGISTRY_BITS 256
#define MMX_REGISTRY_COUNT 16
#define CACHE_LINE_SIZE 64

#define L1_CACHE (32 * 1024)
#define L1_ASSOCIATIVITY 8
#define L1_STRIDE (L1_CACHE / L1_ASSOCIATIVITY)
#define L2_CACHE (256 * 1024)
#define L3_CACHE (6 * 1024 * 1024)

#define MMX_REGISTRY_BYTES (MMX_REGISTRY_BITS / 8)
#define ITEMS_PER_REGISTRY (MMX_REGISTRY_BITS / 8 / sizeof(matrixtype_t))
#define ITEMS_PER_CACHE_LINE (CACHE_LINE_SIZE / sizeof(matrixtype_t))
#define REGISTRIES_PER_CACHE_LINE (CACHE_LINE_SIZE / MMX_REGISTRY_BYTES)

#define PT_LEVEL _MM_HINT_T2
//#define L1_RESERVE

#define MAIN_CORE 2
#define SIBLING_CORE 3

#define MICROCORE_WIDTH 2
#define MICROCORE_HEIGHT 6

#define MICROCORE_WIDTH_2 6
#define MICROCORE_HEIGHT_2 2

#endif // !_PROC_SPEC_H
