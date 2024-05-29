
#ifndef _CALC_ENV_H
#define _CALC_ENV_H

#include "matrixes.h"
#include "proc_spec.h"

#define SYNC_NOT_STARTED 0xFFFFFFFFFFFFFFFE
#define SYNC_FINISHED 0xFFFFFFFFFFFFFFFF

typedef struct FCalcParamsTg
   {
   uint64_t sinhronizer; // Value for thread sinhcronisation
   uint64_t sinhronizer2; // Value for thread sinhcronisation
   uint32_t half_done[2];
   char padding[CACHE_LINE_SIZE * 2 - sizeof(uint64_t)]; // Cache line paddings
   matrixtype_t *A;
   matrixtype_t *B;
   matrixtype_t *C;
   matrixtype_t *micro_A;
   matrixtype_t *micro_B; // Buffer ITEMS_PER_CACHE_LINE * k_step
   matrixtype_t alpha;
   unsigned k_size; // height of B and width of A
   unsigned n_size; // width of B and C
   unsigned m_size; // height of A and C
   unsigned k_step; // height of micro_B buffer and width of micro_A buffer
   unsigned m_step; // height of micro_A buffer
   unsigned n_step;   
   unsigned ap_stride; // Size of micro_A portion
   unsigned k_stride; // Full size of buffer k with padding for local variables
   unsigned error;
   } FCalcParams;

FCalcParams *create_params(const FMatrix *m1, const FMatrix *m2, FMatrix *res,double L1_kt, unsigned core_width, unsigned core_height);
void update_params(FCalcParams *params,double L1_kt, unsigned core_width, unsigned core_height);
void reset_params(FCalcParams *params);
void free_params(FCalcParams *params);

#endif