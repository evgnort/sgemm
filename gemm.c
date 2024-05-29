#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <assert.h>

// OS specific calls
#include "port.h"

#include "proc_spec.h"
#include "matrixes.h"
#include "calc_env.h"

// Common defines

#define MATRIX_SIZE (2*1152)
//#define COUNT_SYNC

void fill_micro_B_6x16(int B_width, int micro_B_height, const matrixtype_t *B, matrixtype_t *micro_B)
   {
   int k;
   for (k = 0; k < micro_B_height; k++, B += B_width, micro_B += MICROCORE_WIDTH * ITEMS_PER_REGISTRY)
      {
      _mm256_stream_ps(micro_B + 0, _mm256_loadu_ps(B + 0));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY));
      }
   }

void fill_micro_A_6x16(int A_width, int micro_A_height, int micro_A_width, int micro_A_step, const matrixtype_t *A, matrixtype_t *micro_A)
   {
   int i,k;
   for (i = 0; i < micro_A_height; i += MICROCORE_HEIGHT)
      {
      matrixtype_t *mA = micro_A;
      for (k = 0; k < micro_A_width; k += 4)
         {
         const float * pA = A + k;
         __m128 a0 = _mm_loadu_ps(pA + 0 * A_width);
         __m128 a1 = _mm_loadu_ps(pA + 1 * A_width);
         __m128 a2 = _mm_loadu_ps(pA + 2 * A_width);
         __m128 a3 = _mm_loadu_ps(pA + 3 * A_width);
         __m128 a4 = _mm_loadu_ps(pA + 4 * A_width);
         __m128 a5 = _mm_loadu_ps(pA + 5 * A_width);
         __m128 a00 = _mm_unpacklo_ps(a0, a2);
         __m128 a01 = _mm_unpacklo_ps(a1, a3);
         __m128 a10 = _mm_unpackhi_ps(a0, a2);
         __m128 a11 = _mm_unpackhi_ps(a1, a3);
         __m128 a20 = _mm_unpacklo_ps(a4, a5);
         __m128 a21 = _mm_unpackhi_ps(a4, a5);
         _mm_storeu_ps(mA + 0 * MICROCORE_HEIGHT, _mm_unpacklo_ps(a00, a01));
         _mm_storel_pi((__m64*)(mA + 0 * MICROCORE_HEIGHT + 4), a20);
         _mm_storeu_ps(mA + 1 * MICROCORE_HEIGHT, _mm_unpackhi_ps(a00, a01));
         _mm_storeh_pi((__m64*)(mA + 10), a20);
         _mm_storeu_ps(mA + 2 * MICROCORE_HEIGHT, _mm_unpacklo_ps(a10, a11));
         _mm_storel_pi((__m64*)(mA + 16), a21);
         _mm_storeu_ps(mA + 3 * MICROCORE_HEIGHT, _mm_unpackhi_ps(a10, a11));
         _mm_storeh_pi((__m64*)(mA + 22), a21);
         mA += MICROCORE_HEIGHT * 4;
         }
      micro_A += micro_A_step;
      A += MICROCORE_HEIGHT * A_width;
      }
   }

void micro_core_6x16(matrixtype_t alpha,unsigned B_height, unsigned C_width, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *C)
   {
   unsigned k;
   __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
   __m256 c10 = c00, c11 = c01, c20 = c00, c21 = c01;
   __m256 c30 = c00, c31 = c01, c40 = c00, c41 = c01, c50 = c00, c51 = c01;

   __m256 b0, b1, a0, a1;

   b0 = _mm256_loadu_ps(B + 0);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY);   
   B += MICROCORE_WIDTH * ITEMS_PER_REGISTRY; 

   a0 = _mm256_set1_ps(A[0]);                  
   a1 = _mm256_set1_ps(A[1]);                  
   c00 = _mm256_fmadd_ps(a0, b0, c00);
   c01 = _mm256_fmadd_ps(a0, b1, c01);
   _mm_prefetch((const char *)&C[0 * C_width],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * C_width],PT_LEVEL);
   c10 = _mm256_fmadd_ps(a1, b0, c10);
   c11 = _mm256_fmadd_ps(a1, b1, c11);
   a0 = _mm256_set1_ps(A[2]);                  
   a1 = _mm256_set1_ps(A[3]);                  
   c20 = _mm256_fmadd_ps(a0, b0, c20);
   c21 = _mm256_fmadd_ps(a0, b1, c21);
   _mm_prefetch((const char *)&C[2 * C_width],PT_LEVEL);
   _mm_prefetch((const char *)&C[3 * C_width],PT_LEVEL);
   c30 = _mm256_fmadd_ps(a1, b0, c30);
   c31 = _mm256_fmadd_ps(a1, b1, c31);
   a0 = _mm256_set1_ps(A[4]);               
   a1 = _mm256_set1_ps(A[5]);            
   c40 = _mm256_fmadd_ps(a0, b0, c40);
   c41 = _mm256_fmadd_ps(a0, b1, c41);
   _mm_prefetch((const char *)&C[4 * C_width],PT_LEVEL);
   _mm_prefetch((const char *)&C[5 * C_width],PT_LEVEL);
   c50 = _mm256_fmadd_ps(a1, b0, c50);
   c51 = _mm256_fmadd_ps(a1, b1, c51);
   A += MICROCORE_HEIGHT;

   for (k = 1; k < B_height; k++)
      {
      b0 = _mm256_loadu_ps(B + 0);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY);   
      B += MICROCORE_WIDTH * ITEMS_PER_REGISTRY; 

      a0 = _mm256_set1_ps(A[0]);                  
      a1 = _mm256_set1_ps(A[1]);                  
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      a0 = _mm256_set1_ps(A[2]);                  
      a1 = _mm256_set1_ps(A[3]);                  
      c20 = _mm256_fmadd_ps(a0, b0, c20);
      c21 = _mm256_fmadd_ps(a0, b1, c21);
      c30 = _mm256_fmadd_ps(a1, b0, c30);
      c31 = _mm256_fmadd_ps(a1, b1, c31);
      a0 = _mm256_set1_ps(A[4]);                  
      a1 = _mm256_set1_ps(A[5]);                  
      c40 = _mm256_fmadd_ps(a0, b0, c40);
      c41 = _mm256_fmadd_ps(a0, b1, c41);
      c50 = _mm256_fmadd_ps(a1, b0, c50);
      c51 = _mm256_fmadd_ps(a1, b1, c51);
      A += MICROCORE_HEIGHT;
      }
   a0 = _mm256_set1_ps(alpha);

   b0 = _mm256_fmadd_ps(a0,c00,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, b0);
   b1 = _mm256_fmadd_ps(a0,c01,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, b1);
   C += C_width;
   b0 = _mm256_fmadd_ps(a0,c10,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, b0);
   b1 = _mm256_fmadd_ps(a0,c11,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, b1);
   C += C_width;
   b0 = _mm256_fmadd_ps(a0,c20,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, b0);
   b1 = _mm256_fmadd_ps(a0,c21,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, b1);
   C += C_width;
   c00 = _mm256_fmadd_ps(a0,c30,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c00);
   c01 = _mm256_fmadd_ps(a0,c31,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c01);
   C += C_width;
   c10 = _mm256_fmadd_ps(a0,c40,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c10);
   c11 = _mm256_fmadd_ps(a0,c41,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c11);
   C += C_width;
   c20 = _mm256_fmadd_ps(a0,c50,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c20);
   c21 = _mm256_fmadd_ps(a0,c51,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c21); 

/* Has little to no effect, possible problem is long latency of _mm256_set1_ps
   b0 = _mm256_loadu_ps(B + 0);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY);   
   a0 = _mm256_set1_ps(A[0]);                  
   a1 = _mm256_set1_ps(A[1]);                  
   c00 = _mm256_fmadd_ps(a0, b0, c00);
   c01 = _mm256_fmadd_ps(a0, b1, c01);
   a0 = _mm256_set1_ps(alpha);
   c10 = _mm256_fmadd_ps(a1, b0, c10);
   c11 = _mm256_fmadd_ps(a1, b1, c11);
   a1 = _mm256_fmadd_ps(a0,c00, _mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0,a1);   
   c00 = _mm256_set1_ps(A[2]);                  
   a1 = _mm256_set1_ps(A[3]);                  
   c20 = _mm256_fmadd_ps(c00, b0, c20);
   c21 = _mm256_fmadd_ps(c00, b1, c21);
   c00 = _mm256_fmadd_ps(a0,c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY,c00);
   C += C_width;
   c30 = _mm256_fmadd_ps(a1, b0, c30);
   c31 = _mm256_fmadd_ps(a1, b1, c31);
   a1 = _mm256_fmadd_ps(a0,c10, _mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, a1);
   c00 = _mm256_set1_ps(A[4]);                  
   c01 = _mm256_set1_ps(A[5]);
   c10 = _mm256_fmadd_ps(a0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c10);
   C += C_width;
   c40 = _mm256_fmadd_ps(c00, b0, c40);
   c41 = _mm256_fmadd_ps(c00, b1, c41);
   a1 = _mm256_fmadd_ps(a0,c20, _mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, a1);
   c50 = _mm256_fmadd_ps(c01, b0, c50);
   c51 = _mm256_fmadd_ps(c01, b1, c51);
   b0 = _mm256_fmadd_ps(a0,c21, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, b0);
   C += C_width;
   c00 = _mm256_fmadd_ps(a0,c30,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c00);
   c01 = _mm256_fmadd_ps(a0,c31,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c01);
   C += C_width;
   c10 = _mm256_fmadd_ps(a0,c40,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c10);
   c11 = _mm256_fmadd_ps(a0,c41,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c11);
   C += C_width;
   c20 = _mm256_fmadd_ps(a0,c50,_mm256_loadu_ps(C + 0));
   _mm256_storeu_ps(C + 0, c20);
   c21 = _mm256_fmadd_ps(a0,c51,_mm256_loadu_ps(C + ITEMS_PER_REGISTRY));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY, c21); */
   }

void one_6x16(FCalcParams *params)
   {
   unsigned i,j,k,n,m,ni;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

         for (m = 0; m < params->m_size; m += params->m_step)
            { 
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;

            fill_micro_A_6x16(params->k_size,micro_A_height,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);

            for (j = 0; j < npart; j += MICROCORE_WIDTH * ITEMS_PER_REGISTRY)
               { 
               if (!m)
                  fill_micro_B_6x16(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride]);
               for (i = 0, ni = 0; i < micro_A_height; i += MICROCORE_HEIGHT, ni++) // Пробег по вертикали матрицы A
                  micro_core_6x16(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
               }
            }
         }
      }
   }

#ifdef COUNT_SYNC
   #define INC_CORE_COUNT core_cnt++
   #define INC_SYNC_COUNT(N) sync_cnt[(N)]++
   #define INIT_COUNTS int core_cnt = 0,sync_cnt[2] = {0,0}
   #define PRINTF_COUNTS(N) printf("%d: core: %d, sync: %d, %d\n",(N),core_cnt,sync_cnt[0],sync2_cnt[1])
#else
   #define INC_CORE_COUNT
   #define INC_SYNC_COUNT(N)
   #define INIT_COUNTS
   #define PRINTF_COUNTS(N)
#endif

#define SYNC1to2(VAL,SYNC1,SYNC2,M,N,J,K) while ((VAL) != (SYNC2)) { _mm_pause(); INC_SYNC_COUNT(0); } \
               (VAL) = ((uint64_t)(J) << 48) + ((uint64_t)(M) << 32) + ((uint64_t)(N) << 16) + (K); \
               (SYNC1) = (VAL); SOFT_BARRIER;

#define SYNC2to1(VAL,NVAL,SYNC1,SYNC2,CNTN) SOFT_BARRIER; (SYNC2) = (VAL); \
               while ((VAL) == ((NVAL) = (SYNC1))) { _mm_pause(); INC_SYNC_COUNT((CNTN)); } \
               (VAL) = (NVAL);

#define SYNCHALF(VAL1,VAL2) (VAL1) = 1; SOFT_BARRIER; \
         while (1 != (VAL2)) { _mm_pause(); } \
         (VAL2) = 0;   SOFT_BARRIER;

THREAD_FUNC ThreadProc_6x16(void *lpParameter)
   {
   FCalcParams *params = (FCalcParams *)lpParameter;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   unsigned i,j,k,m,n,ni,prev_m = 0xFFFFFFFF;
   while (*sync == SYNC_NOT_STARTED)
      _mm_pause();
   if (get_thread_processor() != SIBLING_CORE)   params->error = 1;
   uint64_t npos, pos = *sync;
   while (pos != SYNC_FINISHED)
      {
      k = pos & 0xFFFF;
      n = (pos >> 16) & 0xFFFF;
      m = (pos >> 32) & 0xFFFF;
      j = (pos >> 48) & 0xFFFF;
      unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
      unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;
      unsigned mah2 = micro_A_height / 2;

      if (m != prev_m)
         fill_micro_A_6x16(params->k_size,mah2,micro_B_height,params->ap_stride,&params->A[(m + mah2) * params->k_size + k],
                              &params->micro_A[mah2 / MICROCORE_HEIGHT * params->ap_stride]), prev_m = m;

      for (ni = mah2 / MICROCORE_HEIGHT,i = micro_A_height / 2; i < micro_A_height; i += MICROCORE_HEIGHT,ni++) 
         micro_core_6x16(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);

      SYNC2to1(pos,npos,*sync,*sync2,0);
      }
   THREAD_RETURN(0);
   }

void half_6x16(FCalcParams *params)
   {
   unsigned i,j,k,n,m,ni;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   uint64_t npos = SYNC_NOT_STARTED;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

         for (m = 0; m < params->m_size; m += params->m_step)
            { 
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;
            unsigned mah2 = micro_A_height / 2;

            fill_micro_A_6x16(params->k_size,mah2,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);
      
            for (j = 0; j < npart; j += MICROCORE_WIDTH * ITEMS_PER_REGISTRY)
               { 
               if (!m)
                  fill_micro_B_6x16(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride]);
               SYNC1to2(npos,*sync,*sync2,m,n,j,k);

               for (ni = 0,i = 0; i < mah2; i += MICROCORE_HEIGHT, ni++) // Пробег по вертикали матрицы A
                  micro_core_6x16(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
               }
            }
         }
      }
   }

void fill_micro_B_2x48(int B_width, int micro_B_height, const matrixtype_t *B, matrixtype_t *micro_B)
   {
   int k;
   for (k = 0; k < micro_B_height; k++, B += B_width, micro_B += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
      {
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 0, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 1, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 2, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 3, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 4, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4));
      _mm256_stream_ps(micro_B + ITEMS_PER_REGISTRY * 5, _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5));
      }
   }

void fill_micro_A_2x48(int A_width, int micro_A_height, int micro_B_height, int micro_A_step,const matrixtype_t *A, matrixtype_t *micro_A)
   {
   int i,k;

   for (i = 0; i < micro_A_height; i += MICROCORE_HEIGHT_2)
      {
      matrixtype_t *mA = micro_A;
      for (k = 0; k < micro_B_height; k += ITEMS_PER_CACHE_LINE / MICROCORE_HEIGHT_2)
         {
         const float *pA = A + k;
         __m256 a0 = _mm256_loadu_ps(pA + 0 * A_width);
         __m256 a1 = _mm256_loadu_ps(pA + 1 * A_width);

         __m256 a00 = _mm256_unpacklo_ps(a0, a1);
         __m256 a01 = _mm256_unpackhi_ps(a0, a1);

         _mm256_storeu_ps(mA, a00);
         _mm256_storeu_ps(mA + ITEMS_PER_REGISTRY, a01);

         mA += MICROCORE_HEIGHT_2 * ITEMS_PER_CACHE_LINE / MICROCORE_HEIGHT_2;
         }
      micro_A += micro_A_step;
      A += MICROCORE_HEIGHT_2 * A_width;
      }
   }

void fill_micro_A_2x48_panel(int A_width, int micro_B_height, const matrixtype_t *A, matrixtype_t *micro_A)
   {
   int k;

   for (k = 0; k < micro_B_height; k += ITEMS_PER_CACHE_LINE / MICROCORE_HEIGHT_2)
      {
      __m256 a0 = _mm256_loadu_ps(A + 0 * A_width);
      __m256 a1 = _mm256_loadu_ps(A + 1 * A_width);

      __m256 a00 = _mm256_unpacklo_ps(a0, a1);
      __m256 a01 = _mm256_unpackhi_ps(a0, a1);

      _mm256_storeu_ps(micro_A, a00);
      _mm256_storeu_ps(micro_A + ITEMS_PER_REGISTRY, a01);

      micro_A += MICROCORE_HEIGHT_2 * ITEMS_PER_CACHE_LINE / MICROCORE_HEIGHT_2;
      A += ITEMS_PER_REGISTRY;
      }
   }

#ifdef _WIN32 // Ugly removing xmm store/load on each call under windows
__forceinline 
#endif 
void micro_core_2x48(matrixtype_t alpha,unsigned B_height, unsigned C_width, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *C)
   {
   unsigned k;
   __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
   __m256 c03 = _mm256_setzero_ps(), c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
   __m256 c10 = c00, c11 = c01, c12 = c02, c13 = c03, c14 = c04, c15 = c05; 

   __m256 b0, b1, a0, a1;

   // First iteration - placing prefetches
   a0 = _mm256_set1_ps(A[0]);   
   a1 = _mm256_set1_ps(A[1]);   
   A += MICROCORE_HEIGHT_2;
   b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);   
   c00 = _mm256_fmadd_ps(a0, b0, c00);
   c10 = _mm256_fmadd_ps(a1, b0, c10);
   _mm_prefetch((const char *)&C[0 * C_width],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * C_width + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   c01 = _mm256_fmadd_ps(a0, b1, c01);
   c11 = _mm256_fmadd_ps(a1, b1, c11);
   b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);   
   c02 = _mm256_fmadd_ps(a0, b0, c02);
   c12 = _mm256_fmadd_ps(a1, b0, c12);
   _mm_prefetch((const char *)&C[0 * C_width + ITEMS_PER_REGISTRY * 4],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * C_width],PT_LEVEL);
   c03 = _mm256_fmadd_ps(a0, b1, c03);
   c13 = _mm256_fmadd_ps(a1, b1, c13);
   b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);   
   c04 = _mm256_fmadd_ps(a0, b0, c04);
   c14 = _mm256_fmadd_ps(a1, b0, c14);
   _mm_prefetch((const char *)&C[1 * C_width + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * C_width + ITEMS_PER_REGISTRY * 4],PT_LEVEL);
   c05 = _mm256_fmadd_ps(a0, b1, c05);
   c15 = _mm256_fmadd_ps(a1, b1, c15);
   B += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 

   for (k = 1; k < B_height - 1; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);   
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);   
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);   
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      B += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      }
   // Last iteration - alleviating data dependency through registry puzzle
   a0 = _mm256_set1_ps(A[0]);   
   a1 = _mm256_set1_ps(A[1]);   
   b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);   
   c00 = _mm256_fmadd_ps(a0, b0, c00);
   c10 = _mm256_fmadd_ps(a1, b0, c10);
   b0 = _mm256_set1_ps(alpha);
   c01 = _mm256_fmadd_ps(a0, b1, c01);
   c11 = _mm256_fmadd_ps(a1, b1, c11);
   b1 = _mm256_fmadd_ps(b0,c00, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, b1);
   c00 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
   b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);   
   c02 = _mm256_fmadd_ps(a0, c00, c02);
   c12 = _mm256_fmadd_ps(a1, c00, c12);
   c00 = _mm256_fmadd_ps(b0,c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, c00);
   c03 = _mm256_fmadd_ps(a0, b1, c03);
   c13 = _mm256_fmadd_ps(a1, b1, c13);
   b1 = _mm256_fmadd_ps(b0,c02, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, b1);
   c00 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
   c01 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);   
   c02 = _mm256_fmadd_ps(b0,c03, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, c02);
   c04 = _mm256_fmadd_ps(a0, c00, c04);
   c14 = _mm256_fmadd_ps(a1, c00, c14);
   b1 = _mm256_fmadd_ps(b0,c04, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, b1);
   c05 = _mm256_fmadd_ps(a0, c01, c05);
   c15 = _mm256_fmadd_ps(a1, c01, c15);
   a1 = _mm256_fmadd_ps(b0,c05, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, a1);
   C += C_width;
   c00 = _mm256_fmadd_ps(b0,c10, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, c00);
   c01 = _mm256_fmadd_ps(b0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, c00);
   c02 = _mm256_fmadd_ps(b0,c12, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, c02);
   c03 = _mm256_fmadd_ps(b0,c13, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, c03);
   c04 = _mm256_fmadd_ps(b0,c14, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, c04);
   c05 = _mm256_fmadd_ps(b0,c15, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, c05);
   }

void micro_core_2x48_fip(matrixtype_t alpha,unsigned B_height, unsigned N_size, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *mB, matrixtype_t *C)
   {
   unsigned k;
   __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
   __m256 c03 = _mm256_setzero_ps(), c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
   __m256 c10 = c00, c11 = c01, c12 = c02, c13 = c03, c14 = c04, c15 = c05; 

   _mm_prefetch((const char *)&C[0 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);

   __m256 b0, b1, a0, a1;

   for (k = 0; k < B_height; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 0, b0);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 1, b1);
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 2, b0);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 3, b1);
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 4, b0);
      _mm256_stream_ps(mB + ITEMS_PER_REGISTRY * 5, b1);
      mB += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      B += N_size;
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      }
   // we are already tight on port 4, let's go straight
   b0 = _mm256_set1_ps(alpha);
   b1 = _mm256_fmadd_ps(b0,c00, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, b1);
   a1 = _mm256_fmadd_ps(b0,c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, a1);
   b1 = _mm256_fmadd_ps(b0,c02, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, b1);
   a1 = _mm256_fmadd_ps(b0,c03, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, a1);
   b1 = _mm256_fmadd_ps(b0,c04, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, b1);
   a1 = _mm256_fmadd_ps(b0,c05, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, a1);
   C += N_size;
   c00 = _mm256_fmadd_ps(b0,c10, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, c00);
   c01 = _mm256_fmadd_ps(b0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, c00);
   c02 = _mm256_fmadd_ps(b0,c12, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, c02);
   c03 = _mm256_fmadd_ps(b0,c13, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, c03);
   c04 = _mm256_fmadd_ps(b0,c14, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, c04);
   c05 = _mm256_fmadd_ps(b0,c15, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, c05);
   }

void micro_core_2x48_fip_h1(matrixtype_t alpha,unsigned B_height, unsigned N_size, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *mB, matrixtype_t *C,
                              volatile unsigned *sync)
   {
   unsigned k;
   __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
   __m256 c03 = _mm256_setzero_ps(), c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
   __m256 c10 = c00, c11 = c01, c12 = c02, c13 = c03, c14 = c04, c15 = c05; 

   __m256 b0, b1, a0, a1;

   for (k = 0; k < B_height / 2; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 0, b0);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 1, b1);
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 2, b0);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 3, b1);
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 4, b0);
      _mm256_storeu_ps(mB + ITEMS_PER_REGISTRY * 5, b1);
      mB += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      B += N_size;
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      }
   _mm_prefetch((const char *)&C[0 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);

   SYNCHALF(*sync,*(sync+1));
   for (k = B_height / 2; k < B_height; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 1);
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 3);
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 5);
      mB += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      }

   b0 = _mm256_set1_ps(alpha);

   b1 = _mm256_fmadd_ps(b0,c00, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, b1);
   a1 = _mm256_fmadd_ps(b0,c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, a1);
   b1 = _mm256_fmadd_ps(b0,c02, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, b1);
   a1 = _mm256_fmadd_ps(b0,c03, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, a1);
   b1 = _mm256_fmadd_ps(b0,c04, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, b1);
   a1 = _mm256_fmadd_ps(b0,c05, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, a1);
   C += N_size;
   c00 = _mm256_fmadd_ps(b0,c10, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, c00);
   c01 = _mm256_fmadd_ps(b0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, c00);
   c02 = _mm256_fmadd_ps(b0,c12, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, c02);
   c03 = _mm256_fmadd_ps(b0,c13, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, c03);
   c04 = _mm256_fmadd_ps(b0,c14, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, c04);
   c05 = _mm256_fmadd_ps(b0,c15, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, c05);
   }

void micro_core_2x48_fip_h2(matrixtype_t alpha,unsigned B_height, unsigned N_size, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *mB, matrixtype_t *C,
                              volatile unsigned *sync)
   {
   unsigned k, bh2 = B_height / 2;
   __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
   __m256 c03 = _mm256_setzero_ps(), c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
   __m256 c10 = c00, c11 = c01, c12 = c02, c13 = c03, c14 = c04, c15 = c05; 

   __m256 b0, b1, a0, a1;

   matrixtype_t *mB2 = mB + bh2 * MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY;
   B += bh2 * N_size;
   for (k = bh2; k < B_height; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 1);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 0, b0);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 1, b1);
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 3);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 2, b0);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 3, b1);
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(B + ITEMS_PER_REGISTRY * 5);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 4, b0);
      _mm256_storeu_ps(mB2 + ITEMS_PER_REGISTRY * 5, b1);
      mB2 += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      B += N_size;
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      }

   _mm_prefetch((const char *)&C[0 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[0 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 2],PT_LEVEL);
   _mm_prefetch((const char *)&C[1 * N_size + ITEMS_PER_REGISTRY * 4],PT_LEVEL);

   SYNCHALF(*(sync + 1),*sync);
   for (k = 0; k < bh2; k++)
      {
      a0 = _mm256_set1_ps(A[0]);   
      a1 = _mm256_set1_ps(A[1]);   
      A += MICROCORE_HEIGHT_2;
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 0);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 1);
      c00 = _mm256_fmadd_ps(a0, b0, c00);
      c10 = _mm256_fmadd_ps(a1, b0, c10);
      c01 = _mm256_fmadd_ps(a0, b1, c01);
      c11 = _mm256_fmadd_ps(a1, b1, c11);
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 2);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 3);
      c02 = _mm256_fmadd_ps(a0, b0, c02);
      c12 = _mm256_fmadd_ps(a1, b0, c12);
      c03 = _mm256_fmadd_ps(a0, b1, c03);
      c13 = _mm256_fmadd_ps(a1, b1, c13);
      b0 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 4);                     
      b1 = _mm256_loadu_ps(mB + ITEMS_PER_REGISTRY * 5);
      mB += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; 
      c04 = _mm256_fmadd_ps(a0, b0, c04);
      c14 = _mm256_fmadd_ps(a1, b0, c14);
      c05 = _mm256_fmadd_ps(a0, b1, c05);
      c15 = _mm256_fmadd_ps(a1, b1, c15);
      }

   b0 = _mm256_set1_ps(alpha);

   b1 = _mm256_fmadd_ps(b0,c00, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, b1);
   a1 = _mm256_fmadd_ps(b0,c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, a1);
   b1 = _mm256_fmadd_ps(b0,c02, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, b1);
   a1 = _mm256_fmadd_ps(b0,c03, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, a1);
   b1 = _mm256_fmadd_ps(b0,c04, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, b1);
   a1 = _mm256_fmadd_ps(b0,c05, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, a1);
   C += N_size;
   c00 = _mm256_fmadd_ps(b0,c10, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, c00);
   c01 = _mm256_fmadd_ps(b0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, c00);
   c02 = _mm256_fmadd_ps(b0,c12, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, c02);
   c03 = _mm256_fmadd_ps(b0,c13, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, c03);
   c04 = _mm256_fmadd_ps(b0,c14, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, c04);
   c05 = _mm256_fmadd_ps(b0,c15, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
   _mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, c05);
   }

void one_2x48(FCalcParams *params)
   {
   unsigned i,j,k,n,m,ni;

   for(n = 0; n < params->n_size; n += params->n_step)
      {
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         {
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

         for (m = 0; m < params->m_size; m += params->m_step)
            {
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;

            fill_micro_A_2x48(params->k_size,micro_A_height,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);

            for (j = 0; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               if (!m)
                  fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride]);
               for (ni = 0,i = 0; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
               }
            }
         }
      }
   }

void one_2x48_fip(FCalcParams *params)
   {
   unsigned i,j,k,n,m,ni;

   for(n = 0; n < params->n_size; n += params->n_step)
      {
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         {
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         unsigned micro_A_height = params->m_step;

         // m = 0, j = 0
         fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[k],params->micro_A);
         micro_core_2x48_fip(params->alpha,micro_B_height,params->n_size,params->micro_A,&params->B[k * params->n_size + n],
                              params->micro_B,&params->C[n]);

         for (ni = 1,i = MICROCORE_HEIGHT_2; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],&params->micro_A[ni * params->ap_stride]);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride],
                                 params->micro_B,&params->C[i*params->n_size + n]);
            }

         // m = 0; j != 0;
         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {            
            micro_core_2x48_fip(params->alpha,micro_B_height,params->n_size,params->micro_A,&params->B[k * params->n_size + n + j],
                                    &params->micro_B[j * params->k_stride],&params->C[n + j]);
            for (ni = 1,i = MICROCORE_HEIGHT_2; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride],
                                       &params->micro_B[j * params->k_stride],&params->C[i*params->n_size + n + j]);
               }
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            {
            micro_A_height = min(params->m_size, m + params->m_step) - m;

            // m != 0; j = 0;
            for (ni = 0,i = 0; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],&params->micro_A[ni * params->ap_stride]);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride],params->micro_B,&params->C[(m+i)*params->n_size + n]);
               }

            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               { // m != 0; j != 0
               for (ni = 0,i = 0; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  }
               }
            }
         }
      }
   }

THREAD_FUNC ThreadProc_2x48_sym_fip(void *lpParameter)
   {
   FCalcParams *params = (FCalcParams *)lpParameter;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;

   unsigned i,j,k,m,n;
   INIT_COUNTS;

   while (*sync == SYNC_NOT_STARTED)
      _mm_pause();
   if (get_thread_processor() != SIBLING_CORE)   params->error = 1;
   uint64_t npos, pos = *sync;

   while (pos != SYNC_FINISHED)
      {
      k = pos & 0xFFFF;
      n = (pos >> 16) & 0xFFFF;
      m = (pos >> 32) & 0xFFFF;
      j = (pos >> 48) & 0xFFFF;
      unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
      unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;
      unsigned mah2 = micro_A_height / 2;
      matrixtype_t *mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride];
      const matrixtype_t *pA = &params->A[(m + mah2) * params->k_size + k];

      if (!m)
         { // Filling Bbuf
         if (!j)
            { // Filling A panel
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,pA,mA);
            micro_core_2x48_fip_h2(params->alpha,micro_B_height,params->n_size,mA,&params->B[k * params->n_size + n],
                                    params->micro_B,&params->C[mah2*params->n_size + n],params->half_done);
            INC_CORE_COUNT;

            for (i = mah2 + MICROCORE_HEIGHT_2; i < micro_A_height; i += MICROCORE_HEIGHT_2) 
               {
               mA += params->ap_stride;
               pA += params->k_size;
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,pA,mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA,params->micro_B,&params->C[i*params->n_size + n]);
               INC_CORE_COUNT;
               }
            }
         else
            {
            micro_core_2x48_fip_h2(params->alpha,micro_B_height,params->n_size,mA,&params->B[k * params->n_size + n + j],
                                    &params->micro_B[j * params->k_stride],&params->C[mah2*params->n_size + n + j],params->half_done);
            INC_CORE_COUNT;

            for (i = mah2 + MICROCORE_HEIGHT_2; i < micro_A_height; i += MICROCORE_HEIGHT_2) 
               {
               mA += params->ap_stride;
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA,params->micro_B,&params->C[i*params->n_size + n + j]);
               INC_CORE_COUNT;
               }
            }
         }
      else
         {
         if (!j)
            { // Filling A panel
            for (i = mah2; i < micro_A_height; i += MICROCORE_HEIGHT_2) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,pA,mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA,params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               mA += params->ap_stride;
               pA += params->k_size;
               }
            }
         else
            {
            for (i = mah2; i < micro_A_height; i += MICROCORE_HEIGHT_2) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA,&params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
               mA += params->ap_stride;
               INC_CORE_COUNT;
               }
            }
         }

      SYNC2to1(pos,npos,*sync,*sync2,0);
      }
   PRINTF_COUNTS(2);
   THREAD_RETURN(0);
   }

void half_2x48_fip(FCalcParams *params)
   {
   unsigned i,j,k,n,m,ni;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   uint64_t npos = SYNC_NOT_STARTED;
   INIT_COUNTS;

   for(n = 0; n < params->n_size; n += params->n_step)
      {
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         {
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         // m = 0, j = 0 - filling A panels, filling Bbuf
         unsigned micro_A_height = params->m_step;
         unsigned mah2 = micro_A_height / 2;

         SYNC1to2(npos,*sync,*sync2,0,n,0,k);
               
         fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[k],params->micro_A);

         micro_core_2x48_fip_h1(params->alpha,micro_B_height,params->n_size,params->micro_A,&params->B[k * params->n_size + n],params->micro_B,&params->C[n],params->half_done);
         INC_CORE_COUNT;

         for (ni = 1,i = MICROCORE_HEIGHT_2; i < mah2; i += MICROCORE_HEIGHT_2, ni++)
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],&params->micro_A[ni * params->ap_stride]);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride],params->micro_B,&params->C[i*params->n_size + n]);
            INC_CORE_COUNT;
            }
      
         // m = 0; j != 0, filling Bbuf
         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {
            SYNC1to2(npos,*sync,*sync2,0,n,j,k);

            micro_core_2x48_fip_h1(params->alpha,micro_B_height,params->n_size,params->micro_A,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride],
                                    &params->C[n + j],params->half_done);
            INC_CORE_COUNT;
               
            for (ni = 1,i = MICROCORE_HEIGHT_2; i < mah2; i += MICROCORE_HEIGHT_2, ni++) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],
                                    &params->C[i*params->n_size + n + j]);
               INC_CORE_COUNT;
               }
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            {
            // m != 0, j = 0 - filling a panels
            micro_A_height = min(params->m_size, m + params->m_step) - m;
            mah2 = micro_A_height / 2;

            SYNC1to2(npos,*sync,*sync2,m,n,0,k);
               
            for (ni = 0,i = 0; i < mah2; i += MICROCORE_HEIGHT_2, ni++) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],&params->micro_A[ni * params->ap_stride]);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride],params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               }
      
            // m != 0, j !- 0 - nothing to fill
            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               SYNC1to2(npos,*sync,*sync2,m,n,j,k);
               
               for (ni = 0,i = 0; i < mah2; i += MICROCORE_HEIGHT_2, ni++) 
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  INC_CORE_COUNT;
                  }
               }
            }
         }
      }
   PRINTF_COUNTS(1);
   }

#define B_FILL_DELAY (MICROCORE_HEIGHT_2 * 5)

THREAD_FUNC ThreadProc_2x48_sym(void *lpParameter)
   {
   FCalcParams *params = (FCalcParams *)lpParameter;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   unsigned i,j,k,m,n;
   matrixtype_t *mA;
   INIT_COUNTS;

   while (*sync == SYNC_NOT_STARTED)
      _mm_pause();
   if (get_thread_processor() != SIBLING_CORE)   params->error = 1;
   uint64_t npos, pos = *sync;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         unsigned micro_A_height = params->m_step;
         unsigned mah2 = micro_A_height / 2 - B_FILL_DELAY;
         m = j = 0;

         for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],mA);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[i*params->n_size + n]);
            INC_CORE_COUNT;
             }
         SYNC2to1(pos,npos,*sync,*sync2,1);

         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {
            for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[i*params->n_size + n + j]);
               INC_CORE_COUNT;
               }
            SYNC2to1(pos,npos,*sync,*sync2,1);
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            { 
            micro_A_height = min(params->m_size, m + params->m_step) - m;
            mah2 = micro_A_height / 2;
            j = 0;
               
            for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               }
            SYNC2to1(pos,npos,*sync,*sync2,0);

            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  INC_CORE_COUNT;
                  }
               SYNC2to1(pos,npos,*sync,*sync2,0);
               }
            }
         }
      }
   PRINTF_COUNTS(2);
   THREAD_RETURN(0);
   }

// Do nothing usefull, just stall for some cycles
#define GRAIN    _mm_prefetch((const char *) &params->micro_B[j * params->k_stride],_MM_HINT_T0); \
            _mm_prefetch((const char *) &params->micro_B[j * params->k_stride + 1],_MM_HINT_T0); \
            _mm_prefetch((const char *) &params->micro_B[j * params->k_stride + 2],_MM_HINT_T0); \
            _mm_prefetch((const char *) &params->micro_B[j * params->k_stride + 3],_MM_HINT_T0)

THREAD_FUNC ThreadProc_2x48_asym(void *lpParameter)
   {
   FCalcParams *params = (FCalcParams *)lpParameter;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   unsigned i,j,k,m,n;
   matrixtype_t *mA;
   INIT_COUNTS;

   while (*sync == SYNC_NOT_STARTED)
      _mm_pause();
   if (get_thread_processor() != SIBLING_CORE)   params->error = 1;
   uint64_t npos, pos = *sync;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         unsigned micro_A_height = params->m_step;
         unsigned mah2 = micro_A_height / 2 - B_FILL_DELAY;
         m = j = 0;

         for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],mA);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[i*params->n_size + n]);
            INC_CORE_COUNT;
            GRAIN;
            }
         SYNC2to1(pos,npos,*sync,*sync2,1);

         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {
            for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[i*params->n_size + n + j]);
               INC_CORE_COUNT;
               GRAIN;
               }
            SYNC2to1(pos,npos,*sync,*sync2,1);
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            { 
            micro_A_height = min(params->m_size, m + params->m_step) - m;
            mah2 = micro_A_height / 2;
            j = 0;
               
            for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               GRAIN;
               }

            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               for (i = mah2, mA = &params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]; i < micro_A_height; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  INC_CORE_COUNT;
                  GRAIN;
                  }
               }
            SYNC2to1(pos,npos,*sync,*sync2,0);
            }
         }
      }
   PRINTF_COUNTS(2);
   THREAD_RETURN(0);
   }

void half_2x48(FCalcParams *params)
   {
   unsigned i,j,k,n,m;
   matrixtype_t *mA;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   uint64_t npos = SYNC_NOT_STARTED;
   INIT_COUNTS;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         unsigned micro_A_height = params->m_step;
         unsigned mah2 = micro_A_height / 2 - B_FILL_DELAY;
         m = j = 0;

         fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],params->micro_B);
         SYNC1to2(npos,*sync,*sync2,m,n,j,k);
               
         for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],mA);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[i * params->n_size + n + j]);
            INC_CORE_COUNT;
            }

         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {
            fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride]);
            SYNC1to2(npos,*sync,*sync2,m,n,j,k);
               
            for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
               INC_CORE_COUNT;
               }
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            { 
            micro_A_height = min(params->m_size, m + params->m_step) - m;
            mah2 = micro_A_height / 2;
               
            SYNC1to2(npos,*sync,*sync2,m,n,0,k);
            for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               }

            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               SYNC1to2(npos,*sync,*sync2,m,n,j,k);
               for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  INC_CORE_COUNT;
                  }
               }
            }
         }
      }
   PRINTF_COUNTS(1);
   }

void half_2x48_rare_sync(FCalcParams *params)
   {
   unsigned i,j,k,n,m;
   matrixtype_t *mA;
   volatile uint64_t *sync = &params->sinhronizer;
   volatile uint64_t *sync2 = &params->sinhronizer2;
   uint64_t npos = SYNC_NOT_STARTED;
   INIT_COUNTS;

   for(n = 0; n < params->n_size; n += params->n_step)
      { 
      unsigned npart = min(params->n_size, n + params->n_step) - n;

      for(k = 0; k < params->k_size; k += params->k_step)
         { 
         unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;
         unsigned micro_A_height = params->m_step;
         unsigned mah2 = micro_A_height / 2 - B_FILL_DELAY;

         fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n],params->micro_B);
         SYNC1to2(npos,*sync,*sync2,0,n,0,k);
               
         for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
            {
            fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[i*params->k_size + k],mA);
            micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, params->micro_B,&params->C[i*params->n_size + n]);
            INC_CORE_COUNT;
            }

         for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
            {
            fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_stride]);
            SYNC1to2(npos,*sync,*sync2,0,n,j,k);
               
            for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[i*params->n_size + n + j]);
               INC_CORE_COUNT;
               }
            }

         for (m = params->m_step; m < params->m_size; m += params->m_step)
            { 
            micro_A_height = min(params->m_size, m + params->m_step) - m;
            mah2 = micro_A_height / 2;
            SYNC1to2(npos,*sync,*sync2,m,n,0,k);
               
            for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
               {
               fill_micro_A_2x48_panel(params->k_size,micro_B_height,&params->A[(m+i)*params->k_size + k],mA);
               micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA,params->micro_B,&params->C[(m+i)*params->n_size + n]);
               INC_CORE_COUNT;
               }

            for (j = MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
               {
               for (i = 0, mA = params->micro_A; i < mah2; i += MICROCORE_HEIGHT_2, mA += params->ap_stride) 
                  {
                  micro_core_2x48(params->alpha,micro_B_height,params->n_size,mA, &params->micro_B[j * params->k_stride],&params->C[(m+i)*params->n_size + n + j]);
                  INC_CORE_COUNT;
                  }
               }
            }
         }
      }
   PRINTF_COUNTS(1);
   }

THREAD_FUNC ThreadProc_C1_Dummy(void *lpParameter)
   {
   int i = 9500000;

   while (--i)
      _mm_pause();

   THREAD_RETURN(0);
   }

void stop_thread(FCalcParams *params,THREAD_ID_TYPE cache_runner)
   {
   volatile uint64_t *sync = &params->sinhronizer;
   *sync = SYNC_FINISHED;
   wait_thread(cache_runner);
   }

typedef void (CMMFunc)(FCalcParams *params);

#ifdef _DEBUG
#define CYCLE_COUNT 1
#else
#define CYCLE_COUNT 100
#endif // _DEBUG

void measure(char *name,CMMFunc func,CThreadRoutine thread,FCalcParams *params)
   {
   int i, err = 0;
   double rv = -1.0;
   THREAD_ID_TYPE thid = 0;

   double avg = 0;
   if (get_thread_processor() != MAIN_CORE)
      {
      printf("%15s   failed to run on cpu 2\n",name);
      return;
      }
   for (i = 0; i < CYCLE_COUNT; i++)
      {
      reset_params(params);

      if (thread)
         thid = start_thread(thread,params,SIBLING_CORE);

      int64_t t1 = get_nanotime();
      func(params);
      int64_t t2 = get_nanotime();
      if (thread)
         stop_thread(params,thid);
      if (params->error)
         {
         printf("%15s   failed to run on cpu 3\n",name);
         return;
         }
      double res = (double)params->m_size * params->n_size * params->k_size * 2.0/(double)(t2 - t1);
      avg += res;
      if (res > rv)
         rv = res;
      err += check_result(params->C,params->n_size,params->m_size,params->k_size);
      }
   avg /= (double)CYCLE_COUNT;
   printf("%15s   %7.3f (%7.3f) %d errors\n",name,avg,rv,err);
   }

int main(void)
   {
   maximize_priority();
   set_thread_processor(MAIN_CORE);

   FMatrix *A = alloc_matrix(MATRIX_SIZE,MATRIX_SIZE);
   FMatrix *B = alloc_matrix(MATRIX_SIZE,MATRIX_SIZE);
   FMatrix *C = alloc_matrix(MATRIX_SIZE,MATRIX_SIZE);

   double try_kts[] = {0.5,0.75};

   int i;

   FCalcParams *params = create_params(A,B,C,0.5,MICROCORE_WIDTH,MICROCORE_HEIGHT);

#ifndef _DEBUG
   for (i = 0; i < 10; i++)
      one_6x16(params); // Speedup cpu
#endif // !_DEBUG

   for (i = 0; i < sizeof(try_kts) / sizeof(double); i++)
      {
      update_params(params,try_kts[i],MICROCORE_WIDTH,MICROCORE_HEIGHT);
      printf("L1 kt %.3f, k_step %d\n",try_kts[i],params->k_step);
      measure("One6x16",one_6x16,NULL,params);
      measure("HT6x16",half_6x16,ThreadProc_6x16,params);
      update_params(params,try_kts[i],MICROCORE_WIDTH_2,MICROCORE_HEIGHT_2);
      printf("L1 kt %.3f, k_step %d\n",try_kts[i],params->k_step);
      measure("One2x48",one_2x48,NULL,params);
      measure("One2x48 Dummy",one_2x48,ThreadProc_C1_Dummy,params);
      measure("One2x48fip",one_2x48_fip,NULL,params);
      measure("HT2x48 sym fip",half_2x48_fip,ThreadProc_2x48_sym_fip,params);
      measure("HT2x48 sym",half_2x48,ThreadProc_2x48_sym,params);
      measure("HT2x48 asym",half_2x48_rare_sync,ThreadProc_2x48_asym,params);
      }

   free_params(params);
   delete_matrix(A);
   delete_matrix(B);
   delete_matrix(C);

   printf("Press any key\n");
   getchar();
   return 0;
   }