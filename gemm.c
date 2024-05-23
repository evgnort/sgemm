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
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));		
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c01, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	C += C_width;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	C += C_width;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c21, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	C += C_width;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c31, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	C += C_width;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c41, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	C += C_width;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY, _mm256_add_ps(c51, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY)));
	}

void one_6x16(FCalcParams *params)
	{
	unsigned i,j,k,n,m,ni;

	for(n = 0; n < params->n_size; n += params->n_step)
		{ // Пробег по горизонтали матрицы B шагами по L3
		unsigned npart = min(params->n_size, n + params->n_step) - n;

		for(k = 0; k < params->k_size; k += params->k_step)
			{ // Пробег по горизонтали матрицы A (вертикали матрицы B) шагами по L1
			unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

			for (m = 0; m < params->m_size; m += params->m_step)
				{ // Пробег по вертикали матрицы A шагами по L2
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;

				fill_micro_A_6x16(params->k_size,micro_A_height,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);

				for (j = 0; j < npart; j += MICROCORE_WIDTH * ITEMS_PER_REGISTRY)
					{ // Пробег по горизонтали матрицы B
					if (!m)
						fill_micro_B_6x16(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_full_step]);
					for (i = 0, ni = 0; i < micro_A_height; i += MICROCORE_HEIGHT, ni++) // Пробег по вертикали матрицы A
						micro_core_6x16(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
					}
				}
			}
		}
	}

THREAD_FUNC ThreadProc_6x16(void *lpParameter)
	{
	FCalcParams *params = (FCalcParams *)lpParameter;
	volatile uint64_t *sync = &params->sinhronizer;
	volatile uint64_t *sync2 = &params->sinhronizer2;
	unsigned i,j,k,m,n,ni,prev_m = 0xFFFFFFFF;
	while (*sync == SYNC_NOT_STARTED)
		_mm_pause();
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
			micro_core_6x16(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);

		SOFT_BARRIER;
		*sync2 = pos;
		while (pos == (npos = *sync))
			_mm_pause();
		pos = npos;
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
		{ // Пробег по горизонтали матрицы B шагами по L3
		unsigned npart = min(params->n_size, n + params->n_step) - n;

		for(k = 0; k < params->k_size; k += params->k_step)
			{ // Пробег по горизонтали матрицы A (вертикали матрицы B) шагами по L1
			unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

			for (m = 0; m < params->m_size; m += params->m_step)
				{ // Пробег по вертикали матрицы A шагами по L2
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;
				unsigned mah2 = micro_A_height / 2;

				fill_micro_A_6x16(params->k_size,mah2,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);
		
				for (j = 0; j < npart; j += MICROCORE_WIDTH * ITEMS_PER_REGISTRY)
					{ 
					if (!m)
						fill_micro_B_6x16(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_full_step]);
					while (npos != *sync2)
						_mm_pause();
					npos = ((uint64_t)j << 48) + ((uint64_t)m << 32) + ((uint64_t)n << 16) + k;
					*sync = npos;
					SOFT_BARRIER;
					for (ni = 0,i = 0; i < mah2; i += MICROCORE_HEIGHT, ni++) // Пробег по вертикали матрицы A
						micro_core_6x16(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
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
		assert(k == micro_B_height);
		micro_A += micro_A_step;
		A += MICROCORE_HEIGHT_2 * A_width;
		}
	assert(i == micro_A_height);
	}

void micro_core_2x48(matrixtype_t alpha,unsigned B_height, unsigned C_width, const matrixtype_t *A, const matrixtype_t *B, matrixtype_t *C)
	{
	unsigned k;
	__m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
	__m256 c03 = _mm256_setzero_ps(), c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
	__m256 c10 = c00, c11 = c01, c12 = c02, c13 = c03, c14 = c04, c15 = c05; 

   __m256 b0, b1, a0, a1;

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

   for (k = 1; k < B_height; k++)
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

	C += C_width;

	b1 = _mm256_fmadd_ps(b0,c10, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 0));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 0, b1);
	a1 = _mm256_fmadd_ps(b0,c11, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 1));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 1, a1);
	b1 = _mm256_fmadd_ps(b0,c12, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 2));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 2, b1);
	a1 = _mm256_fmadd_ps(b0,c13, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 3));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 3, a1);
	b1 = _mm256_fmadd_ps(b0,c14, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 4));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 4, b1);
	a1 = _mm256_fmadd_ps(b0,c15, _mm256_loadu_ps(C + ITEMS_PER_REGISTRY * 5));
	_mm256_storeu_ps(C + ITEMS_PER_REGISTRY * 5, a1);
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
						fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_full_step]);
					for (ni = 0,i = 0; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++)
						{
						if ((m+i) == 2302 && !(n + j) && k == 2268)
							micro_core_2x48(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
						else
							micro_core_2x48(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
						}
					}
				}
			}
		}
	}

THREAD_FUNC ThreadProc_2x48_sym(void *lpParameter)
	{
	FCalcParams *params = (FCalcParams *)lpParameter;
	volatile uint64_t *sync = &params->sinhronizer;
	volatile uint64_t *sync2 = &params->sinhronizer2;
	unsigned i,j,k,m,n,ni,prev_m = 0;
#ifdef COUNT_SYNC
	int sync_cnt = 0,core_cnt = 0;
#endif

	while (*sync == SYNC_NOT_STARTED)
		_mm_pause();
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
			fill_micro_A_2x48(params->k_size,mah2,micro_B_height,params->ap_stride,&params->A[(m + mah2) * params->k_size + k],
										&params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]), prev_m = m;

		int from = mah2;
		if (!m)
			from -= MICROCORE_HEIGHT * 2;

		for (ni = from / MICROCORE_HEIGHT,i = from; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++) 
			{
			micro_core_2x48(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
#ifdef COUNT_SYNC
			core_cnt++;
#endif
			}

		SOFT_BARRIER;
		*sync2 = pos;
		while (pos == (npos = *sync))
			{ 
			_mm_pause();
#ifdef COUNT_SYNC
			sync_cnt++; 
#endif
			}
		pos = npos;
		}
#ifdef COUNT_SYNC
	printf("2: core: %d, sync: %d\n",core_cnt,sync_cnt);
#endif
	THREAD_RETURN(0);
	}

THREAD_FUNC ThreadProc_2x48_asym(void *lpParameter)
	{
	FCalcParams *params = (FCalcParams *)lpParameter;
	volatile uint64_t *sync = &params->sinhronizer;
	volatile uint64_t *sync2 = &params->sinhronizer2;
	unsigned i,j,k,m,n,ni,prev_m = 0;
#ifdef COUNT_SYNC
	int sync_cnt = 0,core_cnt = 0;
#endif

	while (*sync == SYNC_NOT_STARTED)
		_mm_pause();
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
			fill_micro_A_2x48(params->k_size,mah2,micro_B_height,params->ap_stride,&params->A[(m + mah2) * params->k_size + k],
										&params->micro_A[mah2 / MICROCORE_HEIGHT_2 * params->ap_stride]), prev_m = m;

		int from = micro_A_height / 2;
		if (!m)
			from -= MICROCORE_HEIGHT * 2;

		for (ni = from / MICROCORE_HEIGHT,i = from; i < micro_A_height; i += MICROCORE_HEIGHT_2, ni++) 
			{
			micro_core_2x48(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
#ifdef COUNT_SYNC
			core_cnt++;
#endif
			// grain. Do nothing usefull, just stall for some cycles
			_mm_prefetch((const char *) &params->micro_B[j * params->k_full_step],_MM_HINT_T0);
			_mm_prefetch((const char *) &params->micro_B[j * params->k_full_step + 1],_MM_HINT_T0);
			_mm_prefetch((const char *) &params->micro_B[j * params->k_full_step + 2],_MM_HINT_T0);
			_mm_prefetch((const char *) &params->micro_B[j * params->k_full_step + 3],_MM_HINT_T0);
			}

		SOFT_BARRIER;
		*sync2 = pos;
		while (pos == (npos = *sync))
			{ 
			_mm_pause(); 
#ifdef COUNT_SYNC
			sync_cnt++; 
#endif
			}
		pos = npos;
		}
#ifdef COUNT_SYNC
	printf("2: core: %d, sync: %d\n",core_cnt,sync_cnt);
#endif
	THREAD_RETURN(0);
	}

void half_2x48(FCalcParams *params)
	{
	unsigned i,j,k,n,m,ni;
	volatile uint64_t *sync = &params->sinhronizer;
	volatile uint64_t *sync2 = &params->sinhronizer2;
	uint64_t npos = SYNC_NOT_STARTED;
	int first_run = 1;
//	int cnt = 0, cnt2 = 0;

	for(n = 0; n < params->n_size; n += params->n_step)
		{ // Пробег по горизонтали матрицы B шагами по L3
		unsigned npart = min(params->n_size, n + params->n_step) - n;

		for(k = 0; k < params->k_size; k += params->k_step)
			{ // Пробег по горизонтали матрицы A (вертикали матрицы B) шагами по L1
			unsigned micro_B_height = min(params->k_size, k + params->k_step) - k;

			for (m = 0; m < params->m_size; m += params->m_step)
				{ // Пробег по вертикали матрицы A шагами по L2
            unsigned micro_A_height = min(params->m_size, m + params->m_step) - m;

				fill_micro_A_2x48(params->k_size,micro_A_height / first_run,micro_B_height,params->ap_stride,&params->A[m * params->k_size + k],params->micro_A);
				first_run = 2;
		
				for (j = 0; j < npart; j += MICROCORE_WIDTH_2 * ITEMS_PER_REGISTRY)
					{ // Пробег по горизонтали матрицы B
					unsigned upto = micro_A_height / 2;
					if (!m)
						{
						fill_micro_B_2x48(params->n_size,micro_B_height,&params->B[k * params->n_size + n + j],&params->micro_B[j * params->k_full_step]);
						upto -= MICROCORE_HEIGHT * 2;
						}
					while (npos != *sync2)
						{ 
						_mm_pause(); 
//						cnt++; 
						}
					npos = ((uint64_t)j << 48) + ((uint64_t)m << 32) + ((uint64_t)n << 16) + k;
					*sync = npos;
					SOFT_BARRIER;
					
					for (ni = 0,i = 0; i < upto; i += MICROCORE_HEIGHT_2, ni++) // Пробег по вертикали матрицы A
						{
						micro_core_2x48(1.0,micro_B_height,params->n_size,&params->micro_A[ni * params->ap_stride], &params->micro_B[j * params->k_full_step],&params->C[(m+i)*params->n_size + n + j]);
//						cnt2++;
						}
					}
				}
			}
		}
//	printf("1 %d %d\n",cnt,cnt2);
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
#define CYCLE_COUNT 10
#endif // _DEBUG

void measure(char *name,CMMFunc func,CThreadRoutine thread,FCalcParams *params)
	{
	int i, err = 0;
	double rv = -1.0;
	THREAD_ID_TYPE thid = 0;

	double avg = 0;
	for (i = 0; i < CYCLE_COUNT; i++)
		{
		reset_params(params);

		if (thread)
			thid = start_thread(thread,params,3);		
		int64_t t1 = get_nanotime();
		func(params);
		int64_t t2 = get_nanotime();
		if (thread)
			stop_thread(params,thid);
		double res = (double)params->m_size * params->n_size * params->k_size * 2.0/(double)(t2 - t1);
		avg += res;
		if (res > rv)
			rv = res;
		err += check_result(params->C,params->n_size,params->m_size);
		}
	avg /= (double)CYCLE_COUNT;
	printf("%15s   %7.3f (%7.3f) %d errors\n",name,avg,rv,err);
	}

int main(void)
	{
	maximizePriority();
	set_thread_processor(2);

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
		measure("HT2x48 sym",half_2x48,ThreadProc_2x48_sym,params);
		measure("HT2x48 asym",half_2x48,ThreadProc_2x48_asym,params);
		}

	free_params(params);
	delete_matrix(A);
	delete_matrix(B);
	delete_matrix(C);

//	getchar();
	return 0;
	}