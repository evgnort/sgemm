#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <immintrin.h>

#include "port.h"
#include "proc_spec.h"
#include "matrixes.h"


FMatrix *alloc_matrix(unsigned width,unsigned height)
	{
	FMatrix *rv = (FMatrix *)malloc(sizeof(FMatrix));
	if (!rv || !(rv->data = malloc(sizeof(matrixtype_t) * width * height)))
		return NULL;
	rv->width = width, rv->height = height;
	return rv;
	}

void delete_matrix(FMatrix *m)
	{
	free(m->data);
	free(m);
	}

void fill_by_zeroes(matrixtype_t *data,unsigned width,unsigned height)
	{
	memset(data,0,sizeof(matrixtype_t) * width * height);
	}

void fill_by_ones(matrixtype_t *data,unsigned width,unsigned height)
	{
	unsigned i,j;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			data[i * width + j] = 1.0;
	}

int check_result(matrixtype_t *data,unsigned width,unsigned height)
	{
	unsigned i,j;
	uint64_t resval = (uint64_t)width * height * height;
	uint64_t sum = 0;
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			sum += (uint64_t)data[i * width + j];
	return (sum == resval) ? 0 : 1;
	}

