#ifndef _MATRIXES_H
#define _MATRIXES_H

typedef float matrixtype_t;

typedef struct FMatrixTg
	{
	unsigned width;
	unsigned height;
	matrixtype_t *data;
	} FMatrix;

FMatrix *alloc_matrix(unsigned width,unsigned height);

void delete_matrix(FMatrix *m);
void fill_by_zeroes(matrixtype_t *data,unsigned width,unsigned height);
void fill_by_ones(matrixtype_t *data,unsigned width,unsigned height);
int check_result(matrixtype_t *data,unsigned width,unsigned height);

#endif