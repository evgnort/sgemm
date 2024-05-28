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

void fill_by_pattern(matrixtype_t *data,unsigned width,unsigned height)
   {
   unsigned i,j;
   for (i = 0; i < height; i++)
      {
      int val = 1;
      for (j = 0; j < width; j++)
         {
         data[i * width + j] = (float)val;
         val = val % 3 + 1;
         }
      }
   }

int check_result(matrixtype_t *data,unsigned width,unsigned height,unsigned k)
   {
   unsigned i,j;
   unsigned adds[3] = {0,1,3};

   float expected = (float)(6 * (k / 3) + adds[k % 3]);

   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
         if ((unsigned)data[i * width + j] != expected)
            {
#ifdef _DEBUG
#ifdef _WIN32
            __debugbreak();
#else
            __asm__ volatile("int $0x03");
#endif
#endif
            return 1;
            }
   return 0;
   }

