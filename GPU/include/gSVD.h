#ifndef GSVD_H
#define GSVD_H
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "based.h"
#include "gFT.h"
#include "gIFT.h"
//#include "kblas.h"


/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: f 图张量 m*n*k
 * OUTPUT:
 * @param: U S V
*/
void gSVD_3D_batched_d(float* d_LU, int k, float* d_A, int m, int n, float* d_U, float* d_S, float* d_V);
void gSVD_3D_based_d(float* d_LU, int k, float* d_A, int m, int n, float* d_U, float* d_S, float* d_V);

#endif
