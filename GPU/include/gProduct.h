#ifndef GPRODUCT_H
#define GPRODUCT_H
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "based.h"
#include "gFT.h"
#include "gIFT.h"
/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: G1 m*r*k 
 * @param: G2 r*n*k
 * OUTPUT:
 * @param: G3 
*/

void gProduct_3D_batched(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3);
void gProduct_3D_based(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3);
void gProduct_3D(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3);
#endif
