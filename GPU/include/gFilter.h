#ifndef GFILTER_H
#define GFLLTER_H
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
 * @param: E 图的拉普拉斯矩阵特征值
 * @param: S 图信号 m*n*k
 * OUTPUT:
 * @param: F 滤波后的图信号
*/
void gFilter_3D_batched_d(float* d_U, float* E, float* d_S, int m, int n, int k, float* d_F);
void gFilter_3D_based_d(float* d_U, float* E, float* d_S, int m, int n, int k, float* d_F);
void gFilter_d(float* d_U, float* E, float* d_S, int m, int k, float* d_F);
#endif
