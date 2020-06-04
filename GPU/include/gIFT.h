#ifndef GIFT_H
#define GIFT_H
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
//#define NDEBUG
#include <assert.h>
#include "based.h"
/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: f 傅里叶变换后的图信号 k*m m可以为1，此时为标量
 * OUTPUT:
 * @param: F 图信号
*/
void gIFT_d(float* d_U, int k, float* d_f, int m, float* d_F);

/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: f 傅里叶变换后的图信号 m*n*k
 * OUTPUT:
 * @param: F 逆傅里叶变换后的图信号
*/
void gIFT_3D_batched_d(float *d_U, int k, float* d_f, int m, int n, float *d_F);
void gIFT_3D_based_d(float *d_U, int k, float* d_f, int m, int n, float *d_F);
#endif
