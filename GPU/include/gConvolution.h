#ifndef GCONVOLUTION_H
#define GCONVOLUTION_H
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
 * @param: H 卷积核 k*k
 * @param: f 图信号 k*m
 * OUTPUT:
 * @param: F 卷积后的图信号
*/
void gConvolution_d(float* d_U, int k, float* d_H, float* d_f, int m, float* d_F);

/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: H 卷积核 k*k
 * @param: f 图信号 m*n*k
 * OUTPUT:
 * @param: F 卷积后的图信号
*/
void gConvolution_3D_batched_d(float* d_U, int k, float* d_H, float* d_f, int m, int n, float* d_F);
void gConvolution_3D_based_d(float* d_U, int k, float* d_H, float* d_f, int m, int n, float* d_F);

#endif
