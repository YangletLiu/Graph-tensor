#ifndef GSHIFT_H
#define GSHIFT_H
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
//#define NDEBUG
#include <assert.h>
#include "based.h"
/* gShift
 * INPUT:
 * @param: T 图的邻接矩阵 k*k
 * @param: f 图信号 k*m
 * OUTPUT:
 * @param: F 变换后的图信号
*/
void gShift_d(float* d_T, int k, float* d_f, int m, float* d_F);

/* gShift_3D
 * INPUT:
 * @param: T 图的邻接矩阵矩阵 k*k
 * @param: f 图信号 m*n*k
 * OUTPUT:
 * @param: F 变换后的图信号
*/
void gShift_3D_batched_d(float* d_T, int k, float* d_f, int m, int n, float* d_F);
void gShift_3D_based_d(float* d_T, int k, float* d_f, int m, int n, float* d_F);
#endif
