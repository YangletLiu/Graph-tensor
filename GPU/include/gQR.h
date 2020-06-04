#ifndef GQR_H
#define GQR_H
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
 * @param: f 图张量 m*n*k
 * OUTPUT:
 * @param: U S V
*/

void AQR(float* d_A, int m, int n, float* d_Q, float* d_R);
void batch_qr(float* d_t, const int m, const int n, const int batch,float* d_tau);
void gQR_3D_batched_d(float* d_U, int k, float* d_A, int m, int n, float* d_R);
void gQR_3D_based_d(float* d_U, int k, float* d_A, int m, int n, float* d_Q, float* d_R);

#endif
