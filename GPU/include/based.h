#ifndef BASED_H
#define BASED_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <assert.h>
#include <iostream>
#include "magma_lapack.h"
#include "magma_v2.h"
#include "kblas.h"
using namespace std;
//#define NDEBUG
#include <assert.h>

#define Check(call)														\
{																		\
	cudaError_t status = call;											\
	if (status != cudaSuccess)											\
	{																	\
		cout << "行号:" << __LINE__ << endl;							    \
		cout << "错误:" << cudaGetErrorString(status) << endl;			\
	}																	\
}

#define Check_CUBLAS(call)												\
{																		\
	cublasStatus_t status = call;									    \
	if (status != CUBLAS_STATUS_SUCCESS)								\
	{																	\
		cout << "行号:" << __LINE__ << endl;							    \
        printf ("Error in CUBLAS\n");                                   \
	}																	\
}

void printMatrix(const float* A, int m, int n, const char* name);
void printMatrix_d(const float* d_A, int m, int n, const char* name);

void printTensor(const float* T, int m, int n, int k, const char* name);
void printTensor_d(const float* d_T, int m, int n, int k, const char* name);

/* SVD
 * INPUT:
 * @param: 矩阵T m*n
 * OUTPUT:
 * @param: U V S T=USV
*/
void SVD(float* d_t ,const int m, const int n, float* d_u, float* d_s, float* d_v);

/* QR
 * INPUT:
 * @param: 矩阵left zuo*rank
 * @param: 向量right zuo*1
 * OUTPUT:
 * @param: 向量 res rank*1
*/
void QR(float *left,float *res,float *right,int zuo,int rank);
void QR_d(float *d_left,float *res,float *d_right,int zuo,int rank);

/*
 * INPUT:d_A
 * OUTPUT
 * @d_V: d_V contains the orthonormal eigenvectors of the matrix A
 * @d_W:The eigenvalue values of A. a real array of dimension m.
 */
void Eig(float* d_A,const int m,float* d_V,float* d_W);

/* transSliceToTubal
 * INPUT:
 * @param: 张量 n1*n2*3
 * OUTPUT:
 * @param: 张量 n2*n1*n3
*/
__global__ void transSliceToTubal(const float* d_A,int n1, int n2,int n3, float* d_B);
void transSliceToTubal_d(const float* d_A,int n1, int n2,int n3, float* d_B);

/* transTubalToSlice
 * INPUT:
 * @param: 张量
 * OUTPUT:
 * @param: 张量 n1*n2*n3
*/
__global__ void transTubalToSlice(const float* d_A, float* d_B, int n1, int n2,int n3);
void transTubalToSlice_d(const float* d_A, float* d_B, int n1, int n2,int n3);

/* tensor_tranpose_XZY
 * INPUT:
 * @param: 张量 m*n*k
 * OUTPUT:
 * @param: 张量 m*k*n
*/
__global__ void tensor_tranpose_XZY(float* T, int m, int n, int k, float* A);
void tensor_tranpose_XZY_h(float* T, int m, int n, int k, float* A);
void tensor_tranpose_XZY_d(float* T, int n1, int n2, int n3, float* A);

/* frontal_slice_transpose
 * INPUT:
 * @param: 张量 m*n*k
 * OUTPUT:
 * @param: 张量 n*m*k
*/
__global__ void frontal_slice_transpose(float* A,const int m,const int n,const int batch,float* T);
void frontal_slice_transpose_d(float* A,const int m,const int n,const int batch,float* T);
void frontal_slice_transpose_h(float *f,  int m, int n, int k,float *ft);

/* matrixMultiplytensor_batch
 * INPUT:
 * @param: 矩阵A m*n
 * @param: 张量B n*k*batch
 * @param:  batch的次数
 * @param: flag 1表示A进行转置; 0表示A不做变换
 * OUTPUT:
 * @param: 矩阵C n*k*batch C为矩阵A乘张量B的每一个frontal_slice
*/
void matrixMultiplytensor_d(float *d_A, int flag_T, int m, int n, float *d_B, int k, int batches,float *d_C);

/* matrixMultiply
 * INPUT:
 * @param: 矩阵A m*n
 * @param: 矩阵B n*k
 * @param: flag 1表示A进行转置
 * OUTPUT:
 * @param: 矩阵C C=A*B m*k
*/
void matrixMultiply_h(float *A, int flag, int m, int n, float *B, int k, float *C);
void matrixMultiply_d(float *d_A, int flag, int m, int n, float *d_B, int k, float *d_C);

/* matrixMultiplytensor
 * INPUT:
 * @param: 矩阵A Am*An
 * @param: 张量B An*Bn*k
 * @param: k batch的次数
 * @param: flag 1表示A进行转置; 0表示A不做变换
 * OUTPUT:
 * @param: 矩阵C Am*Bn*k C为矩阵A乘张量B的每一个frontal_slice
*/
void tensorMultiplytensor_batched(float *A, int flag, int Am, int An, float *B, int Bn, int k, float *C);
void tensorMultiplytensor_d(float *d_A, int flag, int Am, int An, float *d_B, int Bn, int k, float *d_C);

/* vec2circul
 * INPUT:
 * @param: 向量 m*1
 * OUTPUT:
 * @param: 循环矩阵 m*m
*/
__global__ void vec2circul(float* d_V, float* d_M, int m);
void vec2circul_d(float* d_V, float* d_M, int m);
void vec2circul_h(float* V, float* M, int m);

/* tensor2diagmatrix
 * INPUT:
 * @param: 张量 m*n*k
 * OUTPUT:
 * @param: 对角矩阵 k*m*k*r
*/
__global__ void tensor2diagmatrix(float* d_T, int m, int n, int k, float* d_M);
void tensor2diagmatrix_d(float* d_T, int m, int n, int k, float* d_M);

void gproduct(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3);
float norm(float *a,float *b,int n);
#endif
