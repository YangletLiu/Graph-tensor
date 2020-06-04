/*#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "based.h"
#include "gFT.h"
#include "gIFT.h"
*/
#include "gProduct.h"

/*
 * INPUT:
 * @param: G1 m*r*k
 * @param: G2 r*n*k
 * OUTPUT:
 * @param: G3 m*n*k
*/

void gProduct_3D_batched(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3){
	//printTensor_d(d_G1, m, r, k, "G1");
	//printTensor_d(d_G2, r, n, k, "G2");
	float* d_G1_f = NULL;
	cudaMalloc((void**)&d_G1_f,sizeof(float)*m*r*k);
	gFT_3D_batched_d(d_U, k, d_G1, m, r, d_G1_f);
	
	float* d_G2_f = NULL;
	cudaMalloc((void**)&d_G2_f,sizeof(float)*r*n*k);
	gFT_3D_batched_d(d_U, k, d_G2, r, n, d_G2_f);

	float* d_G3_f = NULL;
	cudaMalloc((void**)&d_G3_f,sizeof(float)*m*n*k);
	tensorMultiplytensor_d(d_G1_f, 0, m, r, d_G2_f, n, k, d_G3_f);

	gIFT_3D_batched_d(d_U, k, d_G3_f, m, n, d_G3);

    cudaFree(d_G1_f);
    cudaFree(d_G2_f);
    cudaFree(d_G3_f);
}

void gProduct_3D_based(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3){
	//printTensor_d(d_G1, m, r, k, "G1");
	//printTensor_d(d_G2, r, n, k, "G2");
	float* d_G1_f = NULL;
	cudaMalloc((void**)&d_G1_f,sizeof(float)*m*r*k);
	gFT_3D_based_d(d_U, k, d_G1, m, r, d_G1_f);
	float* d_G2_f = NULL;
	cudaMalloc((void**)&d_G2_f,sizeof(float)*r*n*k);
	gFT_3D_based_d(d_U, k, d_G2, r, n, d_G2_f);

	float* d_G3_f = NULL;
	cudaMalloc((void**)&d_G3_f,sizeof(float)*m*n*k);
	//tensorMultiplytensor_d(d_G1_f, 0, m, r, d_G2_f, n, k, d_G3_f);
	cublasHandle_t handle;
	int Am = m;
	int An = r;
	int Bn = n;
	int Bm = r;
	int strA = Am*An;
	int strB = Bm*Bn;
	int strC = Am*Bn;
	cublasCreate(&handle);
	float alpha = 1;
	float beta = 0;
	for(int i=0; i<k; i++){
	if(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm,
	        &alpha, d_G1_f+i*strA, Am,d_G2_f+i*strB, Bm,  &beta,
	        d_G3_f+i*strC, Am) !=CUBLAS_STATUS_SUCCESS){
	
		fprintf(stdout,"[%s]:[%d] CUFFT ERROR: cublasCgemm failed!",__FUNCTION__,__LINE__);
		return;
	}
	}
	cublasDestroy(handle);

	gIFT_3D_based_d(d_U, k, d_G3_f, m, n, d_G3);

    cudaFree(d_G1_f);
    cudaFree(d_G2_f);
    cudaFree(d_G3_f);
}

void gProduct_3D(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3){
	float* d_G2_f = NULL;
	cudaMalloc((void**)&d_G2_f,sizeof(float)*r*n*k);
	gFT_3D_batched_d(d_U, k, d_G2, r, n, d_G2_f);
	tensorMultiplytensor_d(d_G1, 0, m, r, d_G2_f, n, k, d_G3);
	cudaFree(d_G2_f);
}
