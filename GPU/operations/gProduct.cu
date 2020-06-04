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
