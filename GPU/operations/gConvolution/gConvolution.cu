#include "gConvolution.h"

void gConvolution_d(float* d_U, int k, float* d_H, float* d_f, int m, float* d_F){
/*	float* d_HU = NULL;
	float* d_HS = NULL;
	float* d_HV = NULL;
	cudaMalloc (( void **)& d_HU, sizeof (float)*k*k);
	cudaMalloc (( void **)& d_HS, sizeof (float)*k);
	cudaMalloc (( void **)& d_HV, sizeof (float)*k*k);
	SVD(d_H, k, k, d_HU, d_HS, d_HV);*/

	float* d_F1 = NULL;
	float* d_F2 = NULL;
	cudaMalloc (( void **)& d_F1, sizeof (float)*k*m);
	cudaMalloc (( void **)& d_F2, sizeof (float)*k*k);
	gFT_d(d_U, k, d_f, m, d_F1);
	gFT_d(d_U, k, d_H, k, d_F2);

	float* d_F3 = NULL;
	cudaMalloc (( void **)& d_F3, sizeof (float)*k*m);
	matrixMultiply_d(d_F2, 0, k, k, d_F1, m, d_F3);
	gIFT_d(d_U, k, d_F3, m, d_F);

	//cudaFree(d_HU);
	//cudaFree(d_HV);
	//cudaFree(d_HS);
	cudaFree(d_F1);
	cudaFree(d_F2);
	cudaFree(d_F3);
}

void gConvolution_3D_batched_d(float* d_U, int k, float* d_H, float* d_f, int m, int n, float* d_F){
/*	float* d_HU = NULL;
	float* d_HS = NULL;
	float* d_HV = NULL;
	cudaMalloc (( void **)& d_HU, sizeof (float)*k*k);
	cudaMalloc (( void **)& d_HS, sizeof (float)*k);
	cudaMalloc (( void **)& d_HV, sizeof (float)*k*k);
	SVD(d_H, k, k, d_HU, d_HS, d_HV);*/

	float* d_F1 = NULL;
	float* d_F2 = NULL;
	cudaMalloc (( void **)& d_F1, sizeof (float)*k*m*n);
	cudaMalloc (( void **)& d_F2, sizeof (float)*k*k);
	gFT_d(d_U, k, d_H, k, d_F2);
	gFT_3D_batched_d(d_U, k, d_f, m, n, d_F1);


	float* d_F3 = NULL;
	cudaMalloc (( void **)& d_F3, sizeof (float)*k*m*n);
	matrixMultiplytensor_d(d_F2, 0, k, k, d_F1, m, n, d_F3);
	//gFT_3D_batched_d(d_F2, k, d_F1, m, n, d_F3);

	gIFT_3D_batched_d(d_U, k, d_F3, m, n, d_F);

	//cudaFree(d_HU);
	//cudaFree(d_HV);
	//cudaFree(d_HS);
	cudaFree(d_F1);
	cudaFree(d_F2);
	cudaFree(d_F3);
}

void gConvolution_3D_based_d(float* d_U, int k, float* d_H, float* d_f, int m, int n, float* d_F){

	/*float* d_HU = NULL;
	float* d_HS = NULL;
	float* d_HV = NULL;
	cudaMalloc (( void **)& d_HU, sizeof (float)*k*k);
	cudaMalloc (( void **)& d_HS, sizeof (float)*k);
	cudaMalloc (( void **)& d_HV, sizeof (float)*k*k);
	SVD(d_H, k, k, d_HU, d_HS, d_HV);*/

	float* d_F1 = NULL;
	float* d_F2 = NULL;
	cudaMalloc (( void **)& d_F1, sizeof (float)*k*m*n);
	cudaMalloc (( void **)& d_F2, sizeof (float)*k*k);
	gFT_d(d_U, k, d_H, k, d_F2);
	gFT_3D_based_d(d_U, k, d_f, m, n, d_F1);


	float* d_F3 = NULL;
	cudaMalloc (( void **)& d_F3, sizeof (float)*k*m*n);
	gFT_3D_based_d(d_F2, k, d_F1, m, n, d_F3);

	gIFT_3D_based_d(d_U, k, d_F3, m, n, d_F);

	//cudaFree(d_HU);
	//cudaFree(d_HV);
	//cudaFree(d_HS);
	cudaFree(d_F1);
	cudaFree(d_F2);
	cudaFree(d_F3);
}
