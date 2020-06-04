#include "gShift.h"

void gShift_d(float* d_T, int k, float* d_f, int m, float* d_F){
    matrixMultiply_d(d_f, 0, m, k, d_T, k, d_F);
}

void gShift_3D_batched_d(float* d_T, int k, float* d_f, int m, int n, float* d_F){
	int len =  m*n*k;
	int flag_U_T = 0;
	float * d_F_trans=NULL;
	float * d_f_trans=NULL;
	cudaMalloc (( void **)& d_F_trans , sizeof (float)*len);
	cudaMalloc (( void **)& d_f_trans , sizeof (float)*len);

	transSliceToTubal_d(d_f,m, n,k,d_f_trans);
	matrixMultiplytensor_d(d_T, flag_U_T, k, k,d_f_trans, m,n, d_F_trans);
	transTubalToSlice_d(d_F_trans, d_F, m, n,k);

	if (d_F_trans) cudaFree(d_F_trans);
	if (d_f_trans) cudaFree(d_f_trans);
}

void gShift_3D_based_d(float* d_T, int k, float* d_f, int m, int n, float* d_F){
	int len =  m*n*k;
	int flag_U_T = 0;
	float * d_F_trans=NULL;
	float * d_f_trans=NULL;
	cudaMalloc (( void **)& d_F_trans , sizeof (float)*len);
	cudaMalloc (( void **)& d_f_trans , sizeof (float)*len);

	transSliceToTubal_d(d_f,m, n, k, d_f_trans);
	for(int i=0; i<n; i++){
		matrixMultiply_d(d_T, flag_U_T, k, k, d_f_trans+i*k*m, m, d_F_trans+i*k*m);
	}
	transTubalToSlice_d(d_F_trans, d_F, m, n, k);

	if (d_F_trans) cudaFree(d_F_trans);
	if (d_f_trans) cudaFree(d_f_trans);
}
