#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <fstream>
using namespace std;
#include "based.h"
#include "one_step.cuh"
//#include "gproduct.cuh"
//#define NDEBUG
#include <assert.h>
#define random(x) (float(rand()%x)/x)

int main(int argc, char* argv[]){
    int m = 60;
    int n = 80;
    int k = 30;
    int r = 15;

/**************************step one: load data**************************/
    float* T = (float*)malloc(sizeof(float)*m*n*k);
    FILE *fp = NULL;
	if((fp=fopen("T.txt","r")) == NULL){
		printf("Input file don't exist!\n");
		exit(0);
	}
	for(int i=0; i< m*n*k; i++)
		fscanf(fp, "%f", T+i);
	//printTensor(T, m, n, k, "T");
#if 0
    //构造环状图邻接矩阵
    float* A = (float*)malloc(sizeof(float)*k*k);
    memset(A, 0, sizeof(float)*k*k);
    for(int i = 0; i < k-1; i++){
        A[i*k+i+1] = 1;
        A[(i+1)*k+i] = 1;
    }
    A[k-1] = 1;
    A[(k-1)*k] = 1;
    //printMatrix(A, k, k, "A");

    //计算标准化矩阵Ｌ＝Ｄ.^(-1/2)ＡＤ.^(-1/2)
    float* D = (float*)malloc(sizeof(float)*k*k);//D为度矩阵 此处直接计算成Ｄ.^(-1/2)
    memset(D, 0, sizeof(float)*k*k);
    for(int i = 0; i < k; i++){
        D[i*k+i] = -1/(sqrt(2));
    }
    //printMatrix(D, k, k, "D");

    float* L1 = (float*)malloc(sizeof(float)*k*k);
    float* L = (float*)malloc(sizeof(float)*k*k);
    matrixMultiply(D, 0, k ,k, A, k, L1);
    matrixMultiply(L1, 0, k ,k, D, k, L);
    //printMatrix(L, k, k, "L");
    free(L1);
    free(D);
    free(A);

	float *d_L = NULL;
	float *d_U = NULL;
	float *d_S = NULL;
	float *d_V = NULL;
	Check(cudaMalloc((void**)&d_L, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_U, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_S, sizeof(float)*k));
	Check(cudaMalloc((void**)&d_V, sizeof(float)*k*k));
	Check(cudaMemcpy(d_L, L, sizeof(float)*k*k, cudaMemcpyHostToDevice));
	SVD(d_L, k, k, d_U, d_S, d_V);
	cudaDeviceSynchronize();
	free(L);
	cudaFree(d_L);
	cudaFree(d_S);
	cudaFree(d_V);
#endif
#if 1
	float* U = (float*)malloc(sizeof(float)*k*k);
    FILE *fu = NULL;
	if((fu=fopen("U30.txt","r")) == NULL){
		printf("Input file don't exist!\n");
		exit(0);
	}
	for(int i=0; i< k*k; i++)
		fscanf(fu, "%f", U+i);
	float *d_U = NULL;
	cudaMalloc((void**)&d_U, sizeof(float)*k*k);
	cudaMemcpy(d_U, U, sizeof(float)*k*k, cudaMemcpyHostToDevice);
	//printMatrix_d(d_U, k, k, "U");
#endif
/********************step two: de_noise 获得去噪后的Ｔ1和秩ｒ***********************/
    //int r = 15;

/********************step three:采样**************************/
    int num_miss_frame = 1;
    int miss_index[num_miss_frame];
#if 0
    srand(time(0));
    for(int i= 0; i <num_miss_frame; i++){
    	miss_index[i]=rand()%k;
    	printf("%d ",miss_index[i]);
    }
    printf("\n");
#endif
    miss_index[0] = 2;
    float* omega =  (float*)malloc(sizeof(float)*m*n*k);
    for(int i=0; i<m*n*k; i++){
    	omega[i] = 1;
    }
    for(int i = 0; i < num_miss_frame; i++){
    	memset(omega+miss_index[i]*m*n, 0, sizeof(float)*m*n);
    }
    //printTensor(omega, m, n, k, "omega");
    float* Tomega = (float*)malloc(sizeof(float)*m*n*k);
    for(int i=0; i<m*n*k; i++){
        Tomega[i] = omega[i] * T[i];
    }
    //printTensor(Tomega, m, n, k, "Tomega");

    float* d_Tomega = NULL;
    Check(cudaMalloc((void**)&d_Tomega, sizeof(float)*m*n*k));
    Check(cudaMemcpy(d_Tomega, Tomega, sizeof(float)*m*n*k, cudaMemcpyHostToDevice));

    float* d_omega = NULL;
    Check(cudaMalloc((void**)&d_omega, sizeof(float)*m*n*k));
    Check(cudaMemcpy(d_omega, omega, sizeof(float)*m*n*k, cudaMemcpyHostToDevice));

    float* d_Tomega_f = NULL;
    Check(cudaMalloc((void**)&d_Tomega_f, sizeof(float)*m*n*k));
    gFFT(d_U, k, d_Tomega, m, n, d_Tomega_f);
    cudaDeviceSynchronize();

    float* d_omega_f = NULL;
    Check(cudaMalloc((void**)&d_omega_f, sizeof(float)*m*n*k));
    gFFT(d_U, k, d_omega, m, n, d_omega_f);
    cudaDeviceSynchronize();
    //printTensor_d(d_omega_f, m, n,  k, "omega_f");
    //printTensor_d(d_Tomega_f, m, n,  k, "Tomega_f");

    float* d_omega_ft = NULL;
    Check(cudaMalloc((void**)&d_omega_ft, sizeof(float)*m*n*k));
    float* d_Tomega_ft = NULL;
    Check(cudaMalloc((void**)&d_Tomega_ft, sizeof(float)*m*n*k));

    frontal_slice_transpose_d(d_Tomega_f, m, n, k, d_Tomega_ft);
    cudaDeviceSynchronize();
    frontal_slice_transpose_d(d_omega_f,  m, n, k, d_omega_ft);
    cudaDeviceSynchronize();
    free(omega);
    free(Tomega);
    cudaFree(d_omega);
    cudaFree(d_Tomega);
    //printTensor_d(d_Tomega_ft, n, m, k, "d_Tomega_ft");
    //printTensor_d(d_omega_ft, n, m, k, "d_omega_ft");

    float* Y = (float*)malloc(sizeof(float)*r*n*k);
#if 0
    FILE *fp1 = NULL;
	if((fp1=fopen("Y585.txt","r")) == NULL){
		printf("input file don't exist!\n");
		exit(0);
	}
	for(int i=0; i< r*n*k; i++)
		fscanf(fp1, "%f", Y+i);
#endif
#if 1
    for(int i=0; i<r*n*k;i++)
        Y[i] = random(10);
#endif
    float* d_Y = NULL;
    Check(cudaMalloc((void**)&d_Y, sizeof(float)*r*n*k));
    Check(cudaMemcpy(d_Y, Y, sizeof(float)*r*n*k, cudaMemcpyHostToDevice));
	//printTensor_d(d_Y, r, n, k, "Y");

    float* d_Y_f = NULL;
    Check(cudaMalloc((void**)&d_Y_f, sizeof(float)*r*n*k));
	gFFT(d_U, k, d_Y, r, n, d_Y_f);
	cudaDeviceSynchronize();
	//printTensor_d(d_Y_f, r, n, k, "Y_f");

    float* d_Y_ft = NULL;
    Check(cudaMalloc((void**)&d_Y_ft, sizeof(float)*n*r*k));
	frontal_slice_transpose_d(d_Y_f, r, n, k, d_Y_ft);
	cudaDeviceSynchronize();
	//printTensor_d(d_Y_ft, n, r, k, "Y_ft");

    float* d_X = NULL;
    Check(cudaMalloc((void**)&d_X, sizeof(float)*m*r*k));
    float* d_X_f = NULL;
    Check(cudaMalloc((void**)&d_X_f, sizeof(float)*m*r*k));
    float* d_X_ft = NULL;
    Check(cudaMalloc((void**)&d_X_ft, sizeof(float)*m*r*k));
/**************************step four: Alter-Min**************************/
    for(int iter = 0; iter < 10; iter++){
    	printf("iter %d\n", iter);
        one_step(d_Tomega_ft, d_omega_ft, d_Y_ft, d_X_ft, n, m, k, r);
        //printTensor_d(d_X_ft, r, m, k, "X_ft");
        frontal_slice_transpose_d(d_X_ft, r, m, k, d_X_f);
       // printTensor_d(d_X_f, m, r, k, "X_f");

        one_step(d_Tomega_f, d_omega_f, d_X_f, d_Y_f, m, n, k, r);
        //printTensor_d(d_Y_f, r, n, k, "Y_f");
        frontal_slice_transpose_d(d_Y_f, r, n, k, d_Y_ft);
        //printTensor_d(d_Y_ft, n, r, k, "Y_ft");
     }

     float* T_est = (float*)malloc(sizeof(float)*m*n*k);
     float* d_T_est = NULL;
     Check(cudaMalloc((void**)&d_T_est, sizeof(float)*m*n*k));

#if 0
     gIFFT(d_U, k, d_X_f, m, r, d_X);
     cudaDeviceSynchronize();
     gIFFT(d_U, k, d_Y_f, r, n, d_Y);
     cudaDeviceSynchronize();
     gproduct(d_U, k, d_X, m, r, d_Y, n, d_T_est);
     cudaDeviceSynchronize();
#endif

#if 1
     float* d_T_fest = NULL;
     Check(cudaMalloc((void**)&d_T_fest, sizeof(float)*m*n*k));
     tensorMultiplytensor_d(d_X_f, 0, m, r, d_Y_f, n, k, d_T_fest);
     //printTensor_d(d_T_fest, m, n, k, "Tf");
    // printTensor_d(d_X_f, m, r, k, "X");
     //printTensor_d(d_Y_f, r, n, k, "Y");
     gIFFT(d_U, k, d_T_fest, m, n, d_T_est);
 	 cudaDeviceSynchronize();
     cudaFree(d_T_fest);
#endif

     //printTensor_d(d_T_est, m, n, k, "T");
     Check(cudaMemcpy(T_est,d_T_est,sizeof(float)*m*n*k,cudaMemcpyDeviceToHost));
     cudaFree(d_T_est);

     free(Y);
     cudaFree(d_Y);
     cudaFree(d_Y_f);
     cudaFree(d_Y_ft);

     cudaFree(d_X);
     cudaFree(d_X_f);
     cudaFree(d_X_ft);

     cudaFree(d_omega_f);
     cudaFree(d_omega_ft);
     cudaFree(d_Tomega_f);
     cudaFree(d_Tomega_ft);

/**************************step five:误差计算**************************/
     for(int i=0; i<m*n*k; i++){
    	 T_est[i]=fabs(T_est[i]);
    	 T[i]=T[i]/255;
     }
     float sum_T=0;
     float sum_T_est=0;
     for(int i=0; i<m*n*k; i++){
    	 sum_T+=T[i];
    	 sum_T_est+=T_est[i];
     }
     for(int i=0; i<m*n*k; i++){
    	 T_est[i]=T_est[i]*(sum_T/sum_T_est);
     }
    float* zero = (float*)malloc(sizeof(float)*m*n*k);
    memset(zero, 0, sizeof(float)*m*n*k);
    float norm1 = norm(T, T_est, m*n*k);
    float norm2 = norm(T, zero, m*n*k);
    printf("%f ",norm1);
    printf("%f ",norm2);
	float res = norm1/norm2;
 	printf("Error: %f \n", res);

 	free(zero);
    fclose(fp);

    return 0;
}
