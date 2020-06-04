#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#define NDEBUG
#include <assert.h>
#include <string.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
using namespace std;
#include "based.h"
#include "one_step.cuh"

#define random(x) (double(rand()%x)/x)

int main(int argc, char* argv[]){
    int m = 360;
    int n = 640;
    int k = atoi(argv[1]);
    int r = 20;

    clock_t start, finish;
    start = clock();
/**************************step one: load data**************************/
    double* T = (double*)malloc(sizeof(double)*m*n*k);
    FILE *fp = NULL;
    if((fp=fopen(argv[2],"r")) == NULL){
	printf("Input file don't exist!\n");
	exit(0);
    }
    for(int i=0; i< m*n*k; i++)
	fscanf(fp, "%lf", T+i);
    //printTensor(T, m, n, k, "T");
#if 0
    //构造环状图邻接矩阵
    double* A = (double*)malloc(sizeof(double)*k*k);
    memset(A, 0, sizeof(double)*k*k);
    for(int i = 0; i < k-1; i++){
        A[i*k+i+1] = 1;
        A[(i+1)*k+i] = 1;
    }
    A[k-1] = 1;
    A[(k-1)*k] = 1;
    //printMatrix(A, k, k, "A");

    //计算标准化矩阵Ｌ＝Ｄ.^(-1/2)ＡＤ.^(-1/2)
    double* D = (double*)malloc(sizeof(double)*k*k);//D为度矩阵 此处直接计算成Ｄ.^(-1/2)
    memset(D, 0, sizeof(double)*k*k);
    for(int i = 0; i < k; i++){
        D[i*k+i] = -1/(sqrt(2));
    }
    //printMatrix(D, k, k, "D");

    double* L1 = (double*)malloc(sizeof(double)*k*k);
    double* L = (double*)malloc(sizeof(double)*k*k);
    matrixMultiply(D, 0, k ,k, A, k, L1);
    matrixMultiply(L1, 0, k ,k, D, k, L);
    //printMatrix(L, k, k, "L");
    free(L1);
    free(D);
    free(A);

    double *d_L = NULL;
    double *d_U = NULL;
    double *d_S = NULL;
    double *d_V = NULL;
    Check(cudaMalloc((void**)&d_L, sizeof(double)*k*k));
    Check(cudaMalloc((void**)&d_U, sizeof(double)*k*k));
    Check(cudaMalloc((void**)&d_S, sizeof(double)*k));
    Check(cudaMalloc((void**)&d_V, sizeof(double)*k*k));
    Check(cudaMemcpy(d_L, L, sizeof(double)*k*k, cudaMemcpyHostToDevice));
    SVD(d_L, k, k, d_U, d_S, d_V);
    cudaDeviceSynchronize();
    free(L);
    cudaFree(d_L);
    cudaFree(d_S);
    cudaFree(d_V);
#endif
#if 1
    double* U = (double*)malloc(sizeof(double)*k*k);
    FILE *fu = NULL;
    if((fu=fopen(argv[3],"r")) == NULL){
	printf("Input file don't exist!\n");
	exit(0);
    }
    for(int i=0; i< k*k; i++)
	fscanf(fu, "%lf", U+i);
    double *d_U = NULL;
    cudaMalloc((void**)&d_U, sizeof(double)*k*k);
    cudaMemcpy(d_U, U, sizeof(double)*k*k, cudaMemcpyHostToDevice);
   // printMatrix_d(d_U, k, k, "U");
#endif
/********************step two: de_noise 获得去噪后的Ｔ1和秩ｒ***********************/
    //int r = 15;

/********************step three:采样**************************/
    //int num_miss_frame = int(k/2);
    int num_miss_frame = floor(k*atoi(argv[4])/10);
    printf("num_miss: %d\t",num_miss_frame);
    //int miss_index[num_miss_frame]={1,3,6,7,10,12,9,15,14,18};
    //int miss_index[num_miss_frame]={1,3,6,7,10,12,9,14};
    int miss_index[num_miss_frame]={0};
#if 1
    srand((unsigned)time(NULL));
    int i,j,temp;
    for(i= 0; i <num_miss_frame;i++){
	temp=rand()%(k-1);
	for(j=0;j<i;j++){
	   if(miss_index[j]==temp){ 
	      i=i-1;
	      break;
	   }
        }
        if(j>=i)
	   miss_index[i]=temp;
    	//printf("%d ",miss_index[i]);
    }
    printf("miss_index: ");
    for(i=0;i<num_miss_frame;i++)
	    printf("%d ",miss_index[i]);
    //printf("\n");
#endif
  
    double* omega =  (double*)malloc(sizeof(double)*m*n*k);
    for(int i=0; i<m*n*k; i++){
    	omega[i] = 1;
    }
    for(int i = 0; i < num_miss_frame; i++){
    	memset(omega+miss_index[i]*m*n, 0, sizeof(double)*m*n);
    }
    //printTensor(omega, m, n, k, "omega");
    double* Tomega = (double*)malloc(sizeof(double)*m*n*k);
    for(int i=0; i<m*n*k; i++){
        Tomega[i] = omega[i] * T[i];
    }
    //printTensor(Tomega, m, n, k, "Tomega");

    double* d_Tomega = NULL;
    Check(cudaMalloc((void**)&d_Tomega, sizeof(double)*m*n*k));
    Check(cudaMemcpy(d_Tomega, Tomega, sizeof(double)*m*n*k, cudaMemcpyHostToDevice));

    double* d_omega = NULL;
    Check(cudaMalloc((void**)&d_omega, sizeof(double)*m*n*k));
    Check(cudaMemcpy(d_omega, omega, sizeof(double)*m*n*k, cudaMemcpyHostToDevice));

    double* d_Tomega_f = NULL;
    Check(cudaMalloc((void**)&d_Tomega_f, sizeof(double)*m*n*k));
    gFFT(d_U, k, d_Tomega, m, n, d_Tomega_f);
    cudaDeviceSynchronize();

    double* d_omega_f = NULL;
    Check(cudaMalloc((void**)&d_omega_f, sizeof(double)*m*n*k));
    gFFT(d_U, k, d_omega, m, n, d_omega_f);
    cudaDeviceSynchronize();
    //printTensor_d(d_omega_f, m, n,  k, "omega_f");
    //printTensor_d(d_Tomega_f, m, n,  k, "Tomega_f");

    double* d_omega_ft = NULL;
    Check(cudaMalloc((void**)&d_omega_ft, sizeof(double)*m*n*k));
    double* d_Tomega_ft = NULL;
    Check(cudaMalloc((void**)&d_Tomega_ft, sizeof(double)*m*n*k));

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

    double* Y = (double*)malloc(sizeof(double)*r*n*k);
#if 0
    FILE *fp1 = NULL;
    if((fp1=fopen("Y45.txt","r")) == NULL){
	printf("input file don't exist!\n");
	exit(0);
    }
    for(int i=0; i< r*n*k; i++)
	fscanf(fp1, "%lf", Y+i);
#endif
#if 1
    for(int i=0; i<r*n*k;i++)
        Y[i] = random(1000);
#endif
    double* d_Y = NULL;
    Check(cudaMalloc((void**)&d_Y, sizeof(double)*r*n*k));
    Check(cudaMemcpy(d_Y, Y, sizeof(double)*r*n*k, cudaMemcpyHostToDevice));
    //printTensor_d(d_Y, r, n, k, "Y");

    double* d_Y_f = NULL;
    Check(cudaMalloc((void**)&d_Y_f, sizeof(double)*r*n*k));
    gFFT(d_U, k, d_Y, r, n, d_Y_f);
    cudaDeviceSynchronize();
    //printTensor_d(d_Y_f, r, n, k, "Y_f");

    double* d_Y_ft = NULL;
    Check(cudaMalloc((void**)&d_Y_ft, sizeof(double)*n*r*k));
    frontal_slice_transpose_d(d_Y_f, r, n, k, d_Y_ft);
    cudaDeviceSynchronize();
    //printTensor_d(d_Y_ft, n, r, k, "Y_ft");

    double* d_X = NULL;
    Check(cudaMalloc((void**)&d_X, sizeof(double)*m*r*k));
    double* d_X_f = NULL;
    Check(cudaMalloc((void**)&d_X_f, sizeof(double)*m*r*k));
    double* d_X_ft = NULL;
    Check(cudaMalloc((void**)&d_X_ft, sizeof(double)*m*r*k));
/**************************step four: Alter-Min**************************/
    for(int iter = 0; iter < 8; iter++){
    //printf("iter %d\n", iter);
    one_step(d_Tomega_ft, d_omega_ft, d_Y_ft, d_X_ft, n, m, k, r);
    //printTensor_d(d_X_ft, r, m, k, "X_ft");
    frontal_slice_transpose_d(d_X_ft, r, m, k, d_X_f);
    //printTensor_d(d_X_f, m, r, k, "X_f");

    one_step(d_Tomega_f, d_omega_f, d_X_f, d_Y_f, m, n, k, r);
    //printTensor_d(d_Y_f, r, n, k, "Y_f");
    frontal_slice_transpose_d(d_Y_f, r, n, k, d_Y_ft);
    //printTensor_d(d_Y_ft, n, r, k, "Y_ft");
    }

     double* T_est = (double*)malloc(sizeof(double)*m*n*k);
     double* d_T_est = NULL;
     Check(cudaMalloc((void**)&d_T_est, sizeof(double)*m*n*k));

#if 0
     gIFFT(d_U, k, d_X_f, m, r, d_X);
     cudaDeviceSynchronize();
     gIFFT(d_U, k, d_Y_f, r, n, d_Y);
     cudaDeviceSynchronize();
     gproduct(d_U, k, d_X, m, r, d_Y, n, d_T_est);
     cudaDeviceSynchronize();
#endif

#if 1
     double* d_T_fest = NULL;
     Check(cudaMalloc((void**)&d_T_fest, sizeof(double)*m*n*k));
     tensorMultiplytensor_d(d_X_f, 0, m, r, d_Y_f, n, k, d_T_fest);
     //printTensor_d(d_T_fest, m, n, k, "Tf");
     //printTensor_d(d_X_f, m, r, k, "X");
     //printTensor_d(d_Y_f, r, n, k, "Y");
     gIFFT(d_U, k, d_T_fest, m, n, d_T_est);
     cudaDeviceSynchronize();
     cudaFree(d_T_fest);
#endif

     //printTensor_d(d_T_est, m, n, k, "T");
     Check(cudaMemcpy(T_est,d_T_est,sizeof(double)*m*n*k,cudaMemcpyDeviceToHost));
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

     finish = clock();
     double time = (double)(finish-start) / CLOCKS_PER_SEC;
/**************************step five:误差计算**************************/
     for(int i=0; i<m*n*k; i++){
    	 T_est[i]=fabs(T_est[i]);
    	 T[i]=T[i]/255;
     }
     double sum_T=0;
     double sum_T_est=0;
     for(int i=0; i<m*n*k; i++){
    	 sum_T+=T[i];
    	 sum_T_est+=T_est[i];
     }
     for(int i=0; i<m*n*k; i++){
    	 T_est[i]=T_est[i]*(sum_T/sum_T_est);
     }
    double* zero = (double*)malloc(sizeof(double)*m*n*k);
    memset(zero, 0, sizeof(double)*m*n*k);
    double norm1 = norm(T, T_est, m*n*k);
    double norm2 = norm(T, zero, m*n*k);
    //printf("%lf ",norm1);
    //printf("%lf ",norm2);
    double res = norm1/norm2;
    printf("\tFrame: %d\tError: %lf\t Time: %lf\n", k, res, time);

    fclose(fp);
    free(zero);
    return 0;
}
