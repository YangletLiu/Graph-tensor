#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "gConvolution.h"
#include <iostream>
using namespace std;
#include <random>
using std::default_random_engine;
using std::uniform_real_distribution;
#define random(x) (float(rand()%x))/x
void inital_graph(float* adjacent, float* laplacian, int n, float sparsity){
	default_random_engine e(time(NULL));
    uniform_real_distribution<float> u(0, 1); //随机数分布对象
    for(int i =0; i< n; i++){
        for(int j = (i+1); j<n; j++){
            float random = u(e);
            if( random < sparsity){
            		float temp=random(1000);
                    adjacent[i+j*n] = temp;
                    adjacent[j+i*n] = temp;
             }
             else{
                    adjacent[i+j*n] = 0;
                    adjacent[j+i*n] = 0;
              }
         }
    }
    float* D = (float*)malloc(sizeof(float)*n*n);
    memset(D, 0, sizeof(float)*n*n);
    float* DL = (float*)malloc(sizeof(float)*n*n);
    memset(DL, 0, sizeof(float)*n*n);
    for(int i =0; i< n; i++){
    	int num=0;
        for(int j = 0; j<n; j++){
        	if(adjacent[i*n+j]!=0)
        		D[i*n+i]=++num;
        }
        if(D[i*n+i]!=0)
        	D[i*n+i]=1/sqrt(D[i*n+i]);
    }
    matrixMultiply_h(D, 0, n, n, adjacent, n, DL);
    matrixMultiply_h(DL, 0, n, n, D, n, laplacian);
    for(int i=0;i<n;i++){
    	for(int j=0;j<n;j++){
    		laplacian[j*n+i]=-laplacian[j*n+i];
    	}
    	laplacian[i*n+i]=1+laplacian[i*n+i];
    }
    free(D);
    free(DL);
}

int main(int argc, char* argv[]){
	clock_t start, finish;
//	start = clock();
#if 1
	FILE *fp = NULL;
	if((fp=fopen(argv[1],"r")) == NULL){
		printf("input file don't exist!\n");
		exit(0);
	}
	const int k = atoi(argv[2]);//图的节点数
	const int d = atoi(argv[3]);//图的每个节点的数据维度
	const int m = atoi(argv[5]);
	const int n = atoi(argv[6]);

	float* L = (float*)malloc(sizeof(float)*k*k);//L 图的拉普拉斯矩阵
	for(int i=0; i< k*k; i++)
		fscanf(fp, "%f", L+i);

	float* f = (float*)malloc(sizeof(float)*m*n*k);//f 为图信号
	for(int i=0; i< m*n*k; i++)
		f[i] = random(1000);

        start = clock();
	float *d_f = NULL;
	Check(cudaMalloc((void**)&d_f, sizeof(float)*m*n*k));
	Check(cudaMemcpy(d_f, f, sizeof(float)*m*n*k, cudaMemcpyHostToDevice));

	float *d_F = NULL;
	Check(cudaMalloc((void**)&d_F, sizeof(float)*m*n*k));//F 为变换后的信号

	float *d_L = NULL;
	float *d_U = NULL;
	float *d_S = NULL;
	float *d_V = NULL;
	Check(cudaMalloc((void**)&d_L, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_U, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_S, sizeof(float)*k));
	//Check(cudaMalloc((void**)&d_V, sizeof(float)*k*k));
	Check(cudaMemcpy(d_L, L, sizeof(float)*k*k, cudaMemcpyHostToDevice));
	//SVD(d_L, k, k, d_U, d_S, d_V);
	Eig(d_L, k, d_U, d_S);
	cudaDeviceSynchronize();
	float* S = (float*)malloc(sizeof(float)*k);
	Check(cudaMemcpy(S, d_S, sizeof(float)*k, cudaMemcpyDeviceToHost));

	float* H =(float*)malloc(sizeof(float)*k*k);
	memset(H, 0, sizeof(float)*k*k);
	float alpha=0;
	for(int i=0; i< k; i++){
		alpha=random(1000);
		H[i*k+i] = alpha+alpha*S[i];
	}
	float *d_H = NULL;
	Check(cudaMalloc((void**)&d_H, sizeof(float)*k*k));
	Check(cudaMemcpy(d_H, H, sizeof(float)*k*k, cudaMemcpyHostToDevice));

	if(0 == strcmp("batched", argv[4])){
		switch(d){
		case 1:
			//gConvolution_3D_batched_d(d_U, k, d_H, d_f, m, n, d_F);
			break;
		case 2:
			//gConvolution_3D_batched_d(d_U, k, d_H, d_f, m, n, d_F);
			break;
		case 3:
			gConvolution_3D_batched_d(d_U, k, d_H, d_f, m, n, d_F);
			break;
		default:
			break;
		}
	}
	else if(0 == strcmp("based",argv[4])){
		switch(d){
		case 1:
			gConvolution_d(d_U, k, d_H, d_f, m, d_F);
			break;
		case 2:
			gConvolution_d(d_U, k, d_H, d_f, m, d_F);
			break;
		case 3:
			gConvolution_3D_based_d(d_U, k, d_H, d_f, m, n, d_F);
			break;
		default:
			break;
		}
	}
	else{
		printf("Input error: based or batched\n");
		return 0;
	}
	finish = clock();
	double time = (double)(finish-start) / CLOCKS_PER_SEC;
	switch(d){
	case 1:
		printf("gConvolution_%dD_%s of G[%d*%d] time: %lfs\n", d, argv[4], k, m, time);
		break;
	case 2:
		printf("gConvolution_%dD_%s of G[%d*%d] time: %lfs\n", d, argv[4], k, m, time);
		break;
	case 3:
		printf("gConvolution_%dD_%s of G[%d*%d*%d] time: %lfs\n", d, argv[4], m, n, k, time);
		break;
	default:
		break;
	}

    free(L);
    cudaFree(d_L);
    cudaFree(d_U);
    //cudaFree(d_V);
    cudaFree(d_S);
    free(S);
    free(f);
    cudaFree(d_f);
    cudaFree(d_F);
    free(H);
    cudaFree(d_H);
    fclose(fp);
#endif
#if 0
	const int k = atoi(argv[1]);//图的节点数
	const int m = 1;
	const int n = 1;

	float* L = (float*)malloc(sizeof(float)*k*k);//L 图的拉普拉斯矩阵
    float sparsity = 0.4;
    float* W = (float*)malloc(sizeof(float)*k*k);
    inital_graph(W, L, k, sparsity);

	float* f = (float*)malloc(sizeof(float)*m*n*k);//f 为图信号
	for(int i=0; i< m*n*k; i++)
		f[i] = random(1000);
	float *d_f = NULL;
	Check(cudaMalloc((void**)&d_f, sizeof(float)*m*n*k));
	Check(cudaMemcpy(d_f, f, sizeof(float)*m*n*k, cudaMemcpyHostToDevice));

	float *d_F = NULL;
	Check(cudaMalloc((void**)&d_F, sizeof(float)*m*n*k));//F 为变换后的信号

	float *d_L = NULL;
	float *d_U = NULL;
	float *d_S = NULL;
	float *d_V = NULL;
	Check(cudaMalloc((void**)&d_L, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_U, sizeof(float)*k*k));
	Check(cudaMalloc((void**)&d_S, sizeof(float)*k));
	//Check(cudaMalloc((void**)&d_V, sizeof(float)*k*k));
	Check(cudaMemcpy(d_L, L, sizeof(float)*k*k, cudaMemcpyHostToDevice));
	//SVD(d_L, k, k, d_U, d_S, d_V);
	Eig(d_L, k, d_U, d_S);
	cudaDeviceSynchronize();
	float* S = (float*)malloc(sizeof(float)*k);
	Check(cudaMemcpy(S, d_S, sizeof(float)*k, cudaMemcpyDeviceToHost));

	float* H =(float*)malloc(sizeof(float)*k*k);
	memset(H, 0, sizeof(float)*k*k);
	float alpha=0;
	for(int i=0; i< k; i++){
		alpha=random(1000);
		H[i*k+i] = alpha+alpha*S[i];
	}
	float *d_H = NULL;
	Check(cudaMalloc((void**)&d_H, sizeof(float)*k*k));
	Check(cudaMemcpy(d_H, H, sizeof(float)*k*k, cudaMemcpyHostToDevice));

	gConvolution_d(d_U, k, d_H, d_f, m, d_F);

	finish = clock();
	double time = (double)(finish-start) / CLOCKS_PER_SEC;
	printf("gConvolution_1D of G[%d*%d] time: %lfs\n", k, m, time);

    free(L);
    free(W);
    cudaFree(d_L);
    cudaFree(d_U);
    //cudaFree(d_V);
    cudaFree(d_S);
    free(S);
    free(f);
    cudaFree(d_f);
    cudaFree(d_F);
    free(H);
    cudaFree(d_H);
#endif
	return 0;
}
