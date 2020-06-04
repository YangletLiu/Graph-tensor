#include "gFilter.h"
#define PI 3.14159265

void gFilter_3D_batched_d(float* d_U, float* E, float* d_S, int m, int n, int k, float* d_F){
	//graph fourier transform
	int len = m*n*k;
	float* d_S_f = NULL;
	cudaMalloc((void **)& d_S_f, sizeof (float)*len);
	gFT_3D_batched_d(d_U, k, d_S, m, n, d_S_f);
	//compute the maximum eigenvalue of the Laplacian
	float maxe = E[0];
	for(int i=1; i<k; i++){
		if(E[i]>maxe)
			maxe = E[i];
	}

	float* Ef = (float*)malloc(sizeof(float)*k);
	for(int i=0; i<k; i++)
		Ef[i] = sin(PI/4*E[i]*(2/maxe));

	//实现bsxfun(@time, conj(fe), permute(F,[1 3 2])),得到chat
	float* chat = (float*)malloc(sizeof(float)*len);
	memset(chat,0,sizeof(float)*len);
	float* S_f = (float*)malloc(sizeof(float)*len);
	cudaMemcpy(S_f, d_S_f, sizeof(double)*len,cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
			for(int t=0; t<k; t++)
				chat[t*m*n+j*m+i] = Ef[t]*S_f[t*m*n+j*m+i];

	float* d_chat = NULL;
	cudaMalloc((void **)& d_chat, sizeof (float)*len);
	//实现gsp_igft(G,chat)
	gIFT_3D_batched_d(d_U, k, d_chat, m, n, d_F);
}

void gFilter_3D_based_d(float* d_U, float* E, float* d_S, int m, int n, int k, float* d_F){
	//graph fourier transform
	int len = m*n*k;
	float* d_S_f = NULL;
	cudaMalloc((void **)& d_S_f, sizeof (float)*len);
	gFT_3D_based_d(d_U, k, d_S, m, n, d_S_f);
	//compute the maximum eigenvalue of the Laplacian
	float maxe = E[0];
	for(int i=1; i<k; i++){
		if(E[i]>maxe)
			maxe = E[i];
	}

	float* Ef = (float*)malloc(sizeof(float)*k);
	for(int i=0; i<k; i++)
		Ef[i] = sin(PI/4*E[i]*(2/maxe));

	//实现bsxfun(@time, conj(fe), permute(F,[1 3 2])),得到chat
	float* chat = (float*)malloc(sizeof(float)*len);
	memset(chat,0,sizeof(float)*len);
	float* S_f = (float*)malloc(sizeof(float)*len);
	cudaMemcpy(S_f, d_S_f, sizeof(double)*len,cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++)
		for(int j=0; j<n; j++)
			for(int t=0; t<k; t++)
				chat[t*m*n+j*m+i] = Ef[t]*S_f[t*m*n+j*m+i];

	float* d_chat = NULL;
	cudaMalloc((void **)& d_chat, sizeof (float)*len);
	//实现gsp_igft(G,chat)
	gIFT_3D_based_d(d_U, k, d_chat, m, n, d_F);
}

void gFilter_d(float* d_U, float* E, float* d_S, int m, int k, float* d_F){
	//graph fourier transform
	int len = m*k;
	float* d_S_f = NULL;
	cudaMalloc((void **)& d_S_f, sizeof (float)*len);
	gFT_d(d_U, k, d_S, m, d_S_f);
	//compute the maximum eigenvalue of the Laplacian
	float maxe = E[0];
	for(int i=1; i<k; i++){
		if(E[i]>maxe)
			maxe = E[i];
	}

	float* Ef = (float*)malloc(sizeof(float)*k);
	for(int i=0; i<k; i++)
		Ef[i] = sin(PI/4*E[i]*(2/maxe));

	//实现bsxfun(@time, conj(fe), permute(F,[1 3 2])),得到chat
	float* chat = (float*)malloc(sizeof(float)*len);
	float* S_f = (float*)malloc(sizeof(float)*len);
	cudaMemcpy(S_f, d_S_f, sizeof(double)*len,cudaMemcpyDeviceToHost);
	for(int i=0; i<m; i++)
			for(int t=0; t<k; t++)
				chat[t*m+i] = Ef[t]*S_f[t*m+i];

	float* d_chat = NULL;
	cudaMalloc((void **)& d_chat, sizeof (float)*len);
	//实现gsp_igft(G,chat)
	//gIFT_3D_batched_d(d_U, k, d_chat, m, n, d_F);
	gIFT_d(d_U, k, d_chat, m, d_F);
}
