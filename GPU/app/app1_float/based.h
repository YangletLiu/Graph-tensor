#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <assert.h>
#include <iostream>
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
void printMatrix(const float* A,int m, int n, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*m];
            printf("%s(%d,%d) = %f\t", name, row+1, col+1, Areg);
        }
        printf("\n");
    }
    printf("\n");
}
void printMatrix_d(const float*d_A,int m, int n, const char* name)
{
	float *A=new float[m*n];
	cudaMemcpy(A,d_A,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	printMatrix(A,m,n,name);
	delete[] A;
}

void printTensor(const float*T, int m, int n, int k, const char* name)
{
	for(int tubal = 0 ; tubal < k ; tubal++){
		printf("%s(:,:,%d) \n", name, tubal+1);
		for(int row = 0 ; row < m ; row++){
			for(int col = 0 ; col < n ; col++){
				float Areg = T[row + col*m + tubal*m*n];
				printf("%s(%d,%d) = %f\t", name, row+1, col+1, Areg);
			}
			printf("\n");
		}
		printf("\n");
	}

}
void printTensor_d(const float*d_T, int m, int n, int k, const char* name)
{
	float *T=new float[m*n*k];
	cudaMemcpy(T,d_T,sizeof(float)*m*n*k,cudaMemcpyDeviceToHost);
	printTensor(T,m,n,k, name);
	delete[] T;

}

void SVD(float* d_t ,const int m, const int n, float* d_u, float* d_s, float* d_v){
        //batch_svd
		cusolverDnHandle_t handle;
        gesvdjInfo_t params;
        int* inf = (int*)malloc(sizeof(int));
        int* d_inf = NULL;
        int echo = 1;
        int lda = m;
        int ldu = m;
        int ldv = n;
        int lwork = 0;
        float* work=NULL;

        cudaMalloc((void**)&d_inf,sizeof(int));

        if(cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS){
                fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);

        }

        if(cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS){
                fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info srror",__FUNCTION__,__LINE__);

        }

        if(cusolverDnSgesvdj_bufferSize(
        				handle,
                        CUSOLVER_EIG_MODE_VECTOR,
                        echo,
                        m,
                        n,
                        d_t,
                        m,
                        d_s,
                        d_u,
                        ldu,
                        d_v,
                        ldv,
                        &lwork,
                        params) != CUSOLVER_STATUS_SUCCESS){
                fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);

        }

        if(cudaDeviceSynchronize() != cudaSuccess){
                fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
                return;
        }

        cudaMalloc((void**)&work,sizeof(float)*lwork);

         if(cusolverDnSgesvdj(
        		   	   	 handle,
                         CUSOLVER_EIG_MODE_VECTOR,
                         echo,
                         m,
                         n,
                         d_t,
                         lda,
                         d_s,
                         d_u,
                         ldu,
                         d_v,
                         ldv,
                         work,
                         lwork,
                         d_inf,
                         params) != CUSOLVER_STATUS_SUCCESS){
                 fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);

            }
         cudaDeviceSynchronize();
         cudaMemcpy(&inf, d_inf, sizeof(int), cudaMemcpyDeviceToHost);
         if(cudaDeviceSynchronize() != cudaSuccess){
                 fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
                 return;
         }
         /*  free resources  */
         if(cusolverDnDestroy(handle)!=CUSOLVER_STATUS_SUCCESS){
                 fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);

         }

         if(cusolverDnDestroyGesvdjInfo(params)!=CUSOLVER_STATUS_SUCCESS){
                 fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);

         }

         if(work != NULL){
        	 cudaFree(work);
        	 work = NULL;
         }
         if(d_inf != NULL){
			 cudaFree(d_inf);
			 d_inf = NULL;
         }
 }

__global__ void transSliceToTubal(const float* d_A,int n1, int n2,int n3, float* d_B){
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        int a = tid/n3;
        int b = tid%n3;
        int len = n1*n2*n3;
        if(tid < len){
        	d_B[tid] = d_A[a+b*n1*n2];
        }
        __syncthreads();
}
void transSliceToTubal_d(const float* d_A,int n1, int n2,int n3, float* d_B){
    int threads;
    int blocks;
    int num= n1*n2*n3;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    transSliceToTubal<<<blocks,threads>>>(d_A,n1,n2,n3,d_B);

}

__global__ void transTubalToSlice(const float* d_A, float* d_B, int n1, int n2,int n3){
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        int a = tid/(n1*n2);
        int b = tid%(n1*n2);
        int len = n1*n2*n3;
        if(tid < len){
        	d_B[tid] = d_A[a+b*n3];
        }
        __syncthreads();
}
void transTubalToSlice_d(const float* d_A, float* d_B, int n1, int n2,int n3){
    int threads;
    int blocks;
    int num= n1*n2*n3;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    transTubalToSlice<<<blocks,threads>>>(d_A,d_B,n1,n2,n3);
}

__global__ void tensor_tranpose_XZY(float* T, int m, int n, int k, float* A){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int len = m*n*k;
    if(tid < len){
    	A[tid] = T[tid/(m*k)*m+tid%(m*k)/m*m*n+tid%(m*k)%m];
    }
    __syncthreads();
}
void tensor_tranpose_XZY_h(float* T, int m, int n, int k, float* A){

    for(int tid=0;tid < m*n*k;tid++){
    	A[tid] = T[tid/(m*k)*m+tid%(m*k)/m*m*n+tid%(m*k)%m];
    }
}
void tensor_tranpose_XZY_d(float* T, int n1, int n2, int n3, float* A){
    int threads;
    int blocks;
    int num= n1*n2*n3;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    tensor_tranpose_XZY<<<blocks,threads>>>(T,n1,n2,n3,A);
}

__global__ void frontal_slice_transpose(float* A,const int m,const int n,const int batch,float* T){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int z = tid/(m*n);
	int y = (tid-z*(m*n))/m;
	int x = (tid-z*(m*n))%m;
	//int t_n=blockDim.x*gridDim.x;
	if(tid<m*n*batch){
		T[z*m*n+x*n+y]=A[z*m*n+y*m+x];
		//tid+=t_n;
	}
	__syncthreads();
}
void frontal_slice_transpose_d(float* A,const int m,const int n,const int batch,float* T){
    int threads;
    int blocks;
    int num= m*n*batch;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    frontal_slice_transpose<<<blocks,threads>>>(A,m,n,batch,T);
}
void frontal_slice_transpose_h(float *f,  int m, int n, int k,float *ft){
	int t = 0;
	for(int i=0; i<m*n*k; i++){
		t = i/(m*n)*(m*n)+(i%(m*n))/m+i%m*n;
		ft[t]=f[i];
	}
}
void matrixMultiplytensor_batched(const float *d_A, int flag_T, int m, int n, float *d_B, int k, int batches,float *d_C){
    cublasHandle_t handle;
 	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
 	{
 	    fprintf(stdout, "CUBLAS initialization failed!\n");
 	    exit(EXIT_FAILURE);
 	}

 	const float alpha=1.0, beta=0.0;
 	int strideA=0;
 	int strideB=k*n;
 	int strideC=m*k;

     cublasOperation_t transa;
     int lda;
     if(flag_T==1){
    	 transa=CUBLAS_OP_T;
    	 lda=n;
     }else{
    	 transa=CUBLAS_OP_N;
    	 lda=m;
     }

     cublasSgemmStridedBatched(handle, transa, CUBLAS_OP_N, m, k, n, &alpha, d_A, lda, strideA, d_B, n, strideB, &beta, d_C, m, strideC, batches);
     cublasDestroy(handle);
 }

void matrixMultiply(float *A, int flag, int m, int n, float *B, int k, float *C){
 	float *d_A = NULL;
 	float *d_B = NULL;
 	float *d_C = NULL;
 	cudaMalloc((void**)&d_A, sizeof(float)*m*n);
 	cudaMalloc((void**)&d_B, sizeof(float)*n*k);
 	cudaMalloc((void**)&d_C, sizeof(float)*m*k);

 	cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m);
 	cublasSetMatrix(n, k, sizeof(float), B, n, d_B, n);

    float alpha = 1.0;
    float beta = 0.0;

 	cublasHandle_t handle;
    cublasCreate(&handle);

    if(flag==1)
    	cublasSgemm(handle,
     		CUBLAS_OP_T,//转置 flag表示是否转置
     		CUBLAS_OP_N,
     		m,
     		k,
     		n,
     		&alpha,
     		d_A,
     		n,
     		d_B,
     		n,
     		&beta,
     		d_C,
     		m);
    else
    	cublasSgemm(handle,
    	     		CUBLAS_OP_N,//转置
    	     		CUBLAS_OP_N,
    	     		m,
    	     		k,
    	     		n,
    	     		&alpha,
    	     		d_A,
    	     		m,
    	     		d_B,
    	     		n,
    	     		&beta,
    	     		d_C,
    	     		m);
     cublasGetMatrix(m, k, sizeof(float), d_C, m, C, m);

     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     cublasDestroy(handle);

 }
void matrixMultiply_d(float *d_A, int flag, int m, int n, float *d_B, int k, float *d_C){
/* 	float *d_A = NULL;
 	float *d_B = NULL;
 	float *d_C = NULL;
 	cudaMalloc((void**)&d_A, sizeof(float)*m*n);
 	cudaMalloc((void**)&d_B, sizeof(float)*n*k);
 	cudaMalloc((void**)&d_C, sizeof(float)*m*k);

 	cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m);
 	cublasSetMatrix(n, k, sizeof(float), B, n, d_B, n);*/
    float alpha = 1.0;
    float beta = 0.0;

 	cublasHandle_t handle;
    cublasCreate(&handle);

    if(flag==1)
    	cublasSgemm(handle,
     		CUBLAS_OP_T,//转置 flag表示是否转置
     		CUBLAS_OP_N,
     		m,
     		k,
     		n,
     		&alpha,
     		d_A,
     		n,
     		d_B,
     		n,
     		&beta,
     		d_C,
     		m);
    else
    	cublasSgemm(handle,
    	     		CUBLAS_OP_N,//转置
    	     		CUBLAS_OP_N,
    	     		m,
    	     		k,
    	     		n,
    	     		&alpha,
    	     		d_A,
    	     		m,
    	     		d_B,
    	     		n,
    	     		&beta,
    	     		d_C,
    	     		m);
/*     cublasGetMatrix(m, k, sizeof(float), d_C, m, C, m);

     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);*/
     cublasDestroy(handle);
 }
void tensorMultiplytensor_batched(float *A, int flag, int Am, int An, float *B, int Bn, int k, float *C) {
     cublasHandle_t handle;
     float alpha = 1;
     float beta = 0;
     int Bm = An;
     int strA = Am*An;
     int strB = Bm*Bn;
     int strC = Am*Bn;
     int batchCount = k;

     float *d_A;
     float *d_B, *d_C;
     cublasCreate(&handle);
     cudaMalloc ((void**)&d_A, sizeof(float) * Am*An*k);
     cudaMalloc ((void**)&d_B, sizeof(float) * Bm*Bn*k);
     cudaMalloc ((void**)&d_C, sizeof(float) * Am*Bn*k);

     cudaMemcpy(d_A, A, sizeof(float) * Am*An*k, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, sizeof(float) * Bm*Bn*k, cudaMemcpyHostToDevice);
     if(flag==1)
    	 cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, An, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);
     else
    	 cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);

     cudaMemcpy(C, d_C, sizeof(float) * Am*Bn*k, cudaMemcpyDeviceToHost);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     cublasDestroy(handle);
 }
void tensorMultiplytensor_d(float *d_A, int flag, int Am, int An, float *d_B, int Bn, int k, float *d_C) {
     cublasHandle_t handle;
     float alpha = 1;
     float beta = 0;
     int Bm = An;
     int strA = Am*An;
     int strB = Bm*Bn;
     int strC = Am*Bn;
     int batchCount = k;
     cublasCreate(&handle);

/*     float *d_A;
     float *d_B, *d_C;
     cublasCreate(&handle);
     cudaMalloc ((void**)&d_A, sizeof(float) * Am*An*k);
     cudaMalloc ((void**)&d_B, sizeof(float) * Bm*Bn*k);
     cudaMalloc ((void**)&d_C, sizeof(float) * Am*Bn*k);

     cudaMemcpy(d_A, A, sizeof(float) * Am*An*k, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, sizeof(float) * Bm*Bn*k, cudaMemcpyHostToDevice);*/
     if(flag==1)
    	 cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, An, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);
     else
    	 cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);

/*     cudaMemcpy(C, d_C, sizeof(float) * Am*Bn*k, cudaMemcpyDeviceToHost);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);*/
     cublasDestroy(handle);
 }

void gFFT(const float *d_U, int node, float* d_timeGSignal,int m,int n,float *d_fourierGSignal){

	int len =  m*n*node;
	int flag_U_T = 1;
	float * d_fourierGSignal_tubal=NULL;
	float * d_timeGSignal_tubal=NULL;
	cudaMalloc (( void **)& d_fourierGSignal_tubal , sizeof (float)*len);
	cudaMalloc (( void **)& d_timeGSignal_tubal , sizeof (float)*len);

	int threads = 0;
	int blocks = 0;
	if(len < 512){
	    threads = len;
	    blocks = 1;
	}else{
	    threads = 512;
	    blocks =  ((len%512) == 0)?len/512:len/512+1;
	}

	transSliceToTubal<<<blocks,threads>>>(d_timeGSignal,m, n,node,d_timeGSignal_tubal);
	matrixMultiplytensor_batched(d_U, flag_U_T, node, node,d_timeGSignal_tubal, m,n, d_fourierGSignal_tubal);
	transTubalToSlice<<<blocks,threads>>>(d_fourierGSignal_tubal, d_fourierGSignal, m, n,node);
	////////////////////////////////////////////////////////////////
	//readTensor_dev(d_fourierGSignal, m,n,node, "fourierGSignal");
	////////////////////////////////////////////////////////////////
	if (d_fourierGSignal_tubal) cudaFree(d_fourierGSignal_tubal);
	if (d_timeGSignal_tubal) cudaFree(d_timeGSignal_tubal);
}
void gIFFT(const float *d_U, int node,float* d_fourierGSignal, int m, int n, float *d_timeGSignal){

	int len =  m*n*node;
	int flag_U_T =0;
	float * d_fourierGSignal_tubal=NULL;
	float * d_timeGSignal_tubal=NULL;
	cudaMalloc (( void **)& d_fourierGSignal_tubal , sizeof (float)*len);
	cudaMalloc (( void **)& d_timeGSignal_tubal , sizeof (float)*len);

	int threads = 0;
	int blocks = 0;
	if(len < 512){
	    threads = len;
	    blocks = 1;
	}else{
	    threads = 512;
	    blocks =  ((len%512) == 0)?len/512:len/512+1;
	}
	transSliceToTubal<<<blocks,threads>>>(d_fourierGSignal,m, n,node,d_fourierGSignal_tubal);
	matrixMultiplytensor_batched(d_U, flag_U_T ,node, node,d_fourierGSignal_tubal, m,n, d_timeGSignal_tubal);
	transTubalToSlice<<<blocks,threads>>>(d_timeGSignal_tubal, d_timeGSignal, m, n,node);

	if (d_fourierGSignal_tubal) cudaFree(d_fourierGSignal_tubal);
	if (d_timeGSignal_tubal) cudaFree(d_timeGSignal_tubal);

}

void QR(float *left,float *res,float *right,int zuo,int rank){

	float *d_left;
	cudaMalloc((void**)&d_left,sizeof(float)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(float)*zuo*rank,cudaMemcpyHostToDevice);
	float *d_right;
	cudaMalloc((void**)&d_right,sizeof(float)*zuo);
	cudaMemcpy(d_right,right,sizeof(float)*zuo,cudaMemcpyHostToDevice);

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);

//	int info_gpu=0;
	int *devInfo = NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	float *d_work = NULL;
	int lwork = 0;
	float *d_tau = NULL;
	cudaMalloc((void**)&d_tau,sizeof(float)*zuo);

	cusolverDnSgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(float)*lwork);
	cusolverDnSgeqrf(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			d_tau,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
//	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//	cout<<info_gpu<<endl;

	//step2 compute Q^T*right

	cusolverDnSormqr(
			cusolverH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_OP_T,
			zuo,1,rank,
			d_left,zuo,
			d_tau,
			d_right,zuo,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
//	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//	cout<<info_gpu<<endl;

	// step3 solve R*x = Q^T*b
	float one = 1;
	cublasStrsm(
			cublasH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			rank,1,
			&one,
			d_left,zuo,
			d_right,rank
			);
	cudaDeviceSynchronize();
	cudaMemcpy(res,d_right,sizeof(float)*rank,cudaMemcpyDeviceToHost);

	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}

void QR_d(float *d_left,float *res,float *d_right,int zuo,int rank){
#if 0
	float *d_left;
	cudaMalloc((void**)&d_left,sizeof(float)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(float)*zuo*rank,cudaMemcpyHostToDevice);
	float *d_right;
	cudaMalloc((void**)&d_right,sizeof(float)*zuo);
	cudaMemcpy(d_right,right,sizeof(float)*zuo,cudaMemcpyHostToDevice);
#endif
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);

	int info_gpu=0;
	int *devInfo = NULL;
	Check(cudaMalloc((void**)&devInfo,sizeof(int)));
	float *d_work = NULL;
	int lwork = 0;
	float *d_tau = NULL;
	Check(cudaMalloc((void**)&d_tau,sizeof(float)*rank));

	cusolverDnSgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			&lwork
			);
	Check(cudaMalloc((void**)&d_work,sizeof(float)*lwork));
	cusolverDnSgeqrf(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			d_tau,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
	Check(cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
//	cout<<info_gpu<<endl;

	//step2 compute Q^T*right

	cusolverDnSormqr(
			cusolverH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_OP_T,
			zuo,1,rank,
			d_left,zuo,
			d_tau,
			d_right,zuo,
			d_work,
			lwork,
			devInfo
			);
	cudaDeviceSynchronize();
//	cudaMemcpy(&info_gpu,devInfo,sizeof(int),cudaMemcpyDeviceToHost);
//	cout<<info_gpu<<endl;

	// step3 solve R*x = Q^T*b
	float one = 1;
	cublasStrsm(
			cublasH,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			rank,1,
			&one,
			d_left,zuo,
			d_right,rank
			);
	//cudaDeviceSynchronize();
	Check(cudaMemcpy(res,d_right,sizeof(float)*rank,cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}
void gproduct(float* d_U, int k, float* d_G1, int m, int r, float* d_G2, int n, float* d_G3){
	//printTensor_d(d_G1, m, r, k, "G1");
	//printTensor_d(d_G2, r, n, k, "G2");
	float* d_G1_f = NULL;
	cudaMalloc((void**)&d_G1_f,sizeof(float)*m*r*k);
	gFFT(d_U, k, d_G1, m, r, d_G1_f);
	float* d_G2_f = NULL;
	cudaMalloc((void**)&d_G2_f,sizeof(float)*r*n*k);
	gFFT(d_U, k, d_G2, r, n, d_G2_f);
	printf("\n6******************\n");

	float* d_G3_f = NULL;
	cudaMalloc((void**)&d_G3_f,sizeof(float)*m*n*k);
	tensorMultiplytensor_d(d_G1_f, 0, m, r, d_G2_f, n, k, d_G3_f);
	printf("\n6******************\n");

	gIFFT(d_U, k, d_G3_f, m, n, d_G3);
	printf("\n6******************\n");
    cudaFree(d_G1_f);
    cudaFree(d_G2_f);
    cudaFree(d_G3_f);
}
float norm(float *a,float *b,int n){
	float sum = 0;
	for(int i = 0; i< n; i++){
		sum = sum + (a[i] - b[i])*(a[i] - b[i]);
	}
	sum = sqrt(sum);
	return sum;
}

__global__ void vec2circul(float* d_V, float* d_M, int m){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<m*m){
		d_M[(tid%m+tid/m)%m+tid/m*m]=d_V[tid%m];
	}
	__syncthreads();
}
void vec2circul_d(float* d_V, float* d_M, int m){
    int threads;
    int blocks;
    int num= m*m;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    vec2circul<<<blocks,threads>>>(d_V,d_M,m);
}
void vec2circul_h(float* V, float* M, int m){
	for(int i=0; i<m; i++){
		for(int j=0; j< m; j++){
			M[j+i*m]=V[(j+m-i)%m];
		}
	}
}
__global__ void tensor2diagmatrix(float* d_T, int m, int n, int k, float* d_M){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<m*n*k){
		d_M[tid/m*m*k+tid%m+tid/(m*n)*m]=d_T[tid];
	}
}
void tensor2diagmatrix_d(float* d_T, int m, int n, int k, float* d_M){
    int threads;
    int blocks;
    int num= m*n*k;
    if(num < 512){
        threads=num;
        blocks=1;
    }else{
        threads=512;
        blocks= (num%512 ==0)?num/512:num/512+1;
    }
    tensor2diagmatrix<<<blocks,threads>>>(d_T, m, n, k, d_M);
}
