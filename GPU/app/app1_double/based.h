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
void printMatrix(const double* A,int m, int n, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*m];
            printf("%s(%d,%d) = %12.4E\t", name, row+1, col+1, Areg);
        }
        printf("\n");
    }
    printf("\n");
}
void printMatrix_d(const double*d_A,int m, int n, const char* name)
{
	double *A=new double[m*n];
	cudaMemcpy(A,d_A,sizeof(double)*m*n,cudaMemcpyDeviceToHost);
	printMatrix(A,m,n,name);
	delete[] A;
}

void printTensor(const double*T, int m, int n, int k, const char* name)
{
	for(int tubal = 0 ; tubal < k ; tubal++){
		printf("%s(:,:,%d) \n", name, tubal+1);
		for(int row = 0 ; row < m ; row++){
			for(int col = 0 ; col < n ; col++){
				double Areg = T[row + col*m + tubal*m*n];
				printf("%s(%d,%d) = %12.4E\t", name, row+1, col+1, Areg);
			}
			printf("\n");
		}
		printf("\n");
	}

}
void printTensor_d(const double*d_T, int m, int n, int k, const char* name)
{
	double *T=new double[m*n*k];
	cudaMemcpy(T,d_T,sizeof(double)*m*n*k,cudaMemcpyDeviceToHost);
	printTensor(T,m,n,k, name);
	delete[] T;

}

void SVD(double* d_t ,const int m, const int n, double* d_u, double* d_s, double* d_v){
		cusolverDnHandle_t handle;
        gesvdjInfo_t params;
        int* inf = (int*)malloc(sizeof(int));
        int* d_inf = NULL;
        int echo = 1;
        int lda = m;
        int ldu = m;
        int ldv = n;
        int lwork = 0;
        double* work=NULL;

        cudaMalloc((void**)&d_inf,sizeof(int));

        if(cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS){
                fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);

        }

        if(cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS){
                fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info srror",__FUNCTION__,__LINE__);

        }

        if(cusolverDnDgesvdj_bufferSize(
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

        cudaMalloc((void**)&work,sizeof(double)*lwork);

         if(cusolverDnDgesvdj(
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

__global__ void transSliceToTubal(const double* d_A,int n1, int n2,int n3, double* d_B){
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        int a = tid/n3;
        int b = tid%n3;
        int len = n1*n2*n3;
        if(tid < len){
        	d_B[tid] = d_A[a+b*n1*n2];
        }
        __syncthreads();
}
void transSliceToTubal_d(const double* d_A,int n1, int n2,int n3, double* d_B){
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

__global__ void transTubalToSlice(const double* d_A, double* d_B, int n1, int n2,int n3){
        int tid = threadIdx.x+blockIdx.x*blockDim.x;
        //int c = tid/(n1*n2);
       // int b = tid%(n1*n2)/n1;
	//int a = tid%n1;
	int a= tid/(n1*n2);
	int b= tid%(n1*n2);
        int len = n1*n2*n3;
        if(tid < len){
        	d_B[tid] = d_A[a+b*n3];
        }
        __syncthreads();
}
void transTubalToSlice_d(const double* d_A, double* d_B, int n1, int n2,int n3){
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

__global__ void tensor_tranpose_XZY(double* T, int m, int n, int k, double* A){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    int len = m*n*k;
    if(tid < len){
    	A[tid] = T[tid/(m*k)*m+tid%(m*k)/m*m*n+tid%(m*k)%m];
    }
    __syncthreads();
}
void tensor_tranpose_XZY_h(double* T, int m, int n, int k, double* A){
    for(int tid=0;tid < m*n*k;tid++){
    	A[tid] = T[tid/(m*k)*m+tid%(m*k)/m*m*n+tid%(m*k)%m];
    }
}
void tensor_tranpose_XZY_d(double* T, int n1, int n2, int n3, double* A){
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

__global__ void frontal_slice_transpose(double* A,const int m,const int n,const int batch,double* T){
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
void frontal_slice_transpose_d(double* A,const int m,const int n,const int batch,double* T){
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
void frontal_slice_transpose_h(double *f,  int m, int n, int k,double *ft){
	int t = 0;
	for(int i=0; i<m*n*k; i++){
		t = i/(m*n)*(m*n)+(i%(m*n))/m+i%m*n;
		ft[t]=f[i];
	}
}
void matrixMultiplytensor_batched(const double *d_A, int flag_T, int m, int n, double *d_B, int k, int batches,double *d_C){
    cublasHandle_t handle;
 	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
 	{
 	    fprintf(stdout, "CUBLAS initialization failed!\n");
 	    exit(EXIT_FAILURE);
 	}

 	const double alpha=1.0, beta=0.0;
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

     cublasDgemmStridedBatched(handle, transa, CUBLAS_OP_N, m, k, n, &alpha, d_A, lda, strideA, d_B, n, strideB, &beta, d_C, m, strideC, batches);
     cublasDestroy(handle);
 }

void matrixMultiply(double *A, int flag, int m, int n, double *B, int k, double *C){
 	double *d_A = NULL;
 	double *d_B = NULL;
 	double *d_C = NULL;
 	cudaMalloc((void**)&d_A, sizeof(double)*m*n);
 	cudaMalloc((void**)&d_B, sizeof(double)*n*k);
 	cudaMalloc((void**)&d_C, sizeof(double)*m*k);

 	cublasSetMatrix(m, n, sizeof(double), A, m, d_A, m);
 	cublasSetMatrix(n, k, sizeof(double), B, n, d_B, n);

    double alpha = 1.0;
    double beta = 0.0;

 	cublasHandle_t handle;
    cublasCreate(&handle);

    if(flag==1)
    	cublasDgemm(handle,
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
    	cublasDgemm(handle,
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
     cublasGetMatrix(m, k, sizeof(double), d_C, m, C, m);

     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     cublasDestroy(handle);

 }
void matrixMultiply_d(double *d_A, int flag, int m, int n, double *d_B, int k, double *d_C){
/* 	double *d_A = NULL;
 	double *d_B = NULL;
 	double *d_C = NULL;
 	cudaMalloc((void**)&d_A, sizeof(double)*m*n);
 	cudaMalloc((void**)&d_B, sizeof(double)*n*k);
 	cudaMalloc((void**)&d_C, sizeof(double)*m*k);

 	cublasSetMatrix(m, n, sizeof(double), A, m, d_A, m);
 	cublasSetMatrix(n, k, sizeof(double), B, n, d_B, n);*/
    double alpha = 1.0;
    double beta = 0.0;

 	cublasHandle_t handle;
    cublasCreate(&handle);

    if(flag==1)
    	cublasDgemm(handle,
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
    	cublasDgemm(handle,
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
/*     cublasGetMatrix(m, k, sizeof(double), d_C, m, C, m);

     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);*/
     cublasDestroy(handle);
 }
void tensorMultiplytensor_batched(double *A, int flag, int Am, int An, double *B, int Bn, int k, double *C) {
     cublasHandle_t handle;
     double alpha = 1;
     double beta = 0;
     int Bm = An;
     int strA = Am*An;
     int strB = Bm*Bn;
     int strC = Am*Bn;
     int batchCount = k;

     double *d_A;
     double *d_B, *d_C;
     cublasCreate(&handle);
     cudaMalloc ((void**)&d_A, sizeof(double) * Am*An*k);
     cudaMalloc ((void**)&d_B, sizeof(double) * Bm*Bn*k);
     cudaMalloc ((void**)&d_C, sizeof(double) * Am*Bn*k);

     cudaMemcpy(d_A, A, sizeof(double) * Am*An*k, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, sizeof(double) * Bm*Bn*k, cudaMemcpyHostToDevice);
     if(flag==1)
    	 cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, An, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);
     else
    	 cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);

     cudaMemcpy(C, d_C, sizeof(double) * Am*Bn*k, cudaMemcpyDeviceToHost);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     cublasDestroy(handle);
 }
void tensorMultiplytensor_d(double *d_A, int flag, int Am, int An, double *d_B, int Bn, int k, double *d_C) {
     cublasHandle_t handle;
     double alpha = 1;
     double beta = 0;
     int Bm = An;
     int strA = Am*An;
     int strB = Bm*Bn;
     int strC = Am*Bn;
     int batchCount = k;
     cublasCreate(&handle);

/*     double *d_A;
     double *d_B, *d_C;
     cublasCreate(&handle);
     cudaMalloc ((void**)&d_A, sizeof(double) * Am*An*k);
     cudaMalloc ((void**)&d_B, sizeof(double) * Bm*Bn*k);
     cudaMalloc ((void**)&d_C, sizeof(double) * Am*Bn*k);

     cudaMemcpy(d_A, A, sizeof(double) * Am*An*k, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, sizeof(double) * Bm*Bn*k, cudaMemcpyHostToDevice);*/
     if(flag==1)
    	 cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, An, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);
     else
    	 cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, Am, Bn, Bm, &alpha, d_A, Am, strA, d_B, Bm, strB, &beta, d_C, Am, strC, batchCount);

/*     cudaMemcpy(C, d_C, sizeof(double) * Am*Bn*k, cudaMemcpyDeviceToHost);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);*/
     cublasDestroy(handle);
 }

void gFFT(const double *d_U, int node, double* d_timeGSignal,int m,int n,double *d_fourierGSignal){
	//printTensor_d(d_timeGSignal,m,n,node,"s");
	int len =  m*n*node;
	int flag_U_T = 1;
	double * d_fourierGSignal_tubal=NULL;
	double * d_timeGSignal_tubal=NULL;
	cudaMalloc (( void **)& d_fourierGSignal_tubal , sizeof (double)*len);
	cudaMalloc (( void **)& d_timeGSignal_tubal , sizeof (double)*len);


	transSliceToTubal_d(d_timeGSignal,m, n,node,d_timeGSignal_tubal);
	//printTensor_d(d_timeGSignal_tubal,node,m,n,"s");
	matrixMultiplytensor_batched(d_U, flag_U_T, node, node,d_timeGSignal_tubal, m,n, d_fourierGSignal_tubal);
	transTubalToSlice_d(d_fourierGSignal_tubal, d_fourierGSignal, m, n,node);
	////////////////////////////////////////////////////////////////
	//readTensor_dev(d_fourierGSignal, m,n,node, "fourierGSignal");
	////////////////////////////////////////////////////////////////
	if (d_fourierGSignal_tubal) cudaFree(d_fourierGSignal_tubal);
	if (d_timeGSignal_tubal) cudaFree(d_timeGSignal_tubal);
}
void gIFFT(const double *d_U, int node,double* d_fourierGSignal, int m, int n, double *d_timeGSignal){

	int len =  m*n*node;
	int flag_U_T =0;
	double * d_fourierGSignal_tubal=NULL;
	double * d_timeGSignal_tubal=NULL;
	cudaMalloc (( void **)& d_fourierGSignal_tubal , sizeof (double)*len);
	cudaMalloc (( void **)& d_timeGSignal_tubal , sizeof (double)*len);
	transSliceToTubal_d(d_fourierGSignal,m, n,node,d_fourierGSignal_tubal);
	matrixMultiplytensor_batched(d_U, flag_U_T ,node, node,d_fourierGSignal_tubal, m,n, d_timeGSignal_tubal);
	transTubalToSlice_d(d_timeGSignal_tubal, d_timeGSignal, m, n,node);

	if (d_fourierGSignal_tubal) cudaFree(d_fourierGSignal_tubal);
	if (d_timeGSignal_tubal) cudaFree(d_timeGSignal_tubal);

}

void QR(double *left,double *res,double *right,int zuo,int rank){

	double *d_left;
	cudaMalloc((void**)&d_left,sizeof(double)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(double)*zuo*rank,cudaMemcpyHostToDevice);
	double *d_right;
	cudaMalloc((void**)&d_right,sizeof(double)*zuo);
	cudaMemcpy(d_right,right,sizeof(double)*zuo,cudaMemcpyHostToDevice);

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);

//	int info_gpu=0;
	int *devInfo = NULL;
	cudaMalloc((void**)&devInfo,sizeof(int));
	double *d_work = NULL;
	int lwork = 0;
	double *d_tau = NULL;
	cudaMalloc((void**)&d_tau,sizeof(double)*zuo);

	cusolverDnDgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			&lwork
			);
	cudaMalloc((void**)&d_work,sizeof(double)*lwork);
	cusolverDnDgeqrf(
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

	cusolverDnDormqr(
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
	double one = 1;
	cublasDtrsm(
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
	cudaMemcpy(res,d_right,sizeof(double)*rank,cudaMemcpyDeviceToHost);

	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}

void QR_d(double *d_left,double *res,double *d_right,int zuo,int rank){
#if 0
	double *d_left;
	cudaMalloc((void**)&d_left,sizeof(double)*zuo*rank);
	cudaMemcpy(d_left,left,sizeof(double)*zuo*rank,cudaMemcpyHostToDevice);
	double *d_right;
	cudaMalloc((void**)&d_right,sizeof(double)*zuo);
	cudaMemcpy(d_right,right,sizeof(double)*zuo,cudaMemcpyHostToDevice);
#endif
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cusolverDnCreate(&cusolverH);
	cublasCreate(&cublasH);

	int info_gpu=0;
	int *devInfo = NULL;
	Check(cudaMalloc((void**)&devInfo,sizeof(int)));
	double *d_work = NULL;
	int lwork = 0;
	double *d_tau = NULL;
	Check(cudaMalloc((void**)&d_tau,sizeof(double)*rank));

	cusolverDnDgeqrf_bufferSize(
			cusolverH,
			zuo,rank,
			d_left,zuo,
			&lwork
			);
	Check(cudaMalloc((void**)&d_work,sizeof(double)*lwork));
	cusolverDnDgeqrf(
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

	cusolverDnDormqr(
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
	double one = 1;
	cublasDtrsm(
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
	Check(cudaMemcpy(res,d_right,sizeof(double)*rank,cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(devInfo);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}
void gproduct(double* d_U, int k, double* d_G1, int m, int r, double* d_G2, int n, double* d_G3){
	//printTensor_d(d_G1, m, r, k, "G1");
	//printTensor_d(d_G2, r, n, k, "G2");
	double* d_G1_f = NULL;
	cudaMalloc((void**)&d_G1_f,sizeof(double)*m*r*k);
	gFFT(d_U, k, d_G1, m, r, d_G1_f);
	double* d_G2_f = NULL;
	cudaMalloc((void**)&d_G2_f,sizeof(double)*r*n*k);
	gFFT(d_U, k, d_G2, r, n, d_G2_f);
	printf("\n6******************\n");

	double* d_G3_f = NULL;
	cudaMalloc((void**)&d_G3_f,sizeof(double)*m*n*k);
	tensorMultiplytensor_d(d_G1_f, 0, m, r, d_G2_f, n, k, d_G3_f);
	printf("\n6******************\n");

	gIFFT(d_U, k, d_G3_f, m, n, d_G3);
	printf("\n6******************\n");
    cudaFree(d_G1_f);
    cudaFree(d_G2_f);
    cudaFree(d_G3_f);
}
double norm(double *a,double *b,int n){
	double sum = 0;
	for(int i = 0; i< n; i++){
		sum = sum + (a[i] - b[i])*(a[i] - b[i]);
	}
	sum = sqrt(sum);
	return sum;
}

__global__ void vec2circul(double* d_V, double* d_M, int m){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<m*m){
		d_M[(tid%m+tid/m)%m+tid/m*m]=d_V[tid%m];
	}
	__syncthreads();
}
void vec2circul_d(double* d_V, double* d_M, int m){
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
void vec2circul_h(double* V, double* M, int m){
	for(int i=0; i<m; i++){
		for(int j=0; j< m; j++){
			M[j+i*m]=V[(j+m-i)%m];
		}
	}
}
__global__ void tensor2diagmatrix(double* d_T, int m, int n, int k, double* d_M){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	if(tid<m*n*k){
		d_M[tid/m*m*k+tid%m+tid/(m*n)*m]=d_T[tid];
	}
}
void tensor2diagmatrix_d(double* d_T, int m, int n, int k, double* d_M){
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
