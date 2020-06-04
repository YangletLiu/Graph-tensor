#include "gQR.h"

/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: A 图张量 m*n*k
 * OUTPUT:
 * @param: Q R
*/
/*
void gQR_3D_batched_d(float* d_U, int k, float* d_A, int m, int n, float* d_Q, float* d_R){
        int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
	gFT_3D_batched_d(d_U, k, d_A, m, n, d_A_f);
	
	//tQR
	float* d_Q_f = NULL;
	float* d_R_f = NULL;
	cudaMalloc((void**)&d_Q_f,sizeof(float)*k*m*n);
	cudaMalloc((void**)&d_R_f,sizeof(float)*k*n*n);
	int step = m*n;
	int step_R = n*n;
	for(int i=0;i<k;i++){
		AQR(d_A_f+step*i, m, n, d_Q+step*i, d_R+step_R*i);
	}
	gIFT_3D_batched_d(d_U, k, d_Q_f, m, n, d_Q);
        gIFT_3D_batched_d(d_U, k, d_R_f, n, n, d_R);
	cudaFree(d_Q_f);
	cudaFree(d_R_f);
}
*/

void gQR_3D_batched_d(float* d_U, int k, float* d_A, int m, int n, float* d_R){
	int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
	gFT_3D_batched_d(d_U, k, d_A, m, n, d_A_f);

	//tQR
	//float* d_Q_f = NULL;
	float* d_R_f = NULL;
	//cudaMalloc((void**)&d_Q_f,sizeof(float)*k*m*n);
	cudaMalloc((void**)&d_R_f,sizeof(float)*k*n*n);
    batch_qr(d_A_f, m, n, k, d_R_f);
	//gIFT_3D_batched_d(d_U, k, d_Q_f, m, n, d_Q);
	gIFT_3D_batched_d(d_U, k, d_R_f, n, n, d_R);
	//cudaFree(d_Q_f);
	cudaFree(d_R_f);
}

void gQR_3D_based_d(float* d_U, int k, float* d_A, int m, int n, float* d_Q, float* d_R){
        int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
	gFT_3D_based_d(d_U, k, d_A, m, n, d_A_f);
	
	//tQR
	float* d_Q_f = NULL;
	float* d_R_f = NULL;
	cudaMalloc((void**)&d_Q_f,sizeof(float)*k*m*n);
	cudaMalloc((void**)&d_R_f,sizeof(float)*k*n*n);
	int step = m*n;
	int step_R = n*n;
	for(int i=0;i<k;i++){
		AQR(d_A_f+step*i, m, n, d_Q+step*i, d_R+step_R*i);
	}
	gIFT_3D_based_d(d_U, k, d_Q_f, m, n, d_Q);
        gIFT_3D_based_d(d_U, k, d_R_f, n, n, d_R);
	cudaFree(d_Q_f);
	cudaFree(d_R_f);

}

void AQR(float* d_A, int m, int n, float* d_Q, float* d_R){
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;
cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
//const int m = 3;
//const int n = 2;
const int lda = m;

//float A[lda*n];
//float Q[lda*n]; // orthonormal columns
float R[n*n]; // R = I - Q**T*Q
//float *d_A = NULL;
float *d_tau = NULL;
int *devInfo = NULL;
float *d_work = NULL;
//float *d_R = NULL;
int lwork_geqrf = 0;
int lwork_orgqr = 0;
int lwork = 0;
int info_gpu = 0;
const float h_one = 1;
const float h_minus_one = -1;
// step 1: create cusolverDn/cublas handle
cusolver_status = cusolverDnCreate(&cusolverH);
//assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
cublas_status = cublasCreate(&cublasH);
assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
//cudaStat1 = cudaMalloc ((void**)&d_A , sizeof(float)*lda*n);
cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float)*n);
cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
//cudaStat4 = cudaMalloc ((void**)&d_R , sizeof(float)*n*n);
//assert(cudaSuccess == cudaStat1);
assert(cudaSuccess == cudaStat2);
assert(cudaSuccess == cudaStat3);
//assert(cudaSuccess == cudaStat4);
//cudaStat1 = cudaMemcpy(d_A, A, sizeof(float)*lda*n, cudaMemcpyHostToDevice);
//assert(cudaSuccess == cudaStat1);

// step 3: query working space of geqrf and orgqr
cusolver_status = cusolverDnSgeqrf_bufferSize(
cusolverH,
m,
n,
d_A,
lda,
&lwork_geqrf);
assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
cusolver_status = cusolverDnSorgqr_bufferSize(
cusolverH,
m,
n,
n,
d_A,
lda,
d_tau,
&lwork_orgqr);
assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
// lwork = max(lwork_geqrf, lwork_orgqr)
lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
cusolver_status = cusolverDnSgeqrf(
cusolverH,
m,
n,
d_A,
lda,
d_tau,
d_work,
lwork,
devInfo);
cudaStat1 = cudaDeviceSynchronize();
assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
assert(cudaSuccess == cudaStat1);
// check if QR is successful or not
cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
assert(cudaSuccess == cudaStat1);
//printf("after geqrf: info_gpu = %d\n", info_gpu);
//assert(0 == info_gpu);

// step 5: compute Q
cusolver_status= cusolverDnSorgqr(
cusolverH,
m,
n,
n,
d_A,
lda,
d_tau,
d_work,
lwork,
devInfo);
cudaStat1 = cudaDeviceSynchronize();
assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
assert(cudaSuccess == cudaStat1);

// check if QR is good or not
cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
cudaMemcpyDeviceToHost);
assert(cudaSuccess == cudaStat1);
//printf("after orgqr: info_gpu = %d\n", info_gpu);
assert(0 == info_gpu);
cudaStat1 = cudaMemcpy(d_Q, d_A, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice);
assert(cudaSuccess == cudaStat1);
//printf("Q = (matlab base-1)\n");
//printMatrix(m, n, Q, lda, "Q");

// step 6: measure R = I - Q**T*Q
memset(R, 0, sizeof(float)*n*n);
for(int j = 0 ; j < n ; j++){
R[j + n*j] = 1.0; // R(j,j)=1
}
cudaStat1 = cudaMemcpy(d_R, R, sizeof(float)*n*n, cudaMemcpyHostToDevice);
//assert(cudaSuccess == cudaStat1);
// R = -Q**T*Q + I
cublas_status = cublasSgemm(
cublasH,
CUBLAS_OP_T, // Q**T
CUBLAS_OP_N, // Q
n, // number of rows of R
n, // number of columns of R
m, // number of columns of Q**T
&h_minus_one, /* host pointer */
d_A, // Q**T
lda,
d_A, // Q
lda,
&h_one, /* hostpointer */
d_R,
n);
assert(CUBLAS_STATUS_SUCCESS == cublas_status);
//if (d_A ) cudaFree(d_A);
if (d_tau ) cudaFree(d_tau);
if (devInfo) cudaFree(devInfo);
if (d_work ) cudaFree(d_work);
//if (d_R ) cudaFree(d_R);
if (cublasH ) cublasDestroy(cublasH);
if (cusolverH) cusolverDnDestroy(cusolverH);
//cudaDeviceReset();
}


void batch_qr(float* d_t, const int m, const int n, const int batch, float* d_tau)
{
    
    if(magma_init() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_init error!",__FUNCTION__,__LINE__);
		return;
    }
    magma_queue_t queue=NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);
    
//	    float *h_Amagma;
// 	    float *htau_magma;
    float *d_A, *dtau_magma;
    float **dA_array = NULL;
    float **dtau_array = NULL;

    magma_int_t   *dinfo_magma;
    magma_int_t M, N, lda, ldda, min_mn;
    magma_int_t batchCount;
    magma_int_t column;

    M = m;
    N = n;
    batchCount = batch;
    min_mn = ((m<n)?m:n);
    lda    = M;
//            n2     = lda * N * batchCount;
//    ldda = ((M+31)/32)*32;
    ldda = magma_roundup( M, 32 );
//            magma_cmalloc_cpu( &h_Amagma,   n2     );
//            magma_cmalloc_cpu( &htau_magma, min_mn * batchCount );
     if(magma_smalloc( &d_A,   ldda*N * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_smalloc( &dtau_magma,  min_mn * batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }

     if(magma_imalloc( &dinfo_magma,  batchCount ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
 
     if(magma_malloc((void**) &dA_array,   batchCount * sizeof(float*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     if(magma_malloc((void**) &dtau_array, batchCount * sizeof(float*) ) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_malloc error!",__FUNCTION__,__LINE__);
		return;
     }
     column = N * batchCount;

     magma_scopymatrix(M, column, d_t, M, d_A, ldda, queue );
//print_device_tensor(d_t,M,N,batch);	
//   magma_cprint_gpu(M*column, 1, d_t, M*column, queue );
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
         
     magma_sset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, queue );
     magma_sset_pointer( dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount, queue );
  
//    magma_cprint_gpu(M, column, d_A, ldda, queue );
//    magma_cprint_gpu(M, column, d_t, M, queue );

    if( magma_sgeqrf_batched(M, N, dA_array, ldda, dtau_array, dinfo_magma, batchCount, queue) != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_cgeqrf_batched!",__FUNCTION__,__LINE__);
		return;
    }

    cudaDeviceSynchronize();
//   magma_cprint_gpu(M, column, d_A, ldda, queue );
//         magma_cgetmatrix( M, column, d_A, ldda, h_Amagma, lda, queue );
//   magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, htau_magma, min_mn, queue );

//     magma_cgetmatrix(min_mn, batchCount, dtau_magma, min_mn, tau, min_mn, queue );
     magma_scopymatrix(min_mn, batchCount, dtau_magma, min_mn, d_tau, min_mn, queue );
     
//   magma_cprint( M, column, h_Amagma, lda);
//   magma_cprint(min_mn, batchCount, htau_magma, min_mn);

     magma_scopymatrix(M, column, d_A, ldda, d_t, lda, queue );
//print_device_tensor(d_t,M,N,batch);	
     magma_queue_destroy( queue );

     if( d_A != NULL ){ 
     magma_free( d_A   );
     d_A = NULL;
     }

     if( dtau_magma != NULL ){
     magma_free( dtau_magma  );
     dtau_magma = NULL;
     }

     if( dinfo_magma != NULL){
     magma_free( dinfo_magma );
     dinfo_magma = NULL;
     }

     if( dA_array != NULL){
     magma_free( dA_array   );
     dA_array = NULL;
     }

     if( dtau_array != NULL){
     magma_free( dtau_array  );
     dtau_array = NULL;
     }

     if( magma_finalize() != MAGMA_SUCCESS){
		fprintf(stdout,"[%s]:[%d]magma_finalize error!",__FUNCTION__,__LINE__);
		return;
     }
	
    if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}

}
