#include "gSVD.h"

/*
 * INPUT:
 * @param: U 图的拉普拉斯矩阵特征向量 k*k
 * @param: A 图张量 m*n*k
 * OUTPUT:
 * @param: U S V
*/

/*
void gSVD_3D_batched_d(float* d_LU, int k, float* d_A, int m, int n, float* d_U, float* d_S, float* d_V){
    int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
    gFT_3D_batched_d(d_LU, k, d_A, m, n, d_A_f);
    
    //tsvd
	cusolverDnHandle_t handle;
	gesvdjInfo_t params;
	int* info = NULL;
	int echo = 1;
	int lda = m;
	int ldu = m;
	int ldv = n;
	int lwork = 0;
	float* work=NULL;

	//malloc u s v

	float* d_S_f = NULL;
	float* d_U_f = NULL;
	float* d_V_f = NULL;
	cudaMalloc((void**)&d_S_f,sizeof(float)*k*((m<n)?m:n));
	cudaMalloc((void**)&d_U_f,sizeof(float)*k*m*((m<n)?m:n));
	cudaMalloc((void**)&d_V_f,sizeof(float)*k*n*((m<n)?m:n));
	cudaMalloc((void**)&info,sizeof(int));	
	
	if(cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info srror",__FUNCTION__,__LINE__);
		return;
	}	
	
	if(cusolverDnSgesvdj_bufferSize(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_A_f,
			m,
			d_S_f,
			d_U_f,
			ldu,
			d_V_f,
			ldv,
			&lwork,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMalloc((void**)&work,sizeof(float)*lwork);

	int step_d = m*n;
	int step_u = m*((m<n)?m:n);
	int step_s = ((m<n)?m:n);
	int step_v = n*((m<n)?m:n);	
	
	for(int i=0;i<k;i++){
	  if(cusolverDnSgesvdj(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_A_f+step_d*i,
			lda,
			d_S_f+i*step_s,
			d_U_f+i*step_u,
			ldu,
			d_V_f+i*step_v,
			ldv,
			work,
			lwork,
			info,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		}
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	if(cusolverDnDestroy(handle)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cusolverDnDestroyGesvdjInfo(params)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
	int min=(m<n)?m:n;
    
    gIFT_3D_batched_d(d_LU, k, d_U_f, m, min, d_U);
    gIFT_3D_batched_d(d_LU, k, d_S_f, min, 1, d_S);
	gIFT_3D_batched_d(d_LU, k, d_V_f, n, min, d_V);
	cudaFree(d_U_f);
	cudaFree(d_S_f);
	cudaFree(d_V_f);
}
*/
void gSVD_3D_batched_d(float* d_LU, int k, float* d_A, int m, int n, float* d_U, float* d_S, float* d_V){
	int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
	gFT_3D_batched_d(d_LU, k, d_A, m, n, d_A_f);

	float* d_S_f = NULL;
	float* d_U_f = NULL;
	float* d_V_f = NULL;
	cudaMalloc((void**)&d_S_f,sizeof(float)*k*((m<n)?m:n));
	cudaMalloc((void**)&d_U_f,sizeof(float)*k*m*((m<n)?m:n));
	cudaMalloc((void**)&d_V_f,sizeof(float)*k*n*((m<n)?m:n));
	
    SVD_method svdMethod=SVD_Jacobi;
    int rank =((m<n)?m:n);//or use low rank
    int batch = k;
    kblasHandle_t handle;
    kblasRandState_t state;
    kblasCreate( &handle );
    kblasInitRandState(handle,&state,2*n,0);

    kblasSsvd_full_batch_wsquery(handle, m, n, rank,batch,svdMethod);

    if(kblasAllocateWorkspace(handle) != 1){
            fprintf(stdout,"[%s]:[%d] kblas  wsquery err!",__FUNCTION__,__LINE__);
            return;
    }
    int stride_a=m*n;
    int stride_s=rank;
    int stride_u=m*rank;
    int stride_v=n*rank;
// on output: S: contains singular values (up to rank if rank > 0)
//            U: contains right singular vectors
//            V: contains left singular vectors scaled by S
//            A: not modified
    if(kblasSsvd_full_batch_strided(handle, m, n, rank ,d_A_f, m, stride_a, d_S_f, stride_s, d_U_f,m,stride_u,d_V_f,n,stride_v, svdMethod, state,batch) != 1){
            fprintf(stdout,"[%s]:[%d] kblas  svd  err!",__FUNCTION__,__LINE__);
            return;
    }
    kblasFreeWorkspace(handle);//out of memory
    kblasDestroyRandState(state);
    kblasDestroy(&handle);

    int min=(m<n)?m:n;
    gIFT_3D_batched_d(d_LU, k, d_U_f, m, min, d_U);
    gIFT_3D_batched_d(d_LU, k, d_S_f, min, 1, d_S);
	gIFT_3D_batched_d(d_LU, k, d_V_f, n, min, d_V);
	cudaFree(d_U_f);
	cudaFree(d_S_f);
	cudaFree(d_V_f);


}

void gSVD_3D_based_d(float* d_LU, int k, float* d_A, int m, int n, float* d_U, float* d_S, float* d_V){
    int len = m*n*k;
	float* d_A_f = NULL;//A_f 表示A的gft
	cudaMalloc((void **)& d_A_f, sizeof (float)*len);
    gFT_3D_based_d(d_LU, k, d_A, m, n, d_A_f);
    
    //tsvd
	cusolverDnHandle_t handle;
	gesvdjInfo_t params;
	int* info = NULL;
	int echo = 1;
	int lda = m;
	int ldu = m;
	int ldv = n;
	int lwork = 0;
	float* work=NULL;

	//malloc u s v

	float* d_S_f = NULL;
	float* d_U_f = NULL;
	float* d_V_f = NULL;
	cudaMalloc((void**)&d_S_f,sizeof(float)*k*((m<n)?m:n));
	cudaMalloc((void**)&d_U_f,sizeof(float)*k*m*((m<n)?m:n));
	cudaMalloc((void**)&d_V_f,sizeof(float)*k*n*((m<n)?m:n));
	cudaMalloc((void**)&info,sizeof(int));	
	
	if(cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnCreate failed!",__FUNCTION__,__LINE__);
		return;
	}
	
	if(cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:creation svd info srror",__FUNCTION__,__LINE__);
		return;
	}	
	
	if(cusolverDnSgesvdj_bufferSize(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_A_f,
			m,
			d_S_f,
			d_U_f,
			ldu,
			d_V_f,
			ldv,
			&lwork,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR: create buffersize failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda syncthronize err!",__FUNCTION__,__LINE__);
		return;
	}

	cudaMalloc((void**)&work,sizeof(float)*lwork);

	int step_d = m*n;
	int step_u = m*((m<n)?m:n);
	int step_s = ((m<n)?m:n);
	int step_v = n*((m<n)?m:n);	
	
	for(int i=0;i<k;i++){
	  if(cusolverDnSgesvdj(
			handle,
			CUSOLVER_EIG_MODE_VECTOR,
			echo,
			m,
			n,
			d_A_f+step_d*i,
			lda,
			d_S_f+i*step_s,
			d_U_f+i*step_u,
			ldu,
			d_V_f+i*step_v,
			ldv,
			work,
			lwork,
			info,
			params) != CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] CUSOLVER ERROR:cusolverDnCgesvdj failed!",__FUNCTION__,__LINE__);
		return;
		}
	}

	if(cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stdout,"[%s]:[%d] cuda synchronize err!",__FUNCTION__,__LINE__);
		return;
	}
	if(cusolverDnDestroy(handle)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}

	if(cusolverDnDestroyGesvdjInfo(params)!=CUSOLVER_STATUS_SUCCESS){
		fprintf(stdout,"[%s]:[%d] cusolverDnDestroy failed!",__FUNCTION__,__LINE__);
		return;
	}
	int min=(m<n)?m:n;
    
    gIFT_3D_based_d(d_LU, k, d_U_f, m, min, d_U);
    gIFT_3D_based_d(d_LU, k, d_S_f, min, 1, d_S);
	gIFT_3D_based_d(d_LU, k, d_V_f, n, min, d_V);
	cudaFree(d_U_f);
	cudaFree(d_S_f);
	cudaFree(d_V_f);
}

