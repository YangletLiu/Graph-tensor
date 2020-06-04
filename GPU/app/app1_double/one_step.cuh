#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
//#define NDEBUG
#include <assert.h>

/*
 * INPUT:
 * @param: T_omega_f m*n*k
 * @param: omega_f m*n*k
 * @param: X_f m*r*k
 * OUTPUT:
 * @param: Y_f r*n*k
*/

void one_step(double* dTomega_f, double* domega_f, double* dX_f, double* dY_f, int m, int n, int k, int r)
{
//X_f 张量对角化
	//printTensor_d(dX_f, m, r, k, "A");
    double* dX_f_new = NULL;
    Check(cudaMalloc((void**)&dX_f_new,sizeof(double)*k*m*k*r));
    cudaMemset(dX_f_new, 0, sizeof(double)*k*m*k*r);
    tensor2diagmatrix_d(dX_f, m, r, k, dX_f_new);
    cudaDeviceSynchronize();
   // printMatrix_d(dX_f_new, k*m, k*r, "AM");

    double* dtensor_V = NULL;
    Check(cudaMalloc((void**)&dtensor_V,sizeof(double)*k*m*n));
    tensor_tranpose_XZY_d(dTomega_f, m, n, k, dtensor_V);
    cudaDeviceSynchronize();
    //printTensor_d(dTomega_f, m, n, k, "A");
    //printTensor_d(dtensor_V, m, k, n, "B");

    double* domega_ft = NULL;
    Check(cudaMalloc((void**)&domega_ft,sizeof(double)*k*m*n));
    transSliceToTubal_d(domega_f, m, n, k, domega_ft);
    cudaDeviceSynchronize();
    //printTensor_d(domega_f, m, n, k, "A");
    //printTensor_d(domega_ft, k, m, n, "AT");

	double* omega_f_3D = (double*)malloc(sizeof(double)*k*k*m);
    double* domega_f_3D = NULL;
    Check(cudaMalloc((void**)&domega_f_3D,sizeof(double)*k*k*m));

    //vec2circul_d(domega_ft, domega_f_3D, k);
	for(int i=0; i<m; i++){
		vec2circul_d(domega_ft+i*k, domega_f_3D+i*k*k, k);
		cudaDeviceSynchronize();
	}

	//printTensor_d(domega_f_3D, k, k, m, "D");
	Check(cudaMemcpy(omega_f_3D, domega_f_3D, sizeof(double)*k*k*m, cudaMemcpyDeviceToHost));

	double* omega_f_new = (double*)malloc(sizeof(double)*k*m*k*m);
	memset(omega_f_new,0,sizeof(double)*k*m*k*m);
	double* domega_f_new = NULL;
	Check(cudaMalloc((void**)&domega_f_new,sizeof(double)*k*m*k*m));
	int x = 0;
	int y = 0;
	int z = 0;
	int row=0;
	int col=0;
	for(int i=0; i<k*k*m; i++){
		z = i/(k*k);
		x = (i%(k*k))/k;
		y = i%k;
		row=x*m+z;
		col=y*m+z;
		omega_f_new[row*k*m+col]=omega_f_3D[i];
	}
	Check(cudaMemcpy(domega_f_new, omega_f_new, sizeof(double)*k*m*k*m, cudaMemcpyHostToDevice));

	double* dtemp = NULL;
	Check(cudaMalloc((void**)&dtemp,sizeof(double)*k*m*k*r));
    //printMatrix_d(domega_f_new, k*m, k*m, "A");
	//printMatrix_d(dX_f_new, k*m, k*r, "B");
    matrixMultiply_d(domega_f_new, 0, k*m, k*m, dX_f_new, k*r, dtemp);
    cudaDeviceSynchronize();
    //printMatrix_d(dtemp, k*m, k*r, "B");

    double* temp_Y_f = (double*)malloc(sizeof(double)*r*k);
    double* Y_f = (double*)malloc(sizeof(double)*r*n*k);
    memset(Y_f,0,sizeof(double)*r*n*k);
/*********************************************************************/
    for(int t = 0; t < n; t++){
    	//printf("\n1******************\n");
    	//printMatrix_d(dtemp, k*m, k*r, "T");
    	//printTensor_d(dtensor_V, m, k, n, "V");
    	double* dtensor_half = NULL;
    	Check(cudaMalloc((void**)&dtensor_half,sizeof(double)*k*m));
    	Check(cudaMemcpy(dtensor_half, dtensor_V+t*k*m, sizeof(double)*k*m, cudaMemcpyDeviceToDevice));
    	double* dtemp_qr = NULL;
    	Check(cudaMalloc((void**)&dtemp_qr,sizeof(double)*k*m*k*r));
    	Check(cudaMemcpy(dtemp_qr, dtemp, sizeof(double)*k*m*k*r, cudaMemcpyDeviceToDevice));

    	QR_d(dtemp_qr, temp_Y_f, dtensor_half, k*m, k*r);
    	cudaDeviceSynchronize();
    	cudaFree(dtensor_half);
    	cudaFree(dtemp_qr);

        for (int j = 0; j < k; j++)
        	for (int a =0; a < r; a++)
        		Y_f[j*r*n + t*r + a] = temp_Y_f[j*r + a];
        //printf("\n2******************\n");
    }
    Check(cudaMemcpy(dY_f, Y_f, sizeof(double)*r*n*k, cudaMemcpyHostToDevice));
    //printTensor_d(dY_f, r, n, k, "Y");

    cudaFree(dX_f_new);
    cudaFree(dtensor_V);
    cudaFree(domega_ft);

    free(omega_f_3D);
    cudaFree(domega_f_3D);
    free(omega_f_new);
    cudaFree(domega_f_new);
    cudaFree(dtemp);
    free(temp_Y_f);
    free(Y_f);
}
