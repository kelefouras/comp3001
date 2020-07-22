#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define TIMES_TO_RUN 1 //how many times the function will run

#define N 1024 //input size - USE POWER OF 2 ONLY 
//#define CHECK_OUTPUT   //if do not want to validate the results comment this 


__declspec(align(64)) float C[N*N], test[N*N], A[N*N], B[N*N]; //square matrixes are considered only, stored as 1d arrays

void MMM_init();
void MMM_default();
int Compare_MMM();
inline unsigned short int equal(float const a, float const b);


#define EPSILON 0.01

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)



__global__ void mmm_ver1(float *C,float *A, float*B) {

	/*	for (int i = 0; i < N; i++) //to be parallelized
		  for (int j = 0; j < N; j++) //to be parallelized
			for (int k = 0; k < N; k++) //serial computation - each thread
				C[N * i + j] += A[N * i + k] * B[N * k + j];				*/

	

}




int main()
{
	cudaError_t cudaStatus;
	
	//------create the cuda timers------
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float elapsed_time;

	MMM_init(); //initialize host arrays

	float * C_d, *A_d, *B_d; //pointers to device arrays

//---------------------------create GPU arrays------------------------------------------
	cudaStatus = cudaMalloc((void**)&C_d, N * N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(C_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMalloc((void**)&A_d, N * N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(C_d); cudaFree(A_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMalloc((void**)&B_d, N * N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
		return -1;//returns unsuccessfully
	}



	//--------------------copy arrays from host to device------------------------
	cudaStatus = cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
		printf("\ncuda copy failed");
		return -1;//returns unsuccessfully
	}


	cudaEventRecord(start, 0); //get timer value
	
	for (int it = 0; it < TIMES_TO_RUN; it++) {

	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid((1, 1, 1);
	mmm_ver1 << <dimGrid, dimBlock >> > (C_d, A_d, B_d);
	


	}

	
	cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	double flops = (double) ((double)2 * N * N * N) / (elapsed_time / TIMES_TO_RUN);
	printf("\nGflops achieved %f ", flops/1000000);
	
	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}


	cudaStatus = cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost); //copy array from GPU back to CPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
		return -1;//returns unsuccessfully
	}

	//MMM_default();

#ifdef CHECK_OUTPUT 
	if (Compare_MMM() != 0) 
		printf("\n---------WRONG OUTPUT---------------\n");
	else 
		printf("\n---------OUTPUT IS OK---------------\n");
#endif
	
	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus=cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

	return 0;
}


void MMM_init() {

	float e = 0.1234, p = 0.7264, r = 0.11;

	//MMM
	for (unsigned int i = 0; i < N; i++) { //printf("\n");
		for (unsigned int j = 0; j < N; j++) {
			C[N*i+j] = 0.0;
			test[N * i + j] = 0.0;
			A[N * i + j] = (j % 9) + p; //printf(" %3.1f",A[i][j]);
			B[N * i + j] = (j % 7) - p; //printf(" %3.1f",B[i][j]);
		}
	}


}


void MMM_default() {

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				C[N * i + j] += A[N * i + k] * B[N * k + j];

	
}


unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp) < EPSILON)
		return 0; //success
	else
		return 1;
}


int Compare_MMM() {

	float tmp;
	int i, j, k;

//optimize the following, otherwise it takes too long...however, to allow VS to use the \pragmas you must go 
	//in project  properties and enable that (look at the lab session document for more info)
#pragma omp parallel 
	{
#pragma omp for private(i, j, k, tmp)
			for (i = 0; i < N; i++) {
				for (j = 0; j < N; j++) {
					tmp = 0.0;
#pragma omp simd reduction(+:tmp) aligned(C,A,B:64)
					for (k = 0; k < N; k++) {
						tmp += A[N * i + k] * B[N * k + j];
					}
					test[N * i + j] = tmp;
				}
			}
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			if (equal(C[N * i + j], test[N * i + j]) == 1) {
				printf("\n wrong at (%d,%d) - %f %f", i, j, C[N * i + j], test[N * i + j]);
				return -1;
			}
	return 0;
}





