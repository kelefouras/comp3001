
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define TIMES_TO_RUN 1 //how many times the function will run

#define N 1024*4 //input size - USE POWER OF 2 ONLY 
#define CHECK_OUTPUT   //if do not want to validate the results comment this 

#define TILE 32


__declspec(align(64)) float Y[N], test[N], A[N * N], X[N];

void MVM_init();
int Compare_MVM();
inline unsigned short int equal(float const a, float const b);


#define EPSILON 0.01

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)





__global__ void mvm_ver1(float* Y, float* A, float* X) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; j < N; j++)
		Y[id] += A[N * (id)+j] * X[j];


}



int main()
{
	cudaError_t cudaStatus;

	//------create the cuda timers------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	MVM_init(); //initialize host arrays

	float* Y_d, * A_d, * X_d; //pointers to device arrays

//---------------------------create GPU arrays------------------------------------------
	cudaStatus = cudaMalloc((void**)&Y_d, N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(Y_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMalloc((void**)&A_d, N * N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(Y_d); cudaFree(A_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMalloc((void**)&X_d, N * sizeof(float));//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(Y_d); cudaFree(A_d); cudaFree(X_d);
		return -1;//returns unsuccessfully
	}




	//--------------------copy arrays from host to device------------------------
	cudaStatus = cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(Y_d); cudaFree(A_d); cudaFree(X_d);
		return -1;//returns unsuccessfully
	}

	cudaStatus = cudaMemcpy(X_d, X, N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		cudaFree(Y_d); cudaFree(A_d); cudaFree(X_d);
		printf("\ncuda copy failed");
		return -1;//returns unsuccessfully
	}

	cudaEventRecord(start, 0); //get timer value


	for (int it = 0; it < TIMES_TO_RUN; it++) {

		//dim3 dimBlock(256, 1);
		//dim3 dimGrid(N/256, 1, 1);
		//mvm_ver1 << <dimGrid, dimBlock >> > (Y_d, A_d, X_d);

		dim3 dimBlock(TILE, 1);
		dim3 dimGrid(N / TILE, 1, 1);
		mvm_ver1 << <dimGrid, dimBlock >> > (Y_d, A_d, X_d);

	}



	cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	elapsed_time /= 1000; //convert to secs
	elapsed_time /= TIMES_TO_RUN; //time per run

	double flops = (double) (2 * N * N) / elapsed_time;
	printf("\nflops achieved %e ", flops );

	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}


	cudaStatus = cudaMemcpy(Y, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost); //copy array from GPU back to CPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(Y_d); cudaFree(A_d); cudaFree(X_d);
		return -1;//returns unsuccessfully
	}

	//MMM_default();

	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, 0);
	int numThreads = deviceProperties.clockRate;
	printf("\n % d", numThreads);

#ifdef CHECK_OUTPUT 
	if (Compare_MVM() != 0)
		printf("\n---------WRONG OUTPUT---------------\n");
	else
		printf("\n---------OUTPUT IS OK---------------\n");
#endif

	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

	return 0;
}


void MVM_init() {

	float e = 0.1234, p = 0.7264, r = 0.11;

	for (unsigned int j = 0; j < N; j++) {
		Y[j] = 0.0;
		test[j] = 0.0;
		X[j] = (j % 100) + 0.1; //printf(" %3.1f",B[i][j]);
	}

	for (unsigned int i = 0; i < N; i++) { //printf("\n");
		for (unsigned int j = 0; j < N; j++) {

			A[N * i + j] = (j % 9) + 0.2; //printf(" %3.1f",A[i][j]);

		}
	}


}




unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp) < EPSILON)
		return 0; //success
	else
		return 1;
}


int Compare_MVM() {

	float tmp;
	int i, j, k;

	//optimize the following, otherwise it takes too long...however, to allow VS to use the \pragmas you must go 
		//in project  properties and enable that (look at the lab session document for more info)
#pragma omp parallel 
	{
#pragma omp for private(i, j, tmp)
		for (i = 0; i < N; i++) {
			tmp = 0.0;
#pragma omp simd reduction(+:tmp) aligned(Y,A,X:64)
			for (j = 0; j < N; j++) {
				tmp += A[N * i + j] * X[j];
			}
			test[i] = tmp;
		}
	}

	for (j = 0; j < N; j++)
		if (equal(Y[j], test[j]) == 1) {
			printf("\n wrong at (%d) - %f %f", j, Y[j], test[j]);
			return -1;
		}
	return 0;
}





