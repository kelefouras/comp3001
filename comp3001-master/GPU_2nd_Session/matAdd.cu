/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

int compare();

#define N 3000 //input size
int A[N][N];
int B[N][N];
int C[N][N];

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)
#define MAX_NUMBER_OF_THREADS 1024 //max number of threads per block that our GPU can execute

__device__ int device_a[N][N]; //allocate the device arrays statically (global GPU memory)
__device__ int device_b[N][N]; //allocate the device arrays statically (global GPU memory)
__device__ int device_c[N][N]; //allocate the device arrays statically (global GPU memory)

void initialize() {

	int i, j;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = (i + j) % 100;
			B[i][j] = (i - j) % 12;
		}
}



__global__ void matAdd() {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		device_c[i][j] = device_a[i][j] + device_b[i][j];
	}

}

/* In C, the "main" function is treated the same as every function,
*  it has a return type (and in some cases accepts inputs via parameters).
*  The only difference is that the main function is "called" by the operating
*  system when the user runs the program.
*  Thus the main function is always the first code executed when a program starts.
*  This function returns an integer representing the application software status.
*/
int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;
	initialize();

	/* Copy the first 2D array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_a, A, N * N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	/* Copy the second 2D array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_b, B, N * N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}


	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((N + 32 - 1) / 32, (N + 32 - 1) / 32, 1);

	/* Invocation of the kernel addWith2DGrid1DBlock with the execution configuration previously defined */
	matAdd << <dimGrid, dimBlock >> > ();


	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}

	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpyFromSymbol(C, device_c, N * N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	compare();


	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaDeviceReset();

	return 0;
}

int compare() {

	int i, j;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if ((A[i][j] + B[i][j]) != C[i][j]) {
				printf("\n\wrong results at (%d, %d)\n", i, j);
				return -1;
			}
		}
	printf("\nResults are correct\n");
	return 0;
}
