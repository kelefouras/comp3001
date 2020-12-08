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
#include <time.h>
#include <math.h>

int compare();

#define N 1024 //input size
#define TIMES_TO_RUN 10000
#define TILE 32

float A[N][N], Atranspose[N][N];


#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)

__device__ float device_A[N][N]; //allocate the device arrays statically (global GPU memory)
__device__ float device_Atranspose[N][N]; //allocate the device arrays statically (global GPU memory)


void initialize() {

	int i, j;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = (float) (((i*3 + j) % 1000)+0.01);
		}
}




__global__ void transpose_ver1() {

	int i = blockIdx.x * blockDim.x + threadIdx.x; //2d grid, 2d blocks

	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) { //this is not necessary
		device_Atranspose[i][j] = device_A[j][i]; //each thread copies one element
	}

}

__global__ void transpose_ver2() {

	int i = blockIdx.x * TILE + threadIdx.x;

	int j = blockIdx.y * TILE + threadIdx.y;

	if (i < N && j < N) {
		device_Atranspose[i][j] = device_A[j][i]; //each thread copies one element
	}

}

__global__ void transpose_ver4() {

	int x = blockIdx.x * TILE + threadIdx.x;//2d grid, 2d blocks
	int y = blockIdx.y * TILE + threadIdx.y;

	for (int m = 0; m < TILE; m += 8) {
		device_Atranspose[x][y+m] = device_A[y + m][x];//each thread copies more than one elements
	}

}


__global__ void transpose_ver5() {

	__shared__ float tile[TILE][TILE];

	int x = blockIdx.x * TILE + threadIdx.x;//2d grid, 2d blocks
	int y = blockIdx.y * TILE + threadIdx.y;

	for (int m = 0; m < TILE; m += 8) {
		tile[threadIdx.y+m][threadIdx.x] = device_A[y + m][x];//each thread copies more than one elements
	}

	__syncthreads(); //all the threads wait here until the tile array has been initialized

	x = blockIdx.y * TILE + threadIdx.x;//transpose block offset
	y = blockIdx.x * TILE + threadIdx.y;

	for (int m = 0; m < TILE; m += 8) {
		device_Atranspose[y + m][x] = tile[threadIdx.x][threadIdx.y+m];
	}

}


//this kernel does a normal copy (not transpose) from A to Atranspose. Atranspose it NOT the transpose of A here.
//the purpose of this kernel is to measure the maximum performance of copying two matrices
__global__ void normal_copy() { 

	int x = blockIdx.x * TILE + threadIdx.x; //2d grid, 2d blocks
	int y = blockIdx.y * TILE + threadIdx.y;

	for (int m = 0; m < TILE; m+=8) {
		device_Atranspose[y + m][x] = device_A[y + m][x];//each thread copies more than one elements
	}

}





int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;
	initialize();

	//create the cuda timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	/* Copy the A array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_A, A, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	
	//parameters for normal_copy() 
	//parameters for transpose_ver4()
	//parameters for transpose_ver5()
		dim3 dimBlock(TILE, 8, 1);
		dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);

	//parameters for transpose_ver1()
	//parameters for transpose_ver2()
		//dim3 dimBlock(TILE, TILE, 1);
		//dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);

	cudaEventRecord(start, 0);

	for (int it = 0; it < TIMES_TO_RUN; it++) {
		//normal_copy << <dimGrid, dimBlock >> > ();
		transpose_ver4 << <dimGrid, dimBlock >> > ();
		//transpose_ver2 << <dimGrid, dimBlock >> > ();
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	double bandwidth = double((double)(2 * N * N * sizeof(float))) / ((elapsed_time / 1000) / TIMES_TO_RUN);
	printf("\nElapsed time in msecs = %f - GB/sec = %f", elapsed_time, bandwidth/1000000000 );

	
		//dim3 dimBlock(TILE, TILE, 1);
		//dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);
		//transpose_ver2 << <dimGrid, dimBlock >> > ();
		//transpose_ver2 << <dimGrid, dimBlock >> > ();
		//transpose_ver3 << <dimGrid, dimBlock >> > ();

		
		//transpose_ver4 << <dimGrid, dimBlock >> > ();
	
	


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}

	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpyFromSymbol(Atranspose, device_Atranspose, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}



	compare();


	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus=cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

	return 0;
}

int compare() {

	int i, j;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if ( fabs ((A[i][j]) - Atranspose[j][i]) > 0.001 ) {
				printf("\n\wrong results at (%d, %d)\n", i, j);
				return -1;
			}
		}
	printf("\nResults are correct\n");
	return 0;
}
