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
#include <omp.h>


#define N 256 //input size


__declspec(align(64)) float test[N][N][N], sum[N][N][N], A[N][N][N], C[N][N]; 

__device__ float device_sum[N][N][N], device_A[N][N][N], device_C[N][N]; //allocate the device arrays statically (global GPU memory)

void init();
void q3();
int Compare();
inline unsigned short int equal(float const a, float const b);


#define EPSILON 0.00001

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)


__global__ void diotgen_ver1() {

//write your code here

}

int main()
{
	cudaError_t cudaStatus;

	//------create the cuda timers------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	int devId = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devId);
	printf("\n Device: %s \n", prop.name);

	init(); //initialize host arrays


	/* Copy the A array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_A, A, N * N *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	/* Copy the C array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_C, C, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	cudaEventRecord(start, 0); //get timer value


		dim3 dimBlock(1,1 ,1 );
		dim3 dimGrid(1 ,1 ,1 );
		diotgen_ver1 << <dimGrid, dimBlock >> > ( );
		


	cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpyFromSymbol(sum, device_sum, N * N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		return -1;
	}

	//do not forget to print the flops value achieved

	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}



	if (Compare() != 0)
		printf("\n---------WRONG OUTPUT---------------\n");
	else
		printf("\n---------OUTPUT IS OK---------------\n");


	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

	return 0;
}


void init() {

	float e = 0.12, p = 0.72;
	unsigned int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i][j] = (j % 9) + p;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				sum[i][j][k] = 0.0;
				test[i][j][k] = 0.0;
				A[i][j][k] = (((i + j) % 99) + e);
			}
		}
	}


}

//this is the routine that you will parallelize 
void q3() {

	for (int r = 0; r < N; r++)
		for (int q = 0; q < N; q++)
		   for (int s = 0; s < N; s++)
			for (int p = 0; p < N; p++)
					test[r][q][p] += A[r][q][s] * C[s][p];


}


unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp/b) < EPSILON)
		return 0; //success
	else
		return 1;
}


int Compare() {


	for (int r = 0; r < N; r++)
		for (int q = 0; q < N; q++)
			for (int p = 0; p < N; p++)
				for (int s = 0; s < N; s++)
					test[r][q][p] = test[r][q][p] + A[r][q][s] * C[s][p];


	for (int r = 0; r < N; r++)
		for (int q = 0; q < N; q++)
				for (int p = 0; p < N; p++)
					if (equal(sum[r][q][p], test[r][q][p]) == 1) {
				      printf("\n wrong at (%d,%d,%d)", r, q,p);
					  return -1;
					}
	return 0;
}





