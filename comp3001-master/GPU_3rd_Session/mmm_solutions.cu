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

#define TIMES_TO_RUN 1 //how many times the function will run

#define N 1024 //input size - USE POWER OF 2 ONLY 
//#define CHECK_OUTPUT   //if do not want to validate the results comment this 

#define TILE 16 //use either 32 or 16 only
#define TILE_x2 TILE*2
#define TILE_x4 TILE*4

__declspec(align(64)) float C[N*N], test[N*N], A[N*N], B[N*N]; //square matrixes are considered only, stored as 1d arrays

void MMM_init();
void MMM_default();
int Compare_MMM();
inline unsigned short int equal(float const a, float const b);


#define EPSILON 0.01

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)




//This implementation uses a 2d grid and 2d blocks of threads.
//each thread computes a value in C[], thus N*N threads
__global__ void mmm_ver1(float *C,float *A, float*B) {

	/*	for (int i = 0; i < N; i++) //to be parallelized
		  for (int j = 0; j < N; j++) //to be parallelized
			for (int k = 0; k < N; k++) //serial computation - each thread
				C[N * i + j] += A[N * i + k] * B[N * k + j];				*/

	float tmp = 0.0;

	int i = blockIdx.x * blockDim.x + threadIdx.x; //i loop has been parallelized
	int j = blockIdx.y * blockDim.y + threadIdx.y; //j loop has been parallelized

	if (i < N && j < N) {//if threads not exceed the array bounds
		for (int k = 0; k < N; k++) {
			tmp += A[N * i + k] * B[N * k + j]; //each thread multiplies a row of A by a column of B
		}

		C[N * i + j] = tmp;
	}

}

//Tiled version, uses shared memory
//This implementation uses a 2d grid and 2d blocks of threads.
//each thread computes a value in C[], thus N*N threads
__global__ void mmm_tiled(float* C, float* A, float* B) {

	__shared__ float aa[TILE][TILE];
	__shared__ float bb[TILE][TILE];
	float tmp = 0.0;
	int k, m;

	int row_A = TILE * blockIdx.y + threadIdx.y;
	int col_B = blockIdx.x * TILE + threadIdx.x;
	//initialize the shared arrays
	for (m = 0; m < N / TILE; m++) {
		aa[threadIdx.y][threadIdx.x] = A[N * (row_A)+(m * TILE + threadIdx.x)];
		bb[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)];

		__syncthreads(); //all threads wait until the arrays are initialized. 

		for (k = 0; k < TILE; k ++) {
			tmp += aa[threadIdx.y][k] * bb[k][threadIdx.x]; //each thread multiplies a sub-row of A by a sub-column of B. Each thread has its own tmp register
		}

		__syncthreads();//all threads wait until all the multiplications have finished 
	}
	C[N * row_A + col_B] = tmp; //each thread writes back to memory one element of C[]


}

//Like mmm_tiled() but loop unroll has been manually applied to k loop
//Tiled version, uses shared memory
//This implementation uses a 2d grid and 2d blocks of threads.
//each thread computes a value in C[], thus N*N threads
__global__ void mmm_tiled_unrolled(float* C, float* A, float* B) {

	__shared__ float aa[TILE][TILE];
	__shared__ float bb[TILE][TILE];
	float tmp = 0.0;
	int k,m;

	int row_A = TILE * blockIdx.y + threadIdx.y;
	int col_B = blockIdx.x * TILE + threadIdx.x;
	
		
	//initialize the shared arrays
	for (m = 0; m < N / TILE; m++) {
		aa[threadIdx.y][threadIdx.x] = A[N * (row_A)+(m * TILE + threadIdx.x)];
		bb[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)];

		__syncthreads();

		for (k = 0; k < TILE; k+=8) {
			tmp += aa[threadIdx.y][k] * bb[k][threadIdx.x];
			tmp += aa[threadIdx.y][k+1] * bb[k+1][threadIdx.x];
			tmp += aa[threadIdx.y][k+2] * bb[k+2][threadIdx.x];
			tmp += aa[threadIdx.y][k+3] * bb[k+3][threadIdx.x];
			tmp += aa[threadIdx.y][k+4] * bb[k+4][threadIdx.x];
			tmp += aa[threadIdx.y][k + 5] * bb[k + 5][threadIdx.x];
			tmp += aa[threadIdx.y][k + 6] * bb[k + 6][threadIdx.x];
			tmp += aa[threadIdx.y][k + 7] * bb[k + 7][threadIdx.x];
		}

		__syncthreads();
	}
		C[N*row_A + col_B] = tmp;
	

}


//like mmm_tiled, with register blocking
//This implementation uses a 2d grid and 2d blocks of threads.
//each thread computes a value in C[], thus N*N threads
__global__ void mmm_tiled_regblocking_factor2 (float* C, float* A, float* B) {

	__shared__ float aa1[TILE][TILE];
	__shared__ float bb1[TILE][TILE];
	__shared__ float aa2[TILE][TILE];
	__shared__ float bb2[TILE][TILE];

	float tmp0 = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;
	int k, m;

	int row_A = N*(TILE_x2 * blockIdx.y + threadIdx.y);
	int col_B = blockIdx.x * TILE_x2 + threadIdx.x;


	for (m = 0; m < N / TILE; m++) {
		//initialize the shared arrays
		aa1[threadIdx.y][threadIdx.x] = A[(row_A)+m * TILE + threadIdx.x];
		aa2[threadIdx.y][threadIdx.x] = A[(row_A + N * TILE)+(m * TILE + threadIdx.x)];
		bb1[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)];
		bb2[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)+TILE];

		__syncthreads();//all threads wait until the arrays are initialized.

		for (k = 0; k < TILE; k++) {
			//each thread multiplies 2 sub-rows by 2 sub-columns. each thread uses four registers for storing the intermediate results
			tmp0 += aa1[threadIdx.y][k] * bb1[k][threadIdx.x];
			tmp1 += aa1[threadIdx.y][k] * bb2[k][threadIdx.x];
			tmp2 += aa2[threadIdx.y][k] * bb1[k][threadIdx.x];
			tmp3 += aa2[threadIdx.y][k] * bb2[k][threadIdx.x];
		}

		__syncthreads();//all threads wait until all the multiplications have finished 
	}
	C[row_A + col_B] = tmp0;
	C[row_A + col_B + TILE] = tmp1;
	C[row_A + N * TILE  + col_B ] = tmp2;
	C[row_A + N * TILE + col_B  + TILE] = tmp3;


}



//In mmm_sw_pipeline() routine, software pipelining is used to hide memory latency. In this case, we use two  times  more  shared memory,  since  we  need  the  current tiles  and the  next ones.  When  the  current  tiles are multiplied by each other, the next tiles are loaded from DDR to shared memory, in parallel; in this way,  the  cores  do  not  remain  idle  until  the  tiles  are  fetched.  This  is  an  advanced  implementation  and perhaps out of the scope of this module.
 __global__ void mmm_sw_pipeline(float* C, float* A, float* B) {

	__shared__ float aa1[32][32]; //current tile of A
	__shared__ float bb1[32][32]; //current tile of B

	__shared__ float aa1_next[32][32]; //next tile of A
	__shared__ float bb1_next[32][32]; //next tile of B



	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	int k, m;
	const int row = by * (32 ) + ty;
	const int col = bx * (32 ) + tx;

	float p1 = 0.0;


	//load tiles for m=0, these are the first tiles
	aa1[ty][tx] = A[row * N + (0 * 32 + tx)];
	bb1[ty][tx] = B[(0 * 32 + ty) * N + col];


	__syncthreads();


	for (m = 1; m < ((N / 32) - 1); m += 2) {

		//-------------------------------------m
		for (k = 0; k != 32; k++) {
			p1 += aa1[ty][k] * bb1[k][tx]; //multiply the current tiles
			
		}


		aa1_next[ty][tx] = A[row * N + (m * 32 + tx)];//load the next tiles	 
		bb1_next[ty][tx] = B[(m * 32 + ty) * N + col];
		

		__syncthreads();

		//----------------------------------------m+1
		for (k = 0; k != 32; k++) {
			p1 += aa1_next[ty][k] * bb1_next[k][tx]; //multiply the next tiles
			
		}


		aa1[ty][tx] = A[row * N + ((m + 1) * 32 + tx)]; //load the current tiles 
		bb1[ty][tx] = B[((m + 1) * 32 + ty) * N + col];
		
		__syncthreads();

	}

	//padding code follows

	for (int k = 0; k != 32; k++) {
		p1 += aa1[ty][k] * bb1[k][tx];
				
	}
	m = ((N / 32) - 1);
	aa1_next[ty][tx] = A[row * N + (m * 32 + tx)];
	bb1_next[ty][tx] = B[(m * 32 + ty) * N + col];
	
	__syncthreads();

	for (k = 0; k != 32; k++) {
		p1 += aa1_next[ty][k] * bb1_next[k][tx];
		}

	__syncthreads();

	C[row * N + col] = p1;
	
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

	//dim3 dimBlock(TILE, TILE, 1);
	//dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1);
	//mmm_tiled << <dimGrid, dimBlock >> > (C_d, A_d, B_d);
	//mmm_sw_pipeline <<< dimGrid, dimBlock >>> (C_d, A_d, B_d);
	
		dim3 dimBlock(TILE, TILE, 1); 
		dim3 dimGrid((N + TILE_x2 - 1) / (TILE_x2), (N + TILE_x2 - 1) / (TILE_x2), 1);
		mmm_tiled_regblocking_factor2 << <dimGrid, dimBlock >> > (C_d, A_d, B_d);

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





