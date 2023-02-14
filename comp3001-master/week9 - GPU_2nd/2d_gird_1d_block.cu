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

int compare(const int* a, const int* b, const int* c);

#define VECTOR_LENGTH 100000000 //input size

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)
#define MAX_NUMBER_OF_THREADS 1024 //max number of threads per block that our GPU can execute

/*
* Function:  random_ints
* --------------------
*  generates a 1D vector of size equal to num_elements where each element is between 1 and 100
*
*  Input:    num_elements - int - number of elements composing the 1D vector
*
*  Output:   x - int* (pointer to int) - 1D vector of num_elements elements
*/
void random_ints(int* x, int num_elements) {
	int i;
	int min = 1;
	int range = 1000;
	for (i = 0; i < num_elements; i++)
	{
		x[i] = i % range + 1;
	}
}

/*
* Kernel Function:  addWith2DGrid1DBlock
* --------------------
*  computes the element-wise sum of two 1D vectors of size num_elements by using the GPU
*  the kernel uses a tow-dimensional grid of a one-dimensional block
*  the two-dimensional grid is composed of a certain amount of blocks over the x- and y- dimensions that are computed based of the length of the input vectors
*  the one-dimensional block is composed of MAX_NUMBER_OF_THREADS amount of threads (parallelism of threads)
*
*  Input:    a -  int* (pointer to int) - first 1D vector of num_elements elements
*            b -  int* (pointer to int) - second 1D vector of num_elements elements
*
*  Output:   c - int* (pointer to int) - 1D vector resulting from the element-wise sum of a and b vectors
*/
__global__ void addWith2DGrid1DBlock(int* a, int* b, int* c) {

	/* Global ID of a block within the 2D grid of blocks computed based on the row-major convention
	*  Here the row-major convention is used to pass from the two-dimensional to the one-dimensional representation of the grid
	*/
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	/* Global ID of a thread within the 2D grid of 1D blocks
	*  This id is then used to access to the elements of the vectors
	*/
	int thread_id = blockDim.x * block_id + threadIdx.x;

	if (thread_id < VECTOR_LENGTH) {
		c[thread_id] = a[thread_id] + b[thread_id];
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

	/* Pointer to an integer representing the first input 1D vector on the memory of the HOST */
	int* host_a;
	/* Dynamic allocation of the amount of space in the memory of the host needed to store the values of the elements of the first input 1D vector */
	host_a = (int*)malloc(VECTOR_LENGTH * sizeof(int));
	if (host_a == NULL) {
		printf("\n memory not allocated");
		return -1;
	}

	/* Fill the elements of the first 1D vector stored in the HOST memory using random_ints function */
	random_ints(host_a, VECTOR_LENGTH);

	/* Pointer to an integer representing the second input 1D vector on the memory of the HOST */
	int* host_b;
	/* Dynamic allocation of the amount of space in the memory of the host needed to store the values of the elements of the second input 1D vector */
	host_b = (int*)malloc(VECTOR_LENGTH * sizeof(int));
	if (host_b == NULL) {
		printf("\n memory not allocated");
		free(host_a);
		return -1;
	}

	/* Fill the elements of the second 1D vector stored in the HOST memory using random_ints function */
	random_ints(host_b, VECTOR_LENGTH);

	/* Pointer to an integer representing the output 1D vector on the memory of the HOST resulting from the element-wise sum of the first and the second 1D vectors */
	int* host_c;
	host_c = (int*)malloc(VECTOR_LENGTH * sizeof(int));
	if (host_c == NULL) {
		printf("\n memory not allocated");
		free(host_a); free(host_b);
		return -1;
	}

	/* Pointer to an integer representing the first input 1D vector on the memory of the DEVICE */
	int* device_a;
	/* Dynamic allocation of the amount of space in the memory of the device needed to store the values of the elements of the second input 1D vector */
	cudaStatus = cudaMalloc((void**)&device_a, VECTOR_LENGTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available 
		printf("\ncudaMalloc failed!");
		free(host_a); free(host_b); free(host_c);
		return -1;
	}

	/* Pointer to an integer representing the second input 1D vector on the memory of the DEVICE */
	int* device_b;
	/* Dynamic allocation of the amount of space in the memory of the device needed to store the values of the elements of the second input 1D vector */
	cudaStatus = cudaMalloc((void**)&device_b, VECTOR_LENGTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available 
		printf("\ncudaMalloc failed!");
		free(host_a); free(host_b); free(host_c); cudaFree(device_a);
		return -1;
	}

	/* Pointer to an integer representing the output 1D vector on the memory of the DEVICE resulting from the element-wise sum of the first and the second 1D vectors */
	int* device_c;
	/* Dynamic allocation of the amount of space in the memory of the device needed to store the values of the elements of the second input 1D vector */
	cudaStatus = cudaMalloc((void**)&device_c, VECTOR_LENGTH * sizeof(int));
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available 
		printf("\ncudaMalloc failed!");
		free(host_a); free(host_b); free(host_c); cudaFree(device_a); cudaFree(device_b);
		return -1;
	}

	/* Copy the first 1D vector from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpy(device_a, host_a, VECTOR_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		free(host_a); free(host_b); free(host_c); cudaFree(device_a); cudaFree(device_b); cudaFree(device_c);
		return -1;
	}

	/* Copy the second 1D vector from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpy(device_b, host_b, VECTOR_LENGTH * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		free(host_a); free(host_b); free(host_c); cudaFree(device_a); cudaFree(device_b); cudaFree(device_c);
		return -1;
	}

	/* Definition of the structure of the execution configuration of the kernel based on parallelism of 2D grid of 1D blocks:
*  - Compute the total amount of threads needed to deal with all the elements of the vector
*  - Split these threads accross both the x-dimension and the y-dimension of the 2D grid of the thread blocks
*/
	int number_of_blocks = (VECTOR_LENGTH + MAX_NUMBER_OF_THREADS - 1) / MAX_NUMBER_OF_THREADS;//this is equivalent to the following:
	//number_of_blocks = ceil(VECTOR_LENGTH / MAX_NUMBER_OF_THREADS); However, ceil() introduces a performance overhead and thus it is better not to use it
	printf("\nNumber of blocks needed to cover with the length of the vector: %d\n", number_of_blocks);

	printf("\nMaximum amount of blocks accross the x-dimension and the y-dimension of the 2D grid of the thread blocks: %d\n", MAX_NUMBER_OF_BLOCKS_PER_DIM);

	int num_blocks_y = (number_of_blocks + MAX_NUMBER_OF_BLOCKS_PER_DIM - 1) / MAX_NUMBER_OF_BLOCKS_PER_DIM;//this is equivalent to the following:
	//number_of_blocks_y = ceil(number_of_blocks / MAX_NUMBER_OF_BLOCKS_PER_DIM); 
	printf("\nNumber of blocks in the y-dimension of the 2D grid of the thread blocks: %d\n", num_blocks_y);

	int num_blocks_x = (number_of_blocks + num_blocks_y - 1) / (num_blocks_y);//this is equivalent to the following:
	//number_of_blocks_y = ceil(number_of_blocks / num_blocks_y); 
	printf("\nNumber of blocks in the x-dimension of the 2D grid of the thread blocks: %d\n", num_blocks_x);
	dim3 dimBlock(MAX_NUMBER_OF_THREADS, 1, 1);
	dim3 dimGrid(num_blocks_x, num_blocks_y, 1);

	/* Invocation of the kernel addWith2DGrid1DBlock with the execution configuration previously defined */
	addWith2DGrid1DBlock << <dimGrid, dimBlock >> > (device_a, device_b, device_c);


	/*  Handling function of the CUDA runtime application programming interface.
	*   Returns the last error from a runtime call.
	*/
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}

	/* Copy back the result of the element-wise sum of the first and the second 1D vectors computed by the DEVICE from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpy(host_c, device_c, VECTOR_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		free(host_a); free(host_b); free(host_c); cudaFree(device_a); cudaFree(device_b); cudaFree(device_c);
		return -1;
	}

	compare(host_a, host_b, host_c);


	/* Deallocation of the HOST memory previously allocated by malloc storing the first 1D vector */
	free(host_a);
	/* Deallocation of the HOST memory previously allocated by malloc storing the second 1D vector */
	free(host_b);
	/* Deallocation of the HOST memory previously allocated by malloc storing the output 1D vector */
	free(host_c);
	/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the first 1D vector */
	cudaFree(device_a);
	/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the second 1D vector */
	cudaFree(device_b);
	/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the output 1D vector */
	cudaFree(device_c);

	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaDeviceReset();

	return 0;
}

int compare(const int* a, const int* b, const int* c) {

	int i;
	for (i = 0; i < VECTOR_LENGTH; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("\n\wrong results\n");
			return -1;
		}
	}
	printf("\nResults are correct\n");
	return 0;
}
