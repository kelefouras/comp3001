/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100 //input size

void initialization(float*in, float*out);
void sin_serial(const float* in, float* out);
void print_arrays(const float* in, const float* out);
int compare(const float* in, float* out);
__global__ void sin_parallel(const float* in, float* out); //CUDA kernel - this function will run on the GPU


int main(){

    float* input, * output;
    float* d_input, * d_output;

    cudaError_t cudaStatus;

    input = (float*)malloc(N * sizeof(float)); //dynamically allocate CPU memory using malloc
    if (input == NULL) { //if memory asked cannot be allocated, e.g., too large
        printf("\nmemory did not allocated\n");
        return -1;
    }


    output = (float*)malloc(N * sizeof(float)); //dynamically allocate CPU memory using malloc
    if (output == NULL) {
        printf("\nmemory did not allocated\n");
        free(input);
        return -1;
    }

    initialization(input, output);

    cudaStatus=cudaMalloc(&d_input, N * sizeof(float));//dynamically allocate memory in GPU using cudamalloc
    if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available 
        printf("\ncudaMalloc failed!");
        free(input); free(output);
        return -1;
    }

    cudaStatus=cudaMalloc(&d_output, N * sizeof(float));//dynamically allocate memory in GPU using cudamalloc
    if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available 
        printf("\ncudaMalloc failed!");
        free(input); free(output); cudaFree(d_input);
        return -1;
    }

  

    cudaStatus=cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("\ncudaMemcpy failed!");
        free(input); free(output); cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    dim3 dimGrid(10, 1, 1);//1d grid consists of 10 blocks. x dim is the first (x, y, z )
    dim3 dimBlock(10, 1, 1); //1d blocks consisting of 10 threads. x dim is the first (x, y, z )
    
    sin_parallel << <dimGrid, dimBlock >> > (d_input, d_output); 

    cudaError_t error = cudaGetLastError(); //get the status of the last cuda function that was called
    if (error != cudaSuccess) //if the hello() function did not run appropriately 
        printf("\nError %s\n",cudaGetErrorString(error)); //use this function to show the description of the error

    cudaStatus=cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("\ncudaMemcpy failed!");
        free(input); free(output); cudaFree(d_input); cudaFree(d_output);
        return -1;
    }

    

    compare(input, output);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    /*cudaDeviceReset() Explicitly destroys and cleans up all resources associated with the current device in the 
    current process. Any subsequent API call to this device will reinitialize the device.
    Note that this function will reset the device immediately.It is the caller's responsibility 
    to ensure that the device is not being accessed by any other host threads from the process 
    when this function is called. */

 
   // print_arrays(input, output);

    //sin_serial(input, output);
    //print_arrays(input, output);

    free(input);//free the memory allocated dynamically in the CPU
    free(output); //free the memory allocated dynamically in the CPU
    cudaFree(d_input);//free the memory allocated dynamically in the GPU
    cudaFree(d_output);//free the memory allocated dynamically in the GPU

    return 0;
}


void initialization(float* in, float* out) {

    int i;
    for (i = 0; i < N; i++) {
        in[i] = (float)(rand() / 7.1);
        out[i] = 0.0;
    }

}

void print_arrays(const float* in, const float* out) {

    int i;
    for (i = 0; i < N; i++)
        printf("\ninput, output are %f , %f\n",in[i],out[i]);

    printf("\n\n");

}


void sin_serial(const float* in, float* out) {

    int i;
    for (i = 0; i < N; i++)
        out[i] = sinf(in[i]);

}


int compare(const float* in, float* out) {

    int i;
    for (i = 0; i < N; i++) {
        if (fabs(out[i] - sinf(in[i])) > 0.001) {
            printf("\n\wrong results %f - %f\n", out[i], sinf(in[i]));
            return -1;
        }
    }
    printf("\nResults are correct\n");
    return 0;
}


__global__ void sin_parallel(const float* in, float* out) {
    int g_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (g_id < N) {
        out[g_id] = sinf(in[g_id]);
        //printf("\n %f %f\n", in[g_id], out[g_id]);
    }
    
}
