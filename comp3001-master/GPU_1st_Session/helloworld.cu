//this a naive 1st example

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void hello() //global specifies that this routine will run on the GPU
{
    printf("\nHello from thread %d of block %d - In total there are %d Threads and %d blocks",threadIdx.x,blockIdx.x,blockDim.x,gridDim.x);
}



int main()
{
    hello << <2, 12 >> > (); //2 blocks of threads, 12 thread per block - 24 threads in total

    cudaError_t error = cudaGetLastError(); //get the status of the last cuda function that was called
    if (error != cudaSuccess) //if the hello() function did not run appropriately 
        printf("\nError %s\n",cudaGetErrorString(error)); //use this function to show the description of the error

    cudaDeviceReset();
    /*Explicitly destroys and cleans up all resources associated with the current device in the 
    current process. Any subsequent API call to this device will reinitialize the device.
    Note that this function will reset the device immediately.It is the caller's responsibility 
    to ensure that the device is not being accessed by any other host threads from the process 
    when this function is called. */


    return 0;
}

