/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//In Linux, compile with nvcc dnn.cu  -o p -Xcompiler -mavx2 -Xcompiler -fopenmp -O3  -lm 


#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>

void read_layer_dimensions();

int load_create_input_output_array_FP();
int load_filter_array_FP();
void load_bias_FP();
void deallocate_FP();
void compare_output_result_FP();
unsigned short int equal(float const a, float const b) ;

int create_device_arrays();

__global__ void cuda_layer_v1(float* in, float* filter, float* bias, float* out, unsigned int Input_Output_batch_dim, unsigned int Input_X_dim, unsigned int Input_Y_dim, unsigned int Input_depth_dim,unsigned int Stride_X_dim,unsigned int Stride_Y_dim,unsigned int Output_X_dim,unsigned int Output_Y_dim,unsigned int Output_depth_dim,unsigned int Mask_X_dim,unsigned int Mask_Y_dim);



//debugging routines
void show_32_bit_in_AVX_register(__m256i temp);


//input dimensions
unsigned int Input_Output_batch_dim;
unsigned int Input_X_dim;
unsigned int Input_Y_dim;
unsigned int Input_depth_dim;

unsigned int Stride_X_dim;
unsigned int Stride_Y_dim;
//unsigned int Stride_Z_dim;

//output dimensions
unsigned int Output_X_dim;
unsigned int Output_Y_dim;
unsigned int Output_depth_dim;
//output batch == input batch

//mask dimensions
unsigned int Mask_X_dim;
unsigned int Mask_Y_dim;
//unsigned int Mask_Z_dim;


float Scale;
unsigned int M0_by_n;
unsigned char Zero_point;
__m256i M0_by_n_vector;
__m256 Scale_vector;

//host arrays
float * in_FP; //pointer to input array
float * in_layout_FP; //pointer to input array
float * filter_FP; //pointer to filter array
float * out_FP; //pointer to output array
float * out_to_compare_with_FP; //pointer to output array to compare with
float *bias_array_FP;

//device arrays
float * in_FP_d; //pointer to input array
float * in_layout_FP_d; //pointer to input array
float * filter_FP_d; //pointer to filter array
float * out_FP_d; //pointer to output array
float *bias_array_FP_d;

unsigned long long int In_size,Out_size,Filter_size;//size of the arrays

float Relu_float(const float temp);

int unoptimized_layer_FP(const float * in, const float * filter, const float *bias_array, float * out_to_compare_with);


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define EPSILON 0.0001

#define TILE 8

int main ( ){

   // double start_time, run_time;

    cudaError_t cudaStatus;

	//------create the cuda timers------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	int devId = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devId);
	printf("\n Device: %s - shared memory %lu \n", prop.name,prop.sharedMemPerBlock);

    read_layer_dimensions( );
    
    if (Mask_Y_dim*Mask_X_dim*Input_depth_dim*TILE*sizeof(float) > prop.sharedMemPerBlock){
      printf("\n cannot use this schedule as not enough shared mem\n");
      return -1;
    }

//create and initialize host arrays
    load_bias_FP();
    load_create_input_output_array_FP();
    load_filter_array_FP();
    
    //create and initialize device arrays
    if ( create_device_arrays() !=0 ){
      printf("\n device arrays are  not created\n");
      return -1;
    }
    
    //run on the CPU to get the CPU result
    unoptimized_layer_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);


   // start_time = omp_get_wtime();
    cudaEventRecord(start, 0); //get timer value
    
    
	//--------------------copy arrays from host to device------------------------
	cudaStatus = cudaMemcpy(in_FP_d, in_FP, In_size * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d); cudaFree(bias_array_FP_d);
		return -1;//returns unsuccessfully
	}
	
	cudaStatus = cudaMemcpy(out_FP_d, out_FP, Out_size * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d); cudaFree(bias_array_FP_d);
		return -1;//returns unsuccessfully
	}
	
	cudaStatus = cudaMemcpy(filter_FP_d, filter_FP, Filter_size * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d); cudaFree(bias_array_FP_d);
		return -1;//returns unsuccessfully
	}
	
	cudaStatus = cudaMemcpy(bias_array_FP_d, bias_array_FP, Output_depth_dim * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
	if (cudaStatus != cudaSuccess) {//if cuda copy fails
		printf("\ncuda copy failed");
		cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d); cudaFree(bias_array_FP_d);
		return -1;//returns unsuccessfully
	}
	
	
//---------YOU NEED TO MODIFY THIS PART-------------------------------------
	dim3 dimBlock(1);
	dim3 dimGrid(1);
	
cuda_layer_v1 << <dimGrid, dimBlock >> > (in_FP_d, filter_FP_d, bias_array_FP_d, out_FP_d,  Input_Output_batch_dim,  Input_X_dim,  Input_Y_dim, Input_depth_dim, Stride_X_dim,Stride_Y_dim,  Output_X_dim, Output_Y_dim, Output_depth_dim, Mask_X_dim, Mask_Y_dim);

//-----------------------------------------------------------

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}
	
	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpy(out_FP, out_FP_d, Out_size * sizeof(float), cudaMemcpyDeviceToHost );
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed! (out)");
		cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d); cudaFree(bias_array_FP_d);
		return -1;
	}
    
    //run_time = (omp_get_wtime() - start_time);
    cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	//printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsed_time/=1000; //convert to seconds
	
    double FLOPS=(double) Input_Output_batch_dim*Output_Y_dim*Output_X_dim*Output_depth_dim;
    FLOPS= (FLOPS*(2*Mask_Y_dim*Mask_X_dim*Input_depth_dim+1)) / elapsed_time;

    printf("\n\nTime = %.3e seconds",  elapsed_time);
    printf(" or %.0f mseconds", elapsed_time*1000);//printf time in msecs
    printf("\nFLOPS achieved: %e\n", (double) FLOPS);//print FLOPS


        compare_output_result_FP();


        deallocate_FP();
    
	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

    return 0;
}



__global__ void cuda_layer_v1(float* in_FP, float* filter_FP, float* bias_array_FP, float* out_FP, unsigned int Input_Output_batch_dim, unsigned int Input_X_dim, unsigned int Input_Y_dim, unsigned int Input_depth_dim,unsigned int Stride_X_dim,unsigned int Stride_Y_dim,unsigned int Output_X_dim,unsigned int Output_Y_dim,unsigned int Output_depth_dim,unsigned int Mask_X_dim,unsigned int Mask_Y_dim){

//YOU NEED TO WRITE YOUR ROUTINE HERE
  
}




void compare_output_result_FP(){


    for (unsigned long long int i=0; i<Input_Output_batch_dim*Output_Y_dim*Output_X_dim*Output_depth_dim;i++){
        if (equal(out_FP[i], out_to_compare_with_FP[i]) == 1){
            printf("\n wrong values (%llu): %f %f",i, out_FP[i], out_to_compare_with_FP[i]);

        }
    }
}

unsigned short int equal(float const a, float const b) {
    float temp = a - b;

    if (b==0.0f){//cannot divide with zero
        if (a==0.0f){
            return 0;//success
        }
        else {
            return 1;
        }
    }
    else {

        if ((fabs(temp) / fabs(b)) < EPSILON) {
            return 0; //success
        } else {
            return 1;
        }
    }
}





void read_layer_dimensions(){


    Input_Output_batch_dim=40;
    Input_Y_dim=52;
    Input_X_dim=52;
    Input_depth_dim=128;

    Stride_Y_dim=1;
    Stride_X_dim=1;

    Mask_Y_dim=3;
    Mask_X_dim=3;

    Output_depth_dim=128;
    Output_X_dim=(Input_X_dim-(Mask_X_dim-Stride_X_dim)) / Stride_X_dim;
    Output_Y_dim=(Input_Y_dim-(Mask_Y_dim-Stride_Y_dim)) / Stride_Y_dim;

    In_size= (unsigned long long int) Input_Output_batch_dim*Input_X_dim*Input_Y_dim*Input_depth_dim;
    Filter_size=(unsigned long long int) Input_depth_dim*Mask_X_dim*Mask_Y_dim*Output_depth_dim;
    Out_size=(unsigned long long int) Input_Output_batch_dim*Output_Y_dim*Output_X_dim*Output_depth_dim;






    printf("\n Layer dimensions are read");
    printf("\n Input dims (batch,y,x,depth) = (%d, %d, %d, %d)       - Size in Elements = %llu", Input_Output_batch_dim, Input_Y_dim, Input_X_dim, Input_depth_dim,In_size);
    printf("\n Filter dims (m,y,x,depth) = (%d, %d, %d, %d)           - Size in Elements = %llu", Output_depth_dim, Mask_Y_dim, Mask_X_dim, Input_depth_dim,Filter_size);
    printf("\n Output dims (batch,y,x,out_depth) = (%d, %d, %d, %d) - Size in Elements = %llu", Input_Output_batch_dim, Output_Y_dim, Output_X_dim, Output_depth_dim,Out_size);

}





void load_bias_FP(){

    bias_array_FP = (float*) _mm_malloc( Output_depth_dim * sizeof(float),64);
    if (bias_array_FP==NULL) {
        printf("\nerror with malloc allocating bias array");
        exit(EXIT_FAILURE);
    }


    for (unsigned int i=0; i<Output_depth_dim; i++){
        *(bias_array_FP+i)=((float) (rand() % 5)) + 1;
        //  *(bias_array_FP+i)=0.0f;
        // printf("  %d",*(in+i));
    }



}


int create_device_arrays(){
/*
    unsigned long long int input_size= (unsigned long long int) Input_Output_batch_dim * Input_depth_dim * Input_Y_dim * Input_X_dim;
    unsigned long long int output_size=(unsigned long long int) Input_Output_batch_dim * Output_depth_dim * Output_Y_dim * Output_X_dim;
    unsigned int filter_size= Mask_X_dim * Mask_Y_dim * Input_depth_dim * Output_depth_dim;
    */
    
cudaError_t cudaStatus;

    
   cudaStatus = cudaMalloc((void**)&bias_array_FP_d, Output_depth_dim * sizeof(float) );//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(bias_array_FP_d);
		return -1;//returns unsuccessfully
	}

   cudaStatus = cudaMalloc((void**)&in_FP_d, In_size * sizeof(float) );//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(bias_array_FP_d); cudaFree(in_FP_d);
		return -1;//returns unsuccessfully
	}

   cudaStatus = cudaMalloc((void**)&out_FP_d, Out_size * sizeof(float) );//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(bias_array_FP_d); cudaFree(in_FP_d); cudaFree(out_FP_d);
		return -1;//returns unsuccessfully
	}
	
   cudaStatus = cudaMalloc((void**)&filter_FP_d, Filter_size * sizeof(float) );//allocate memory dynamically 
	if (cudaStatus != cudaSuccess) {//if the GPU memory asked is not available
		printf("\nCudaMalloc failed");
		cudaFree(bias_array_FP_d); cudaFree(in_FP_d); cudaFree(out_FP_d); cudaFree(filter_FP_d);
		return -1;//returns unsuccessfully
	}
	
	return 0;
	
}



//in[] is stored into memory like that : in[Input_Output_batch_dim] [Input_Y_dim] [Input_X_dim] [Input_depth_dim] ;
//out[] is stored into memory like that : out[Input_Output_batch_dim] [Output_Y_dim] [Output_X_dim] [Output_depth_dim] ;
int load_create_input_output_array_FP(){

    //unsigned long long int input_size= (unsigned long long int) Input_Output_batch_dim * Input_depth_dim * Input_Y_dim * Input_X_dim;
    //unsigned long long int output_size=(unsigned long long int) Input_Output_batch_dim * Output_depth_dim * Output_Y_dim * Output_X_dim;
    unsigned long long int in_subscript,out_subscript;

    in_FP = (float*) _mm_malloc( In_size * sizeof(float),64);
    if (in_FP==NULL) {
        printf("\nerror with malloc allocating input array");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for private(in_subscript)
    for (int b = 0; b < Input_Output_batch_dim; b++)
        for (int y = 0; y < Input_Y_dim; y++)
            for (int x = 0; x < Input_X_dim; x ++)
                for (unsigned int d = 0; d < Input_depth_dim; d++) {
                    in_subscript = (unsigned long long int) b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                                   + (y ) * Input_X_dim * Input_depth_dim
                                   + (x ) * Input_depth_dim +d;

                    in_FP[in_subscript] =  ( (float) (d % 50) ) + 0.73f;
                    //in_FP[in_subscript] = ((float) (rand() % 50) ) +0.73f;
                    // printf("  %d",*(in+i));
                }


    out_FP = (float*) _mm_malloc( Out_size * sizeof(float),64);
    if (out_FP==NULL) {
        printf("\nerror with malloc allocating output array");
        exit(EXIT_FAILURE);
    }


    out_to_compare_with_FP = (float*) _mm_malloc( Out_size * sizeof(float),64);
    if (out_to_compare_with_FP==NULL) {
        printf("\nerror with malloc allocating output array to compare with");
        exit(EXIT_FAILURE);
    }


#pragma omp parallel for private(out_subscript)
    for (int b = 0; b < Input_Output_batch_dim; b++)
        for (int y = 0; y < Output_Y_dim; y++)
            for (int x = 0; x < Output_X_dim; x ++)
                for (unsigned int m = 0; m < Output_depth_dim; m++) {
                    out_subscript = (unsigned long long int) b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                                    y * (Output_depth_dim * Output_X_dim) +
                                    x * Output_depth_dim
                                    + m;

                    out_to_compare_with_FP[out_subscript]=0.0f;

                    out_FP[out_subscript]=0.0f;
                }



    // printf("\n Input / Output arrays are created. Input is loaded. \n");
    return 0;
}




void deallocate_FP(){

    _mm_free(in_FP);
    _mm_free(out_FP);

    _mm_free(out_to_compare_with_FP);

    _mm_free(bias_array_FP);

    _mm_free(filter_FP);



}




// filter array is stored into memory tile-wise
int load_filter_array_FP(){

   // unsigned int filter_size= Mask_X_dim * Mask_Y_dim * Input_depth_dim * Output_depth_dim;
    unsigned int y,x,m,d,offset,cnt=0;

    filter_FP = (float*) _mm_malloc( Filter_size * sizeof(float),64);
    if (filter_FP==NULL) {
        printf("\nerror with malloc allocating filter array");
        exit(EXIT_FAILURE);
    }



    //read the filter array
    for (m=0;m<Output_depth_dim;m++)
        for (y=0;y<Mask_Y_dim;y++)
            for (x=0;x<Mask_X_dim;x++){
                //printf("\n");
                for (d=0;d<Input_depth_dim;d+=2){
                    offset=m * Mask_Y_dim*Mask_X_dim*Input_depth_dim +
                           y * Mask_X_dim*Input_depth_dim +
                           x*Input_depth_dim + d;

                    filter_FP[offset]= ((rand() % 8) + 0.973);
                    filter_FP[offset+1]=-((rand() % 8) + 0.973);
                    // printf("\n %d, %d",filter_FP[offset],filter_FP[offset+1]);
                    cnt++;
                }}


    //printf("\n Filter array is created and loaded. \n");
    return 0;
}

int unoptimized_layer_FP(const float * in_FP, const float * filter_FP, const float *bias_array_FP, float * out_to_compare_with_FP){

    float temp,bias;


    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { 
        for(unsigned int m = 0; m < Output_depth_dim; m++){
            for (unsigned int od = 0; od < 1; od++) {	//Output Depth , for 3D convolution only
                for (unsigned int y = 0; y < Output_Y_dim; y++) {			//Output height
                    for (unsigned int x = 0; x < Output_X_dim; x++) {			//Output Width
                        bias = bias_array_FP[m];
                        temp = 0.0f;
                        for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                                for(unsigned int d = 0; d < Input_depth_dim; d++) {

                                    unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim*Input_depth_dim)
                                                                          + (y*Stride_Y_dim+off_y) * Input_X_dim*Input_depth_dim
                                                                          + (x*Stride_X_dim+off_x) * Input_depth_dim
                                                                          + d;
                                    unsigned long long int filter_subscript = m * Mask_Y_dim*Mask_X_dim*Input_depth_dim
                                                                              + off_y * Mask_X_dim*Input_depth_dim
                                                                              + off_x*Input_depth_dim
                                                                              + d;

                                    float s = in_FP[in_subscript];
                                    float w = filter_FP[filter_subscript];
                                    temp = temp + s * w;


                                }
                            }
                        }


                        unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                                                               y * (Output_depth_dim * Output_X_dim) +
                                                               x * Output_depth_dim
                                                               + m;

                        temp+=bias;
                        out_to_compare_with_FP[out_subscript] = Relu_float(temp);

                    }
                }
            }
        }
    }

    //printf("\n from unopt %d %d ",out_to_compare_with[0],out_to_compare_with[1]);
    return 0;

}

float Relu_float(const float temp){


    if (temp<0.0f)
        return 0.0f;
    else
        return temp;

}






