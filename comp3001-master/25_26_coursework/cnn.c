/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

/*
THIS IS AN IMPLEMENTATION OF A SIMPLE CNN MODEL USING 17 LAYERS. 
COMPILE USING gcc cnn.c -o p -O3 -fopenmp
*/

//CNN architecture with 17 layers is as follows:
//LAYER #1 - conv2d [batch, 64, 64, 3, 1, 1, 3, 3, 32]
//LAYER #2 - ReLU 	[batch, 62, 62, 32]
//LAYER #3 - pooling 2,2 [batch, 32, 31, 31]
//LAYER #4 - conv2d [batch, 31, 31, 32, 1, 1, 3, 3, 64]
//LAYER #5 - ReLU
//LAYER #6 - pooling 2,2 [batch, 64, 14, 14]
//LAYER #7 - conv2d [batch, 14, 14, 64, 1, 1, 3, 3, 128]
//LAYER #8 - ReLU
//LAYER #9 - conv2d [batch, 12, 12, 128, 1, 1, 3, 3, 256]
//LAYER #10 - ReLU
//LAYER #11 - conv2d [batch, 10, 10, 256, 1, 1, 3, 3, 256]
//LAYER #12 - ReLU
//LAYER #13 - pooling 2,2 [batch, 256, 4, 4]
//LAYER #14 - flatten [batch, 256 * 4 * 4 = 4096]
//LAYER #15 - Fully Connected [batch,4096,1024]
//LAYER #16 - 2d ReLU [batch, 1024]
//LAYER #17 - Fully Connected [batch, 1024,10]


#include <stdio.h>
#include <float.h> 
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


float * tensor1; //pointer to tensor
float * tensor2; //pointer to tensor 
float * tensor3; //pointer to tensor 
float * tensor4; //pointer to tensor 
float * tensor5; //pointer to tensor 
float * tensor6; //pointer to tensor 
float * tensor7; //pointer to tensor 
float * tensor8; //pointer to tensor 
float * tensor9; //pointer to tensor 
float * tensor10; //pointer to tensor 
float * tensor11; //pointer to tensor 
float * filter1; //pointer to filter array
float * filter2; //pointer to filter array
float * filter3; //pointer to filter array
float * filter4; //pointer to filter array
float * filter5; //pointer to filter array
float * filter6; //pointer to filter array
float * filter7; //pointer to filter array
float *bias1;
float *bias2;
float *bias3;
float *bias4;
float *bias5;
float *bias6;
float *bias7;

#define MAX(a, b) ((a) > (b) ? (a) : (b))



void conv_2d(float ** in, float ** filter, float **bias, float ** out, unsigned int B,unsigned int Yin, unsigned int Xin,unsigned int D,unsigned int StrideY,unsigned int StrideX, unsigned int MaskY, unsigned int MaskX, unsigned int M){

    float temp;

    unsigned int X=(Xin-(MaskX-StrideX)) / StrideX;
    unsigned int Y=(Yin-(MaskY-StrideY)) / StrideY;

    for (unsigned int b = 0; b < B; b++) { //batch
        for(unsigned int m = 0; m < M; m++){
                for (unsigned int y = 0; y < Y; y++) {			//Output height
                    for (unsigned int x = 0; x < X; x++) {			//Output Width
                        temp = 0.0f;
                        for (unsigned int off_y = 0; off_y < MaskY; off_y++) {
                            for (unsigned int off_x = 0; off_x < MaskX; off_x++) {
                                for(unsigned int d = 0; d < D; d++) {

                                    unsigned int in_subscript = b * (Yin * Xin * D)
                                                                          + (y*StrideY+off_y) * Xin * D
                                                                          + (x*StrideX+off_x) * D
                                                                          + d;
                                    unsigned int filter_subscript = m * MaskY * MaskX * D
                                                                              + off_y * MaskX * D
                                                                              + off_x * D
                                                                              + d;

                                    float s = (*in)[in_subscript];
                                    float w = (*filter)[filter_subscript];
                                    temp += s * w;


                                }
                            }
                        }

                        unsigned int out_subscript = b * (M * Y * X) +
                                                               y * (M * X) +
                                                               x * M
                                                               + m;

                        (*out)[out_subscript] = temp + (*bias)[m];

                    }
                }
            }
         }


/*
    //In case you find the above implementation complicated, it is equivalent to the code below. 
    //So, when you are thinking about optimization perhaps it is easier to study this version of the code instead which is equivalent
    
    for (unsigned int b = 0; b < B; b++) { 
        for(unsigned int m = 0; m < M; m++){
                for (unsigned int y = 0; y < Y; y++) {			
                    for (unsigned int x = 0; x < X; x++) {			
                        temp = 0.0f;
                        for (unsigned int off_y = 0; off_y < MaskY; off_y++) {
                            for (unsigned int off_x = 0; off_x < MaskX; off_x++) {
                                for(unsigned int d = 0; d < D; d++) {
                                    temp += in[b][y][x][d] * filter[m][off_y][off_x][d];
                                }
                            }
                        }
                        out[b][y][x][m] = temp + bias[m];
                    }
                }
            }
        }
    */

}




void max_pooling(float** input, float** output,
                      int batch_size, int in_height, int in_width, int channels,
                      int pool_size, int stride) {
    
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width  = (in_width - pool_size) / stride + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                for (int c = 0; c < channels; c++) {

                    float max_val = FLT_MIN;
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int h = oh * stride + ph;
                            int w = ow * stride + pw;
                            int in_index = ((b * in_height + h) * in_width + w) * channels + c;
                            max_val = MAX(max_val, (*input)[in_index]);
                        }
                    }

                    int out_index = ((b * out_height + oh) * out_width + ow) * channels + c;
                    (*output)[out_index] = max_val;
                }
            }
        }
    }
}



// Fully connected layer function - the same weights array is used for each batch
void FC(float** input, float** weights, float** bias, float** output, int batch_size, int input_dim, int output_dim) {

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < output_dim; i++) {
        
            float sum = (*bias)[i];
            
            for (int j = 0; j < input_dim; j++) {
                sum += (*weights)[i * input_dim + j] * (*input)[b * input_dim + j];
            }
            
            (*output)[b * output_dim + i] = sum;
        }
    }
    
    /*
    //In case you find the above implementation complicated, it is equivalent to the code below. 
    //So, when you are thinking about optimization perhaps it is easier to study this version of the code instead which is equivalent
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < output_dim; i++) {  
            for (int j = 0; j < input_dim; j++) {
                output[b][i] += weights[i][j] * input[b][j] + bias[i];
            }
            
        }
    }
    */
    
}



void ReLU(float** input, float** output,
               int batch_size, int height, int width, int channels) {
    int index = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    index = ((b * height + h) * width + w) * channels + c;
                    
                    if ( (*input)[index] > 0 )
                     (*output)[index] = (*input)[index];
                    else 
                      (*output)[index] = 0.0f; 

                }
            }
        }
    }
    
        /*
    //In case you find the above implementation complicated, it is equivalent to the code below. 
    //So, when you are thinking about optimization perhaps it is easier to study this version of the code instead which is equivalent
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {                   
                    if ( input[b][h][w][c] > 0 )
                     output[b][h][w][c] = input[b][h][w][c];
                    else 
                     output[b][h][w][c] = 0.0f; 
                }
            }
        }
    */
    
    
}



void create_load_bias(float** bias, const unsigned int M){

    *bias = (float *) malloc( M * sizeof(float));
    if (*bias==NULL) {
        printf("\nerror with malloc allocating bias array");
        exit(EXIT_FAILURE);
    }

    for (unsigned int i=0; i<M; i++){
        (*bias)[i]=(float) (i % 99) + 0.41f;
        //*(bias_array_FP+i)=((float) (rand() % 5)) + 1;
        //  *(bias_array_FP+i)=0.0f;
        // printf("  %d",*(in+i));
    }



}


void create_load_filter(float** filter, const unsigned int M, const unsigned int D, const unsigned int MaskX, const unsigned int MaskY){

    unsigned int filter_size= M*D*MaskX*MaskY;
    unsigned int y,x,m,d,offset,cnt=0;

    *filter = (float *) malloc( filter_size * sizeof(float));
    if (*filter==NULL) {
        printf("\nerror with malloc allocating filter array");
        exit(EXIT_FAILURE);
    }



    //read the filter array
    for (m=0;m<M;m++)
        for (y=0;y<MaskY;y++)
            for (x=0;x<MaskX;x++){
                //printf("\n");
                for (d=0;d<D;d++){
                    offset=m * MaskY*MaskX*D +
                           y * MaskX*D +
                           x*D + d;

                    (*filter)[offset]= (float) (m-y-x+d % 33) + 0.97f;
                      cnt++;
                }}

}


void create_load_input_tensor(float** input, unsigned int B,unsigned int Yin, unsigned int Xin,unsigned int D,unsigned int StrideY,unsigned int StrideX, unsigned int MaskY, unsigned int MaskX, unsigned int M){

    unsigned long long int input_size= (unsigned long long int) B * D * Yin * Xin;
    
    unsigned int X=(Xin-(MaskX-StrideX)) / StrideX;
    unsigned int Y=(Yin-(MaskY-StrideY)) / StrideY;
    
    unsigned long long int output_size=(unsigned long long int) B * M * Y * X;
    unsigned long long int in_subscript,out_subscript;

    *input = (float *) malloc( input_size * sizeof(float));
    if (*input==NULL) {
        printf("\nerror with malloc allocating input array");
        exit(EXIT_FAILURE);
    }


    for (int b = 0; b < B; b++)
        for (int y = 0; y < Yin; y++)
            for (int x = 0; x < Xin; x ++)
                for (unsigned int d = 0; d < D; d++) {
                    in_subscript = (unsigned long long int) b * (Yin * Xin * D)
                                   + (y ) * Xin * D
                                   + (x ) * D + d;

                    (*input)[in_subscript] =  ( (float) ((b-y-x+d) % 50) ) + 0.73f;
                }


}



void create_load_output_tensor(float** output, unsigned int B,unsigned int Y, unsigned int X, unsigned int M){
   
    unsigned long long int output_size=(unsigned long long int) B * M * Y * X;
    unsigned long long int in_subscript,out_subscript;


    *output = (float *) malloc( output_size * sizeof(float));
    if (*output==NULL) {
        printf("\nerror with malloc allocating output array");
        exit(EXIT_FAILURE);
    }


    for (int b = 0; b < B; b++)
        for (int y = 0; y < Y; y++)
            for (int x = 0; x < X; x ++)
                for (unsigned int m = 0; m < M; m++) {
                    out_subscript = (unsigned long long int) b * (M * Y * X) +
                                    y * (M * X) +
                                    x * M
                                    + m;

                    (*output)[out_subscript]=0.0f;
                }


}


void deallocate_memory(){

free(tensor1);
free(tensor2);
free(tensor3);
free(tensor4);
free(tensor5);
free(tensor6);
free(tensor7);
free(tensor8);
free(tensor9);
free(tensor10);
free(tensor11);

free(filter1);
free(filter2);
free(filter3);
free(filter4);
free(filter5);
free(filter6);
free(filter7);

free(bias1);
free(bias2);
free(bias3);
free(bias4);
free(bias5);
free(bias6);
free(bias7);

}


void cnn(){

unsigned int batch_size=32;//INPUT

//LAYER #1 conv2d [batch, 64, 64, 3, 1, 1, 3, 3, 32]
create_load_input_tensor(&tensor1,batch_size,64,64,3,1,1,3,3,32);
create_load_output_tensor(&tensor2,batch_size,62,62,32);
create_load_filter(&filter1,32,3,3,3);
create_load_bias(&bias1,32);
conv_2d(&tensor1,&filter1,&bias1,&tensor2,batch_size,64,64,3,1,1,3,3,32);

//LAYER #2 - ReLU 	[batch, 62, 62, 32]
ReLU(&tensor2,&tensor2,batch_size,62,62,32);

//LAYER #3 - pooling 2,2 [batch, 32, 31, 31]
create_load_output_tensor(&tensor3,batch_size,31,31,32);
max_pooling(&tensor2,&tensor3, batch_size, 62, 62, 32,2,2);

//LAYER #4 - conv2d [batch, 31, 31, 32, 1, 1, 3, 3, 64]
create_load_output_tensor(&tensor4,batch_size,29,29,64);
create_load_filter(&filter2,64,32,3,3);
create_load_bias(&bias2,64);
conv_2d(&tensor3,&filter2,&bias2,&tensor4,batch_size,31,31,32,1,1,3,3,64);

//LAYER #5 - ReLU
ReLU(&tensor4,&tensor4,batch_size,29,29,64);

//LAYER #6 - pooling 2,2 [batch, 64, 14, 14]
create_load_output_tensor(&tensor5,batch_size,14,14,64);
max_pooling(&tensor4,&tensor5, batch_size, 29, 29, 64,2,2);

//LAYER #7 - conv2d [batch, 14, 14, 64, 1, 1, 3, 3, 128]
create_load_output_tensor(&tensor6,batch_size,12,12,128);
create_load_filter(&filter3,128,64,3,3);
create_load_bias(&bias3,128);
conv_2d(&tensor5,&filter3,&bias3,&tensor6,batch_size,14,14,64,1,1,3,3,128);

//LAYER #8 - ReLU
ReLU(&tensor6,&tensor6,batch_size,12,12,128);

//LAYER #9 - conv2d [batch, 12, 12, 128, 1, 1, 3, 3, 256]
create_load_output_tensor(&tensor7,batch_size,10,10,256);
create_load_filter(&filter4,256,128,3,3);
create_load_bias(&bias4,256);
conv_2d(&tensor6,&filter4,&bias4,&tensor7,batch_size,12,12,128,1,1,3,3,256);

//LAYER #10 - ReLU
ReLU(&tensor7,&tensor7,batch_size,10,10,256);

//LAYER #11 - conv2d [batch, 10, 10, 256, 1, 1, 3, 3, 256]
create_load_output_tensor(&tensor8,batch_size,8,8,256);
create_load_filter(&filter5,256,256,3,3);
create_load_bias(&bias5,256);
conv_2d(&tensor7,&filter5,&bias5,&tensor8,batch_size,10,10,256,1,1,3,3,256);

//LAYER #12 - ReLU
ReLU(&tensor8,&tensor8,batch_size,8,8,256);

//LAYER #13 - pooling 2,2 [batch, 256, 4, 4]
create_load_output_tensor(&tensor9,batch_size,4,4,256);
max_pooling(&tensor8,&tensor9, batch_size, 8, 8, 256,2,2);

//LAYER #14 - flatten [batch, 256 * 4 * 4 = 4096]
//the tensor is stored as an 1D array in memory, thus no action is needed. 

//LAYER #15 - FC [batch,4096,1024]
create_load_output_tensor(&tensor10,batch_size,1,1,1024);
create_load_filter(&filter6,4096,1024,1,1);
create_load_bias(&bias6,1024);
FC(&tensor9, &filter6, &bias6, &tensor10, batch_size, 4096, 1024);

//LAYER #16 - 2d ReLU [batch, 1024]
ReLU(&tensor10,&tensor10,batch_size,1,1,1024);

//LAYER #17 - FC [batch, 1024,10]
create_load_output_tensor(&tensor11,batch_size,1,1,10);
create_load_filter(&filter7,1024,10,1,1);
create_load_bias(&bias7,10);
FC(&tensor10, &filter7, &bias7, &tensor11, batch_size, 1024, 10);

deallocate_memory();

}



int main() {

double start_time, run_time;


start_time = omp_get_wtime();

cnn();

run_time = (omp_get_wtime() - start_time);
printf("\n\nThe model's latency is %f seconds\n",  run_time);

    
return 0;
}


