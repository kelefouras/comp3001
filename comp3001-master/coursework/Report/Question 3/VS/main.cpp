/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

/*
This is a software application that executes the convolution and ReLU layer in a Deep Neural Network.
The layer input parameters are specified in  void read_layer_dimensions(). Feel free to change the parameters if you want, especially the batch size. If your code runs very fast, then increase the batch size to get more accurate execution time values.

Your task is to replace line #87 by another faster routine with the same functionality.

*/

#include "convolution_layer_2D.h"
#include <windows.h> //this library is needed for pause() function

void read_layer_dimensions();

int load_create_input_output_array_FP();
int load_filter_array_FP();
void load_bias_FP();
void deallocate_FP();
void compare_output_result_FP();
unsigned short int equal(float const a, float const b);

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


float* in_FP; //pointer to input array
float* in_layout_FP; //pointer to input array
float* filter_FP; //pointer to filter array
float* out_FP; //pointer to output array
float* out_to_compare_with_FP; //pointer to output array to compare with
float* bias_array_FP;




#define EPSILON 0.000001


int main() {

    double start_time, run_time;

    read_layer_dimensions();



    load_bias_FP();
    load_create_input_output_array_FP();
    load_filter_array_FP();
    unoptimized_layer_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);


    start_time = omp_get_wtime();

    //---------this routine will be replaced by your routine------------
    unoptimized_layer_FP(in_FP, filter_FP, bias_array_FP, out_FP);

    run_time = (omp_get_wtime() - start_time);

    double FLOPS = (double)Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim;
    FLOPS = (FLOPS * ( (double) 2 * Mask_Y_dim * Mask_X_dim * Input_depth_dim + 1)) / run_time;

    printf("\n\nTime = %.3e seconds", run_time);
    printf(" or %.0f mseconds", run_time * 1000);//printf time in msecs
    printf("\nGiga FLOPS achieved: %.0f\n", (double)FLOPS / 1000000000);//print Giga FLOPS


    compare_output_result_FP();


    deallocate_FP();


    system("pause"); //this command does not let the output window to close

    return 0;
}


void compare_output_result_FP() {
    
    for (unsigned long long int i = 0; i < (unsigned long long int) Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim; i++) {
        if (equal(out_FP[i], out_to_compare_with_FP[i]) == 1) {
            printf("\n wrong values (%llu): %f %f", i, out_FP[i], out_to_compare_with_FP[i]);

        }
    }
}

unsigned short int equal(float const a, float const b) {
    float temp = a - b;

    if (b == 0.0f) {//cannot divide with zero
        if (a == 0.0f) {
            return 0;//success
        }
        else {
            return 1;
        }
    }
    else {

        if ((fabs(temp) / fabs(b)) < EPSILON) {
            return 0; //success
        }
        else {
            return 1;
        }
    }
}





void read_layer_dimensions() {


    Input_Output_batch_dim = 20;
    Input_Y_dim = 54;
    Input_X_dim = 54;
    Input_depth_dim = 256;

    Stride_Y_dim = 1;
    Stride_X_dim = 1;

    Mask_Y_dim = 3;
    Mask_X_dim = 3;

    Output_depth_dim = 128;
    Output_X_dim = (Input_X_dim - (Mask_X_dim - Stride_X_dim)) / Stride_X_dim;
    Output_Y_dim = (Input_Y_dim - (Mask_Y_dim - Stride_Y_dim)) / Stride_Y_dim;

    unsigned long long int In_size = (unsigned long long int) Input_Output_batch_dim * Input_X_dim * Input_Y_dim * Input_depth_dim;
    unsigned long long int Filter_size = (unsigned long long int) Input_depth_dim * Mask_X_dim * Mask_Y_dim * Output_depth_dim;
    unsigned long long int Out_size = (unsigned long long int) Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim;






    printf("\n Layer dimensions are read");
    printf("\n Input dims (batch,y,x,depth) = (%d, %d, %d, %d)       - Size in Elements = %llu", Input_Output_batch_dim, Input_Y_dim, Input_X_dim, Input_depth_dim, In_size);
    printf("\n Filter dims (m,y,x,depth) = (%d, %d, %d, %d)           - Size in Elements = %llu", Output_depth_dim, Mask_Y_dim, Mask_X_dim, Input_depth_dim, Filter_size);
    printf("\n Output dims (batch,y,x,out_depth) = (%d, %d, %d, %d) - Size in Elements = %llu", Input_Output_batch_dim, Output_Y_dim, Output_X_dim, Output_depth_dim, Out_size);

}





void load_bias_FP() {

    bias_array_FP = (float *) _mm_malloc(Output_depth_dim * sizeof(float), 64);
    if (bias_array_FP == NULL) {
        printf("\nerror with malloc allocating bias array");
        exit(EXIT_FAILURE);
    }


    for (unsigned int i = 0; i < Output_depth_dim; i++) {
        *(bias_array_FP + i) = ((float)(rand() % 5)) + 1;
        //  *(bias_array_FP+i)=0.0f;
        // printf("  %d",*(in+i));
    }



}




//in[] is stored into memory like that : in[Input_Output_batch_dim] [Input_Y_dim] [Input_X_dim] [Input_depth_dim] ;
//out[] is stored into memory like that : out[Input_Output_batch_dim] [Output_Y_dim] [Output_X_dim] [Output_depth_dim] ;
int load_create_input_output_array_FP() {

    unsigned long long int input_size = (unsigned long long int) Input_Output_batch_dim * Input_depth_dim * Input_Y_dim * Input_X_dim;
    unsigned long long int output_size = (unsigned long long int) Input_Output_batch_dim * Output_depth_dim * Output_Y_dim * Output_X_dim;
    unsigned long long int in_subscript, out_subscript;

    in_FP = (float*)_mm_malloc(input_size * sizeof(float), 64);
    if (in_FP == NULL) {
        printf("\nerror with malloc allocating input array");
        exit(EXIT_FAILURE);
    }


    for (unsigned int b = 0; b < Input_Output_batch_dim; b++)
        for (unsigned int y = 0; y < Input_Y_dim; y++)
            for (unsigned int x = 0; x < Input_X_dim; x++)
                for (unsigned int d = 0; d < Input_depth_dim; d++) {
                    in_subscript = (unsigned long long int) b * Input_Y_dim * Input_X_dim * Input_depth_dim + (unsigned long long int) y*Input_X_dim * Input_depth_dim + (unsigned long long int) x*Input_depth_dim + d;

                    in_FP[in_subscript] = ((float)(d % 50)) + 0.73f;
                    //in_FP[in_subscript] = ((float) (rand() % 50) ) +0.73f;
                    // printf("  %d",*(in+i));
                }


    out_FP = (float*)_mm_malloc(output_size * sizeof(float), 64);
    if (out_FP == NULL) {
        printf("\nerror with malloc allocating output array");
        exit(EXIT_FAILURE);
    }


    out_to_compare_with_FP = (float*) _mm_malloc(output_size * sizeof(float), 64);
    if (out_to_compare_with_FP == NULL) {
        printf("\nerror with malloc allocating output array to compare with");
        exit(EXIT_FAILURE);
    }



    for (unsigned int b = 0; b < Input_Output_batch_dim; b++)
        for (unsigned int y = 0; y < Output_Y_dim; y++)
            for (unsigned int x = 0; x < Output_X_dim; x++)
                for (unsigned int m = 0; m < Output_depth_dim; m++) {
                    out_subscript = (unsigned long long int) b * Output_depth_dim * Output_X_dim * Output_Y_dim +
                        (unsigned long long int) y * Output_depth_dim * Output_X_dim +
                        (unsigned long long int) x * Output_depth_dim
                        + m;

                    out_to_compare_with_FP[out_subscript] = 0.0f;

                    out_FP[out_subscript] = 0.0f;
                }



    // printf("\n Input / Output arrays are created. Input is loaded. \n");
    return 0;
}




void deallocate_FP() {

    _mm_free(in_FP);
    _mm_free(out_FP);

    _mm_free(out_to_compare_with_FP);

    _mm_free(bias_array_FP);

    _mm_free(filter_FP);



}




// filter array is stored into memory tile-wise
int load_filter_array_FP() {

    unsigned int filter_size = Mask_X_dim * Mask_Y_dim * Input_depth_dim * Output_depth_dim;
    unsigned int y, x, m, d, offset, cnt = 0;

    filter_FP = (float *) _mm_malloc(filter_size * sizeof(float), 64);
    if (filter_FP == NULL) {
        printf("\nerror with malloc allocating filter array");
        exit(EXIT_FAILURE);
    }



    //read the filter array
    for (m = 0; m < Output_depth_dim; m++)
        for (y = 0; y < Mask_Y_dim; y++)
            for (x = 0; x < Mask_X_dim; x++) {
                //printf("\n");
                for (d = 0; d < Input_depth_dim; d += 2) {
                    offset = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim +
                        y * Mask_X_dim * Input_depth_dim +
                        x * Input_depth_dim + d;

                    filter_FP[offset] = ((rand() % 8) + 0.973f);
                    filter_FP[offset + 1] = -((rand() % 8) + 0.973f);
                    // printf("\n %d, %d",filter_FP[offset],filter_FP[offset+1]);
                    cnt++;
                }
            }


    //printf("\n Filter array is created and loaded. \n");
    return 0;
}



