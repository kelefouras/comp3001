/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc MVM_parallel.c -o p -O2 -fopenmp -lm

#include <stdio.h>
#include <stdlib.h> //this library allows for malloc to run
 #include <sys/time.h>
#include <stdint.h>	/* for uint64 definition */
#include <pthread.h>
#include <omp.h>
#include <math.h>

#define N 1000 //array size
#define BILLION 1000000000L
#define TIMES 1
#define EPSILON 0.00001

void init(float *y, float *a,  float *x);
void MVM_serial(float *y, float *a,  float *x);
unsigned short int equal(float const a, float const b);
unsigned short int Compare_MVM(const float *y, const float *a,  const float *x);

float test[N];



int main () {
int i,it;
float *x, *y, *a; // These pointers will hold the base addresses of the memory blocks created 

struct timespec start, end; //for timers
uint64_t diff;

/* Dynamically allocate memory storage for the arrays */
x = (float*) malloc (N * sizeof(float));
    if (x == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0); //terminates the process immediately
    } 

y = (float*) malloc (N * sizeof(float));
    if (y == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    } 

a = (float*) malloc (N * N * sizeof(float));
    if (a == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    }  

init(y,a,x); //initialize the arrays



clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (it=0; it< TIMES; it++)
 MVM_serial(y,a,x); //execute the main routine


clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);



if (Compare_MVM(y,a,x) == 0)
	printf("\n\n\r ----- output is correct -----\n\r");
else
	printf("\n\n\r ---- output is INcorrect -----\n\r");

free (x); //deallocate the memory
free (y); //deallocate the memory
free (a); //deallocate the memory

return 0;

}   


void init(float *y, float *a,  float *x){

float e=0.63;
int i,j;

for (i=0; i<N; i++) {
  x[i]=(float) (i%100)+e; 
  y[i]=0.0; 
  }

for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
  a[N*i+j]=(float) (j%30)-e;

}



void MVM_serial(float *y, float *a,  float *x){

int i,j;

for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
 y[i]+=a[N*i+j]*x[j];


}





unsigned short int Compare_MVM(const float *y, const float *a,  const float *x) {

	for (int i = 0; i < N; i++) {
                test[i]=0.0;
		for (int j = 0; j < N; j++) {
			test[i] += a[N*i+j] * x[j];
		}
	}

	for (int j = 0; j < N; j++)
		if (equal(y[j], test[j]) == 1) {
			printf("\n j=%d %f %f\n", j, test[j],y[j]);
			return 1;
		}

	return 0;
}


unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp/b) < EPSILON)
		return 0; //success
	else
		return 1;
}




