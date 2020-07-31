/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc dot_product_parallel.c -o p -O2 -fopenmp -fopt-info-vec-optimized

#include <stdio.h>
#include <stdlib.h> //this library allows for malloc to run
 #include <sys/time.h>
#include <stdint.h>	/* for uint64 definition */
#include <pthread.h>
#include <omp.h>

void init(double *a, double *b);
double dot_prod_serial(double *a, double *b);

#define N 1000000 //array size
#define BILLION 1000000000L
#define TIMES 1000

int main () {
int i,it;
double *a, *b; // These pointers will hold the base addresses of the memory blocks created 
double sum;

struct timespec start, end; //for timers
uint64_t diff;

/* Dynamically allocate memory storage for the arrays */
a = (double*) malloc (N * sizeof(double));
    if (a == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0); //terminates the process immediately
    } 

b = (double*) malloc (N * sizeof(double));
    if (b == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    } 
 

init(a,b); //initialize the arrays



clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (it=0; it< TIMES; it++)
sum=dot_prod_serial(a,b); //execute the main routine

clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);

printf ("Done. Serial version: sum  =  %f \n", sum);


free (a); //deallocate the memory
free (b); //deallocate the memory

return 0;

}   


void init(double *a, double *b){

double e=0.12134;
/* Initialize dot product vectors */
for (int i=0; i<N; i++) {
  a[i]=(double) (i%100)+e; 
  b[i]=a[i]+e; 
  }

}


/* Perform the dot product */
double dot_prod_serial(double *a, double *b){

double sum = 0.0;
int i;

for (i=0; i<N; i++) 
  {
    sum += (a[i] * b[i]);
  }


return sum;
}





