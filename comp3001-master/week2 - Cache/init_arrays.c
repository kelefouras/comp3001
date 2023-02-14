/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //library for printf() 
#include <time.h>	/* for clock_gettime */
#include <stdint.h>	/* for uint64 definition */

#define N 1000  //arrays input size
#define BILLION 1000000000L

//In C, all the routines must be declared
void initialize();

int A[N][N];


int main( ) {


struct timespec start, end; //timers
uint64_t diff;

/* measure monotonic time */
clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

initialize();

clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);

return 0; //normally, by returning zero, we mean that the program ended successfully. 
}



void initialize(){

int i,j;

for (i=0;i<N;i++)
 for (j=0;j<N;j++){
  A[j][i]=i+j;

}

}





