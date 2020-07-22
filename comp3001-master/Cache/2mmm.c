
#include <stdio.h> //library for printf() 
#include <time.h>	/* for clock_gettime */
#include <stdint.h>	/* for uint64 definition */

#define N 500  //arrays input size
#define BILLION 1000000000L
#define TIMES 1


//In C, all the routines must be declared
void initialize();
void mmm();


int A[N][N], B[N][N], C[N][N], E[N][N];


int main( ) {


struct timespec start, end; //timers
uint64_t diff;

initialize();

/* measure monotonic time */
clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (int t=0;t<TIMES;t++)
 mmm();

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
  A[i][j]=(i+j)%1000;
  B[i][j]=(i-j)%1000;
  C[i][j]=0;
  E[i][j]=0;

}

}

void mmm(){

int i,j,k;

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  for (k=0;k<N;k++) 
 C[i][j]+=A[i][k]*B[k][j];
 


for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  for (k=0;k<N;k++) 
 E[i][j]+=C[i][k]*B[k][j];

}





