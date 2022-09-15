/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//COMPILE WITH
//gcc q1b.c -o p -O3 -march=native  -D_GNU_SOURCE -g
 
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>	/* for uint64 definition */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <sys/time.h>
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>


#define TIMES 1
#define N 4096

void writedata();


float  X[N],Y[N]  __attribute__((aligned(64)));
float A[N][N], Atr[N*N] __attribute__((aligned(64)));


int main(){


int i,it,ii,j,j0,i3,j3,temp,times,count=0;
int tem,iii,jj,jjj,k,c;


time_t start1, end1;
struct timeval start2, end2;

//----------------set the current thread to core 0 only----------------
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(0,&mask);
if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
       printf("WARNING: Could not set CPU Affinity, continuing...\n");


//-------------------initialize ---------------------------
for (i=0;i<N;i++){
Y[i]=0.0;
X[i]= (float) ((i%99)/3);
}


for (i=0;i<N;i++)
for (j=0;j<N;j++)
A[i][j]=(float) (((i+j)%99)/3);


start1 = clock();
gettimeofday(&start2, NULL);




//-------------------main kernel ---------------------------
for (int it=0;it<TIMES;it++)
 for (i=0;i<N;i++)
  for (j=0;j<N;j++)
   Y[i] += A[i][j] * X[j];





end1 = clock();
gettimeofday(&end2, NULL);
printf(" clock() method: %ldms\n", (end1 - start1) / (CLOCKS_PER_SEC/1000));
printf(" gettimeofday() method: %ldms\n", (end2.tv_sec - start2.tv_sec) *1000 + (end2.tv_usec - start2.tv_usec)/1000);



printf("\n The first and last values are %f %f\n",Y[0],Y[N-1]);


return 0;
}





