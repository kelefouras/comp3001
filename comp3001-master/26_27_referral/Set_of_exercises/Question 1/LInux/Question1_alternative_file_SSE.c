/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc Question1_alternative_file.c -o p -O3 -D_GNU_SOURCE  -march=native -msse -lm -D_GNU_SOURCE -Wall -fopenmp

#include <math.h>
#include <stdio.h>
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




#define N 16384 //arrays input size
#define TIMES 1 //times to run


//In C, all the routines must be declared
void step1();
void step2();
float step3();
void initialize();

float A[N][N],X[N],Y[N] __attribute__((aligned(16)));


int main( ) {


double start,end;
float reduction;
unsigned int i;

//the following code binds this thread to code number 0. Without this code, the OS will tongle the thread among the cores, to reduce heat dissipation
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(0,&mask);
if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
   printf("WARNING: Could not set CPU Affinity, continuing...\n");




start=omp_get_wtime();

for (i=0;i<TIMES;i++){

 initialize();
 step1();
 step2();
 reduction=step3();
 
}

end=omp_get_wtime();

printf("\n output is %e \n",reduction);
printf("\ntime elapsed is %f secs while time per run is %f secs\n",end-start, (end-start)/TIMES);


return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize(){

unsigned int i,j;

for (i=0;i<N;i++){
X[i]=(float) (i%4)+0.01;
Y[i]=0.0;
}

for (i=0;i<N;i++)
for (j=0;j<N;j++){
A[i][j]=(float) (i%3)+(j%6)-0.02;
}


}


void step1(){

unsigned int i;
__m128 const1,const2,const3,const4,const5,const6,const7,const8,a,tmp;

const1 = _mm_set_ps(0.91,0.21,0.24,0.84);
const2 = _mm_set_ps(0.92,0.22,0.25,0.85);
const3 = _mm_set_ps(0.93,0.23,0.26,0.86);
const4 = _mm_set_ps(0.94,0.24,0.27,0.87);
const5 = _mm_set_ps(0.95,0.25,0.28,0.88);
const6 = _mm_set_ps(0.96,0.26,0.29,0.89);
const7 = _mm_set_ps(0.97,0.23,0.45,0.76);
const8 = _mm_set_ps(0.98,0.43,0.43,0.77);

for (i=0;i<N;i+=4){
a = _mm_load_ps(&X[i]);
tmp = _mm_setzero_ps();
tmp += _mm_mul_ps(a, const1);
tmp += _mm_mul_ps(a, const2);
tmp += _mm_mul_ps(a, const3);
tmp += _mm_mul_ps(a, const4);
tmp += _mm_mul_ps(a, const5);
tmp += _mm_mul_ps(a, const6);
tmp += _mm_mul_ps(a, const7);
tmp += _mm_mul_ps(a, const8);
_mm_store_ps(&X[i], tmp);
}


}


void step2(){

unsigned int i,j;
__m128 const1,const2,const3,const4,const5,const6,const7,const8,a,tmp;
const1 = _mm_set_ps(0.91,0.21,0.24,0.84);
const2 = _mm_set_ps(0.92,0.22,0.25,0.85);
const3 = _mm_set_ps(0.93,0.23,0.26,0.86);
const4 = _mm_set_ps(0.94,0.24,0.27,0.87);
const5 = _mm_set_ps(0.95,0.25,0.28,0.88);
const6 = _mm_set_ps(0.96,0.26,0.29,0.89);
const7 = _mm_set_ps(0.97,0.23,0.45,0.76);
const8 = _mm_set_ps(0.98,0.43,0.43,0.77);

for (i=0;i<N;i++){
 for (j=0;j<N;j+=4){
a = _mm_load_ps(&A[i][j]);
tmp = _mm_setzero_ps();
tmp += _mm_mul_ps(a, const1);
tmp += _mm_mul_ps(a, const2);
tmp += _mm_mul_ps(a, const3);
tmp += _mm_mul_ps(a, const4);
tmp += _mm_mul_ps(a, const5);
tmp += _mm_mul_ps(a, const6);
tmp += _mm_mul_ps(a, const7);
tmp += _mm_mul_ps(a, const8);
_mm_store_ps(&A[i][j], tmp);
} }


}


float step3(){

unsigned int i,j;
float reduction=0.0;

for (i=0;i<N;i++){
 for (j=0;j<N;j++){
  Y[i] += A[i][j] * X[j];
} }

 for (j=0;j<N;j+=8){
  reduction+=Y[j];
  }
  
  return reduction;
  
}






