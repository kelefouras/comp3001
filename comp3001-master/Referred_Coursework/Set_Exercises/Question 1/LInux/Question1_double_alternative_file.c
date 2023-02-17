/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


//compile with gcc Question1_double_alternative_file.c -o p -O3 -D_GNU_SOURCE  -march=native -mavx -lm -D_GNU_SOURCE -Wall -fopenmp

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
double step3();
void initialize();

double A[N][N],X[N],Y[N] __attribute__((aligned(64)));


int main( ) {


double start,end;
double reduction;
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
X[i]=(double) (i%4)+0.01;
Y[i]=0.0;
}

for (i=0;i<N;i++)
for (j=0;j<N;j++){
A[i][j]=(double) (i%3)+(j%6)-0.02;
}


}


void step1(){

unsigned int i;
__m256d const1,const2,const3,const4,const5,const6,const7,const8,a,tmp;
const1 = _mm256_set_pd(-0.94,-0.53,-0.76,-0.64);
const2 = _mm256_set_pd(-0.95,-0.54,-0.73,-0.67);
const3 = _mm256_set_pd(-0.96,-0.55,-0.72,-0.66);
const4 = _mm256_set_pd(-0.97,-0.56,-0.74,-0.65);
const5 = _mm256_set_pd(-0.98,-0.57,-0.71,-0.63);
const6 = _mm256_set_pd(-0.99,-0.58,-0.75,-0.62);
const7 = _mm256_set_pd(-0.98,-0.54,-0.74,-0.62);
const8 = _mm256_set_pd(-0.94,-0.12,-0.71,-0.61);

for (i=0;i<N;i+=4){
a = _mm256_load_pd(&X[i]);
tmp = _mm256_setzero_pd();
tmp += _mm256_mul_pd(a, const1);
tmp += _mm256_mul_pd(a, const2);
tmp += _mm256_mul_pd(a, const3);
tmp += _mm256_mul_pd(a, const4);
tmp += _mm256_mul_pd(a, const5);
tmp += _mm256_mul_pd(a, const6);
tmp += _mm256_mul_pd(a, const7);
tmp += _mm256_mul_pd(a, const8);
_mm256_store_pd(&X[i], tmp);
}


}


void step2(){

unsigned int i,j;
__m256d const1,const2,const3,const4,const5,const6,const7,const8,a,tmp;
const1 = _mm256_set_pd(-0.94,-0.53,-0.76,-0.64);
const2 = _mm256_set_pd(-0.95,-0.54,-0.73,-0.67);
const3 = _mm256_set_pd(-0.96,-0.55,-0.72,-0.66);
const4 = _mm256_set_pd(-0.97,-0.56,-0.74,-0.65);
const5 = _mm256_set_pd(-0.98,-0.57,-0.71,-0.63);
const6 = _mm256_set_pd(-0.99,-0.58,-0.75,-0.62);
const7 = _mm256_set_pd(-0.98,-0.54,-0.74,-0.62);
const8 = _mm256_set_pd(-0.94,-0.12,-0.71,-0.61);

for (i=0;i<N;i++){
 for (j=0;j<N;j+=4){
a = _mm256_load_pd(&A[i][j]);
tmp = _mm256_setzero_pd();
tmp += _mm256_mul_pd(a, const1);
tmp += _mm256_mul_pd(a, const2);
tmp += _mm256_mul_pd(a, const3);
tmp += _mm256_mul_pd(a, const4);
tmp += _mm256_mul_pd(a, const5);
tmp += _mm256_mul_pd(a, const6);
tmp += _mm256_mul_pd(a, const7);
tmp += _mm256_mul_pd(a, const8);
_mm256_store_pd(&A[i][j], tmp);
} }


}


double step3(){

unsigned int i,j;
double reduction=0.0;

for (i=0;i<N;i++){
 for (j=0;j<N;j++){
  Y[i] += A[i][j] * X[j];
} }

 for (j=0;j<N;j+=8){
  reduction+=Y[j];
  }
  
  return reduction;
  
}






