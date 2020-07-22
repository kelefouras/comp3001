//This is naive program that measures the maximum FLOPS achieved in a PC

//compile with gcc flops.c -o p -O2 -D_GNU_SOURCE  -march=native -mavx -lm -D_GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>


#define BILLION 1000000000L

#define N 4096*16 //arrays input size
#define TIMES 100000 //times to run
#define TILE 20 //tile size
#define ARITHMETICAL_OPS N*16


//In C, all the routines must be declared
inline void flops();
void initialize();

float A[N] __attribute__((aligned(64)));


int main( ) {


struct timespec start, end; //timers
uint64_t diff;
double gflops;
float out;
int i;

//the following code binds this thread to code number 0. Without this code, the OS will tongle the thread among the cores, to reduce heat dissipation
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(0,&mask);
if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
   printf("WARNING: Could not set CPU Affinity, continuing...\n");


initialize();

/* measure monotonic time */
clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (i=0;i<TIMES;i++){
 flops();
}

clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
gflops = (double) ARITHMETICAL_OPS / (diff / TIMES); //ARITHMETICAL_OPS /(nanoseconds/TIMES)
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
printf("output = %f \n%f GigaFLOPS achieved\n", out, gflops);



return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize(){

int i;

for (i=0;i<N;i++)
A[i]=(float) (i%7+0.01);

}


void flops(){

int i;
__m256 const1,const2,const3,const4,const5,const6,const7,const8,a,tmp;
const1 = _mm256_set_ps(0.91,0.21,0.24,0.84,-0.94,-0.53,-0.76,-0.64);
const2 = _mm256_set_ps(0.92,0.22,0.25,0.85,-0.95,-0.54,-0.73,-0.67);
const3 = _mm256_set_ps(0.93,0.23,0.26,0.86,-0.96,-0.55,-0.72,-0.66);
const4 = _mm256_set_ps(0.94,0.24,0.27,0.87,-0.97,-0.56,-0.74,-0.65);
const5 = _mm256_set_ps(0.95,0.25,0.28,0.88,-0.98,-0.57,-0.71,-0.63);
const6 = _mm256_set_ps(0.96,0.26,0.29,0.89,-0.99,-0.58,-0.75,-0.62);
const7 = _mm256_set_ps(0.97,0.23,0.45,0.76,-0.98,-0.54,-0.74,-0.62);
const8 = _mm256_set_ps(0.98,0.43,0.43,0.77,-0.94,-0.12,-0.71,-0.61);

//Think of the non-vectorized version where i increases by one (i++).
//There are 1 load, 1 store, 8 add and 8 mul operations. 
//Arithmetic intensity=16/8bytes=2
for (i=0;i<N;i+=8){
a = _mm256_load_ps(&A[i]);
tmp = _mm256_setzero_ps();
tmp = _mm256_fmadd_ps(a, const1, tmp);
tmp = _mm256_fmadd_ps(a, const2, tmp);
tmp = _mm256_fmadd_ps(a, const3, tmp);
tmp = _mm256_fmadd_ps(a, const4, tmp);
tmp = _mm256_fmadd_ps(a, const5, tmp);
tmp = _mm256_fmadd_ps(a, const6, tmp);
tmp = _mm256_fmadd_ps(a, const7, tmp);
tmp = _mm256_fmadd_ps(a, const8, tmp);
_mm256_store_ps(&A[i], tmp);
//printf("\n%f %f %f %f %f %f %f %f\n",A[i],A[i+1],A[i+2],A[i+3],A[i+4],A[i+5],A[i+6],A[i+7]);
}


}








