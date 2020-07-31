/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/
//compile with gcc MVM_parallel.c -o p -O2 -fopenmp -lm -fopt-info-vec-optimized

#include <stdio.h>
#include <stdlib.h> //this library allows for malloc to run
 #include <sys/time.h>
#include <stdint.h>	/* for uint64 definition */
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>

#define N 128//array size
#define TIMES 1 //times to run
#define NUM_THREADS 4 //number of threads

#define EPSILON 0.1
#define BILLION 1000000000L
void init(float *y, float *a,  float *x);
void MVM_serial(float *y, float *a,  float *x);
unsigned short int equal(float const a, float const b);
unsigned short int Compare_MVM(const float *y, const float *a,  const float *x);
void MVM_parallel_ver1(float *y, float *a,  float *x);
void MVM_parallel_ver2(float *y, float *a,  float *x);
void MVM_parallel_ver3(float *y, float *a,  float *x);
void MVM_parallel_ver4(float *y, float *a,  float *x);
void MVM_parallel_ver5(float *y, float *a,  float *x);
void MVM_parallel_ver6(float *y, float *a,  float *x);
void MVM_parallel_ver7(float *y, float *a,  float *x);
float test[N];



int main () {
int i,it;
float *x, *y, *a; // These pointers will hold the base addresses of the memory blocks created 

struct timespec start, end; //for timers
uint64_t diff;

/* Dynamically allocate memory storage for the arrays */
x =  _mm_malloc (N * sizeof(float),64); //dynamically allcate memory 64byte aligned
    if (x == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0); //terminates the process immediately
    } 

y = _mm_malloc ( N * sizeof(float), 64);//dynamically allcate memory 64byte aligned
    if (y == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    } 

a = _mm_malloc ( N * N * sizeof(float), 64);//dynamically allcate memory 64byte aligned
    if (a == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    }  

init(y,a,x); //initialize the arrays



clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (it=0; it< TIMES; it++)
 //MVM_serial(y,a,x); //execute the main routine
 MVM_parallel_ver4(y,a,x); //execute the main routine

clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);



if (Compare_MVM(y,a,x) == 0)
	printf("\n\n\r ----- output is correct -----\n\r");
else
	printf("\n\n\r ---- output is INcorrect -----\n\r");

_mm_free (x); //deallocate the memory
_mm_free (y); //deallocate the memory
_mm_free (a); //deallocate the memory

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
	if (fabs(temp) < EPSILON)
		return 0; //success
	else
		return 1;
}






//multi-threading
void MVM_parallel_ver1(float *y, float *a,  float *x){

int i,j;

#pragma omp parallel for private(i,j) schedule(static)//openmp will automatically make i private. However, j must be declared as private
for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
 y[i]+=a[N*i+j]*x[j];

}

//multi-threading
void MVM_parallel_ver2(float *y, float *a,  float *x){

int i,j;

#pragma omp parallel for shared(y,a,x) private(i,j) //y,a,x are shared by default, so the shared(y,a,x) is not needed, it is good practice though.
for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
 y[i]+=a[N*i+j]*x[j];

}

//multi-threading
void MVM_parallel_ver3(float *y, float *a,  float *x){

int i,j;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for shared(y,a,x) private(i,j) schedule(static)
for (i=0; i<N; i++) 
 for (j=0; j<N; j++) 
 y[i]+=a[N*i+j]*x[j];

}

//multi-threading + vectorization using OpenMP
void MVM_parallel_ver4(float *y, float *a,  float *x){

int i,j;
float tmp;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for shared(y,a,x) private(i,j,tmp) schedule(static) 
for (i=0; i<N; i++) {
tmp=y[i];
#pragma omp simd aligned(y,x,a:64) reduction(+:tmp)
 for (j=0; j<N; j++) {
 tmp+=a[N*i+j]*x[j];
}
y[i]=tmp;
}

}

//multi-threading + vectorization using AVX instrinsics
void MVM_parallel_ver5(float *y, float *a,  float *x){

int i,j;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel shared(y,a,x) private(i,j) 
{
__m256 ymm2, num0, num1, num2, num3, num4, num5;
__m128 xmm1, xmm2;

#pragma omp for schedule(static)
for (i=0; i<N; i++) {
		num1 = _mm256_setzero_ps();

		for (j = 0; j < ((N / 8) * 8); j += 8) { //main loop that vectorizes the code

			num5 = _mm256_load_ps(&x[j]);
			num0 = _mm256_loadu_ps(&a[N*i+j]); //if (N%8)!=0, then loadu is needed
			num1 = _mm256_fmadd_ps(num0, num5, num1);
		}

		ymm2 = _mm256_permute2f128_ps(num1, num1, 1);
		num1 = _mm256_add_ps(num1, ymm2);
		num1 = _mm256_hadd_ps(num1, num1);
		num1 = _mm256_hadd_ps(num1, num1);
		xmm2 = _mm256_extractf128_ps(num1, 0);
		_mm_store_ss(&y[i], xmm2);

		for (; j < N; j++) { //padding code for the case where (N%8)!=0
			y[i] += a[N*i+j] * x[j];
		}
}

}

}


//use N=1024 and run this code
//multi-threading + vectorization + register blocking
void MVM_parallel_ver6(float *y, float *a,  float *x){

int i,j;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel shared(y,a,x) private(i,j) 
{
__m256 tmp, numy0, numy1,numy2,numy3,numy4,numy5,numy6,numy7,numa, numx;
__m128 xmm1, xmm2;

#pragma omp for schedule(static)
for (i=0; i<N; i+=8) {
		numy0 = _mm256_setzero_ps();
		numy1 = _mm256_setzero_ps();
		numy2 = _mm256_setzero_ps();
		numy3 = _mm256_setzero_ps();
		numy4 = _mm256_setzero_ps();
		numy5 = _mm256_setzero_ps();
		numy6 = _mm256_setzero_ps();
		numy7 = _mm256_setzero_ps();

		for (j = 0; j < ((N / 8) * 8); j += 8) { //main loop that vectorizes the code

			numx = _mm256_load_ps(&x[j]);

			numa = _mm256_load_ps(&a[N*i+j]); //if (N%8)!=0, then loadu is needed
			numy0 = _mm256_fmadd_ps(numa, numx, numy0);
			numa = _mm256_load_ps(&a[N*(i+1)+j]); //if (N%8)!=0, then loadu is needed
			numy1 = _mm256_fmadd_ps(numa, numx, numy1);
			numa = _mm256_load_ps(&a[N*(i+2)+j]); //if (N%8)!=0, then loadu is needed
			numy2 = _mm256_fmadd_ps(numa, numx, numy2);
			numa = _mm256_load_ps(&a[N*(i+3)+j]); //if (N%8)!=0, then loadu is needed
			numy3 = _mm256_fmadd_ps(numa, numx, numy3);
			numa = _mm256_load_ps(&a[N*(i+4)+j]); //if (N%8)!=0, then loadu is needed
			numy4 = _mm256_fmadd_ps(numa, numx, numy4);
			numa = _mm256_load_ps(&a[N*(i+5)+j]); //if (N%8)!=0, then loadu is needed
			numy5 = _mm256_fmadd_ps(numa, numx, numy5);
			numa = _mm256_load_ps(&a[N*(i+6)+j]); //if (N%8)!=0, then loadu is needed
			numy6 = _mm256_fmadd_ps(numa, numx, numy6);
			numa = _mm256_load_ps(&a[N*(i+7)+j]); //if (N%8)!=0, then loadu is needed
			numy7 = _mm256_fmadd_ps(numa, numx, numy7);
		}

		//the following procedure can be optimized, but this is out of the scope of this session
		tmp = _mm256_permute2f128_ps(numy0, numy0, 1);
		numy0 = _mm256_add_ps(numy0, tmp);
		numy0 = _mm256_hadd_ps(numy0, numy0);
		numy0 = _mm256_hadd_ps(numy0, numy0);
		xmm2 = _mm256_extractf128_ps(numy0, 0);
		_mm_store_ss(&y[i], xmm2);

		tmp = _mm256_permute2f128_ps(numy1, numy1, 1);
		numy1 = _mm256_add_ps(numy1, tmp);
		numy1 = _mm256_hadd_ps(numy1, numy1);
		numy1 = _mm256_hadd_ps(numy1, numy1);
		xmm2 = _mm256_extractf128_ps(numy1, 0);
		_mm_store_ss(&y[i+1], xmm2);

		tmp = _mm256_permute2f128_ps(numy2, numy2, 1);
		numy2 = _mm256_add_ps(numy2, tmp);
		numy2 = _mm256_hadd_ps(numy2, numy2);
		numy2 = _mm256_hadd_ps(numy2, numy2);
		xmm2 = _mm256_extractf128_ps(numy2, 0);
		_mm_store_ss(&y[i+2], xmm2);

		tmp = _mm256_permute2f128_ps(numy3, numy3, 1);
		numy3 = _mm256_add_ps(numy3, tmp);
		numy3 = _mm256_hadd_ps(numy3, numy3);
		numy3 = _mm256_hadd_ps(numy3, numy3);
		xmm2 = _mm256_extractf128_ps(numy3, 0);
		_mm_store_ss(&y[i+3], xmm2);

		tmp = _mm256_permute2f128_ps(numy4, numy4, 1);
		numy4 = _mm256_add_ps(numy4, tmp);
		numy4 = _mm256_hadd_ps(numy4, numy4);
		numy4 = _mm256_hadd_ps(numy4, numy4);
		xmm2 = _mm256_extractf128_ps(numy4, 0);
		_mm_store_ss(&y[i+4], xmm2);

		tmp = _mm256_permute2f128_ps(numy5, numy5, 1);
		numy5 = _mm256_add_ps(numy5, tmp);
		numy5 = _mm256_hadd_ps(numy5, numy5);
		numy5 = _mm256_hadd_ps(numy5, numy5);
		xmm2 = _mm256_extractf128_ps(numy5, 0);
		_mm_store_ss(&y[i+5], xmm2);

		tmp = _mm256_permute2f128_ps(numy6, numy6, 1);
		numy6 = _mm256_add_ps(numy6, tmp);
		numy6 = _mm256_hadd_ps(numy6, numy6);
		numy6 = _mm256_hadd_ps(numy6, numy6);
		xmm2 = _mm256_extractf128_ps(numy6, 0);
		_mm_store_ss(&y[i+6], xmm2);

		tmp = _mm256_permute2f128_ps(numy7, numy7, 1);
		numy7 = _mm256_add_ps(numy7, tmp);
		numy7 = _mm256_hadd_ps(numy7, numy7);
		numy7 = _mm256_hadd_ps(numy7, numy7);
		xmm2 = _mm256_extractf128_ps(numy7, 0);
		_mm_store_ss(&y[i+7], xmm2);

		for (; j < N; j++) { //padding code for the case where (N%8)!=0
			y[i] += a[N*i+j] * x[j];
		}
}

}

}

/*
//use N=8192 and run this routine
//multi-threading + vectorization using OpenMP + loop tiling
void MVM_parallel_ver7(float *y, float *a,  float *x){

int i,j,jj;
float tmp;

omp_set_num_threads(4); // 4 threads

for (jj=0; jj<N; jj+=N/2){
#pragma omp parallel for shared(y,a,x) private(i,j) schedule(static)
for (i=0; i<N; i++) {
tmp=y[i];
#pragma omp simd aligned(y,x,a:64) reduction(+:tmp)
 for (j=jj; j<jj+N/2; j++) {
 tmp+=a[N*i+j]*x[j];
}
y[i]=tmp;
}
}

}
*/



