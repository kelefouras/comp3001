/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//This is naive program that measures the maximum FLOPS achieved in a PC

#include <Windows.h>
#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include <omp.h>


#define BILLION 1000000000

#define N 4096*16 //arrays input size
#define TIMES 100000 //times to run
#define TILE 20 //tile size
#define ARITHMETICAL_OPS N*16


//In C, all the routines must be declared
inline void flops();
void initialize();

__declspec(align(64))  float A[N];

int main() {


	double my_flops;
	int i;

	//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	BOOL success = SetProcessAffinityMask(GetCurrentProcess(), 1);
	if (success == 0) {
		//cout << "SetProcessAffinityMask failed" << endl;
		printf("\nSetProcessAffinityMask failed\n");
		system("pause");
		return -1;
	}

	initialize();

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now
	start_1 = clock();

	for (i = 0; i < TIMES; i++) {
		flops();
	}

	end_1 = clock() - start_1; //end the timer

	printf("elapsed time = %f seconds\n", (float)(end_1) / CLOCKS_PER_SEC);
	my_flops = (double) (TIMES * (double) ( (ARITHMETICAL_OPS ) / ( (end_1) / CLOCKS_PER_SEC)) ); 
	printf("\n%f GigaFLOPS achieved\n", my_flops/BILLION);


	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize() {

	int i;

	for (i = 0; i < N; i++)
		A[i] = (float)(i % 7 + 0.01);

}


void flops() {

	int i;
	__m256 const1, const2, const3, const4, const5, const6, const7, const8, a, tmp;
	const1 = _mm256_set_ps(0.91f, 0.21f, 0.24f, 0.84f, -0.94f, -0.53f, -0.76f, -0.64f);
	const2 = _mm256_set_ps(0.92f, 0.22f, 0.25f, 0.85f, -0.95f, -0.54f, -0.73f, -0.67f);
	const3 = _mm256_set_ps(0.93f, 0.23f, 0.26f, 0.86f, -0.96f, -0.55f, -0.72f, -0.66f);
	const4 = _mm256_set_ps(0.94f, 0.24f, 0.27f, 0.87f, -0.97f, -0.56f, -0.74f, -0.65f);
	const5 = _mm256_set_ps(0.95f, 0.25f, 0.28f, 0.88f, -0.98f, -0.57f, -0.71f, -0.63f);
	const6 = _mm256_set_ps(0.96f, 0.26f, 0.29f, 0.89f, -0.99f, -0.58f, -0.75f, -0.62f);
	const7 = _mm256_set_ps(0.97f, 0.23f, 0.45f, 0.76f, -0.98f, -0.54f, -0.74f, -0.62f);
	const8 = _mm256_set_ps(0.98f, 0.43f, 0.43f, 0.77f, -0.94f, -0.12f, -0.71f, -0.61f);

	//Think of the non-vectorized version where i increases by one (i++).
	//There are 1 load, 1 store, 8 add and 8 mul operations. 
	//Arithmetic intensity=16/8bytes=2
	for (i = 0; i < N; i += 8) {
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








