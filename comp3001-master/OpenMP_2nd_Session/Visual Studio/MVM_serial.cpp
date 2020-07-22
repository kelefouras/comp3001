//omp simd is supported in VS2019 only

#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>

#define N 128 //array size
#define TIMES 1 //times to run
#define NUM_THREADS 4 //number of threads

#define EPSILON 0.1
#define BILLION 1000000000L
void init(float *y, float *a, float *x);
void MVM_serial(float *y, float *a, float *x);
unsigned short int equal(float const a, float const b);
unsigned short int Compare_MVM(const float *y, const float *a, const float *x);
float test[N];



int main() {
	int i, it;
	float *x, *y, *a; // These pointers will hold the base addresses of the memory blocks created 


	/* Dynamically allocate memory storage for the arrays */
	x = (float *)_mm_malloc(N * sizeof(float), 64); //dynamically allcate memory 64byte aligned
	if (x == NULL) { // Check if the memory has been successfully allocated by malloc or not 
		printf("\nMemory not allocated.\n");
		system("pause"); //this command does not let the output window to clo
		exit(0); //terminates the process immediately
	}

	y = (float *)_mm_malloc(N * sizeof(float), 64);//dynamically allcate memory 64byte aligned
	if (y == NULL) { // Check if the memory has been successfully allocated by malloc or not 
		printf("\nMemory not allocated.\n");
		system("pause"); //this command does not let the output window to clo
		exit(0);  //terminates the process immediately
	}

	a = (float *)_mm_malloc(N * N * sizeof(float), 64);//dynamically allcate memory 64byte aligned
	if (a == NULL) { // Check if the memory has been successfully allocated by malloc or not 
		printf("\nMemory not allocated.\n");
		system("pause"); //this command does not let the output window to clo
		exit(0);  //terminates the process immediately
	}

	init(y, a, x); //initialize the arrays



	//define the timers measuring execution time
	//clock_t start_1, end_1; //ignore this for  now
	//start_1 = clock(); //start the timer (THIS IS NOT A VERY ACCURATE TIMER) - ignore this for now

	auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only

	for (it = 0; it < TIMES; it++)
		MVM_serial(y, a, x); //execute the main routine

	auto finish = std::chrono::high_resolution_clock::now();
	//end_1 = clock(); //end the timer - ignore this for now
	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";


	if (Compare_MVM(y, a, x) == 0)
		printf("\n\n\r ----- output is correct -----\n\r");
	else
		printf("\n\n\r ---- output is INcorrect -----\n\r");

	_mm_free(x); //deallocate the memory
	_mm_free(y); //deallocate the memory
	_mm_free(a); //deallocate the memory

	system("pause"); //this command does not let the output window to close
	return 0;

}


void init(float *y, float *a, float *x) {

	float e = 0.63;
	int i, j;

	for (i = 0; i < N; i++) {
		x[i] = (float)(i % 100) + e;
		y[i] = 0.0;
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			a[N*i + j] = (float)(j % 30) - e;

}



void MVM_serial(float *y, float *a, float *x) {

	int i, j;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			y[i] += a[N*i + j] * x[j];


}





unsigned short int Compare_MVM(const float *y, const float *a, const float *x) {

	for (int i = 0; i < N; i++) {
		test[i] = 0.0;
		for (int j = 0; j < N; j++) {
			test[i] += a[N*i + j] * x[j];
		}
	}

	for (int j = 0; j < N; j++)
		if (equal(y[j], test[j]) == 1) {
			printf("\n j=%d %f %f\n", j, test[j], y[j]);
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




