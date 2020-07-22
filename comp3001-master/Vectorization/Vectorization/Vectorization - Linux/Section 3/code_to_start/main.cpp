//THE FOLLOWING GCC OPTIONS CAN BE DEFINED FROM HERE INSTEAD OF TYPING THEM WHEN COMPILING
/*
#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx")  //Enable AVX
#include <x86intrin.h> //AVX/SSE Extensions
*/


//The '-fopt-info-vec-optimized' will give me info about vectorization
//gcc main.cpp array_addition.cpp array_constant_addition.cpp MVM.cpp -o p -O2 -march=native -mavx -lm -D_GNU_SOURCE  -g  -pthread

#include "MVM.h"
#include "array_addition.h"
#include "array_constant_addition.h"

#define BILLION 1000000000L
#define TIMES_TO_RUN 1 //how many times the function will run

void print_message(char *s, unsigned short int outcome);

//using namespace std; 

char message[50];


int main() {

	unsigned short int output;

	struct timespec start, end; //timers
        uint64_t diff;

//the following command pins the current process to the 1st core
	//otherwise, the OS tongles this process between different cores
	time_t start1, end1;
	struct timeval start2, end2;

	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0,&mask);
	if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
	       printf("WARNING: Could not set CPU Affinity, continuing...\n");


	//initialize the arrays
	initialization_MVM();
	initialization_Add();
	initialization_ConstAdd();

	/* measure monotonic time */
	clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

	//run this several times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	//---Appropriately MODIFY the 'TIMES_TO_RUN' and the input size (defined in the appropriate header file)---
	for (int t = 0; t < TIMES_TO_RUN; t++) {

		//output = ConstAdd_default();
		//output = ConstAdd_SSE();
		//output = ConstAdd_AVX();

		//output = Add_default();
		//output = Add_SSE();
		//output = Add_AVX();

		//output=MVM_default();
		output=MVM_SSE();
		//output=MVM_AVX();

	}



	clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */


	if (output == 1) {
		snprintf(message, sizeof(message) - 1, "MVM");
		print_message(message, Compare_MVM());
	}
	else if (output == 0) {
		snprintf(message, sizeof(message) - 1, "Array Addition");
		print_message(message, Compare_Add());
	}
	else if (output == 2) {
		snprintf(message, sizeof(message) - 1, "Array Constant Addition");
		print_message(message, Compare_ConstAdd());
	}
	else {
		printf("\n Error\n");
	}

	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);

	return 0;
}




void print_message(char *s, unsigned short int outcome) {

	if (outcome == 0)
		printf("\n\n\r ----- %s output is correct -----\n\r", s);
	else
		printf("\n\n\r -----%s output is INcorrect -----\n\r", s);

}


