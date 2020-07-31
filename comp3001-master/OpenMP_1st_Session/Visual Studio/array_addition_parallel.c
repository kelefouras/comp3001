/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
//#include <chrono>


#define NUM_THREADS 4
#define N 1000000
#define TIMES 1

int A[N], B[N], D[N];

void init();
void un_opt();
void  version1();
void  version2();
void  version3();
bool compare();

int main() {

	double pi1, pi2;

	init(); //initialize the arrays

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now
	start_1 = clock(); //start the timer (THIS IS NOT A VERY ACCURATE TIMER) - ignore this for now

	//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only

	for (int i = 0; i < TIMES; i++) { //run this many times to get accurate measurement. The output might be wrong in this case. Debug with TIMES=1.
		//un_opt();
		version1();
		//version2();
		//version3();

	}

	//auto finish = std::chrono::high_resolution_clock::now(); 
	end_1 = clock(); //end the timer - ignore this for now

	if (compare() == true)
		printf("\nResult is ok\n");
	else
		printf("\nResult is FALSE\n");

	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	system("pause"); //this command does not let the output window to close
	return 0;

}


void init() {

	int i;

	for (i = 0; i < N; i++) {
		A[i] = rand() % 50;
		B[i] = rand() % 1000;
		D[i] = A[i];
	}

}

//this is the serial version of the program you need to parallelize
void un_opt() {

	int i;

	for (i = 0; i < N; i++)
		A[i] = A[i] + B[i];

}


bool compare() {

	int i;

	for (i = 0; i < N; i++)
		D[i] = D[i] + B[i];

	for (i = 0; i < N; i++) {
		//printf(" %d %d - ",D[i],A[i]);
		if (D[i] != A[i])
			return false;
	}

	return true;

}


//This implementation does not use omp for construct. It is Hard to write, but important to understand how it works
void  version1() {

#pragma omp parallel
	{
		int id, i, Nthrds, start, end;

		id = omp_get_thread_num();
		Nthrds = omp_get_num_threads();

		start = id * N / Nthrds;
		end = (id + 1) * N / Nthrds;

		if (id == Nthrds - 1)
			end = N;

		for (i = start; i < end; i++)
			A[i] = A[i] + B[i];

	}

}


void  version2() {

	int i;

#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < N; i++) //openmp makes the loop control index on a parallel loop private to a thread
			A[i] = A[i] + B[i];

	}

}


void  version3() {

	int i;

#pragma omp parallel for
	for (i = 0; i < N; i++) //openmp makes the loop control index on a parallel loop private to a thread
		A[i] = A[i] + B[i];


}





