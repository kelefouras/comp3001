/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h>
#include <time.h>
#include <chrono>
#include <iostream>


#define M 8096

float  X[M], Y[M], A[M][M], X2[M], Y2[M];


void no_loop_merge();
void loop_merge();
void initialize();

int main() {


	initialize();

	clock_t start_1, end_1;

	//start_1 = clock();
	auto start = std::chrono::high_resolution_clock::now();

	//run this 10 times because this routine runs very fast 
	//The execution time needs to be at least some seconds in order to have a good measurement (why?) 
	//			because other processes run at the same time too, preempting our thread
	for (int t = 0; t < 20; t++) {

		no_loop_merge();
	}

	

	auto finish = std::chrono::high_resolution_clock::now();
	//end_1 = clock();

	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	system("pause");
	return 0;
}

void initialize() {
	int i, j;

	for (i = 0; i < M; i++)
		for (j = 0; j < M; j++)
			A[i][j] = (float) ((i - j)%7 + 0.1);

	for (j = 0; j < M; j++) {
		Y[j] = 0.0;
		Y2[j] = 0.0;
		X[j] = (float)((j % 9) + 0.2);
		X2[j] = (float)((j % 9) - 0.2);
	}
}


void no_loop_merge() {


	//implementation #1
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			Y[i] += A[i][j] * X[j];

	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			Y2[i] += A[i][j] * X2[j];

}

void loop_merge(){


	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++) {
			Y[i] += A[i][j] * X[j];
			Y2[i] += A[i][j] * X2[j];
		}

		
}
