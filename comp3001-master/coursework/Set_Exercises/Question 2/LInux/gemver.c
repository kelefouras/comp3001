/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc gemver.c -o p -O3 -D_GNU_SOURCE  -march=native -mavx -lm -D_GNU_SOURCE

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

void initialization();
unsigned short int gemver_default();
unsigned short int gemver_vectorized();
unsigned short int Compare_Gemver();
inline unsigned short int equal(float const a, float const b);

#define P 4096 //input size
float A2[P][P], test4[P][P], u1[P], v1[P], u2[P], v2[P] __attribute__((aligned(64)));

#define TIMES_TO_RUN 1 //how many times the function will run
#define EPSILON 0.0001

int main() {

	uint64_t diff;

	//define the timers measuring execution time
	struct timespec start, end; //timers

	initialization();

	clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

	for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
		gemver_default();
		

	clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

	printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);

	if (Compare_Gemver() == 0)
		printf("\nCorrect Result\n");
	else 
		printf("\nINcorrect Result\n");


	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}



void initialization() {

	float e = 0.1234, p = 0.7264, r = 0.11;

	//gemver
	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++) {
			A2[i][j] = 0.0;
			test4[i][j] = 0.0;
		}

	for (int j = 0; j < P; j++) {
		u1[j] = e + (j % 9);
		v1[j] = e - (j % 9) + 1.1;
		u2[j] = p + (j % 9) - 1.2;
		v2[j] = p - (j % 9) + 2.2;
	}


}



unsigned short int gemver_default() {

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			A2[i][j] += u1[i] * v1[j] + u2[i] * v2[j];

	return 0;
}



unsigned short int Compare_Gemver() {

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			test4[i][j] += u1[i] * v1[j] + u2[i] * v2[j];

	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++)
			if (equal(test4[i][j], A2[i][j]) == 1)
				return -1;

	return 0;
}






unsigned short int equal(float const a, float const b) {
	
	if (fabs(a-b)/fabs(a) < EPSILON)
		return 0; //success
	else
		return 1;
}



