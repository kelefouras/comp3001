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
#define N 100

int A[N], B[N], Acopy1[N];

void init();
void un_opt();
void  version1();
void  version2();
void  version3();
void  version4();
bool compare();

int main() {


	init();

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now
	start_1 = clock(); //start the timer (THIS IS NOT A VERY ACCURATE TIMER) - ignore this for now

	//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only
	//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only

	for (int i = 0; i < 1; i++) {
		//un_opt();
		//version1();
		//version2();
		version3();

	}

	//auto finish = std::chrono::high_resolution_clock::now(); 
	end_1 = clock(); //end the timer - ignore this for now


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
		Acopy1[i] = A[i];
	}

}

//this is the serial version of the program you need to parallelize
void un_opt() {

	int ave = 0;
	int i;

	for (i = 0; i < N; i++) {
		ave += A[i];
	}

	printf("\nave=%d\n", ave);
}


//THIS IS THE RECOMMENDED IMPLEMENTATION
void version1() {

	int ave = 0;
	int i;

#pragma omp parallel for reduction(+:ave)
	for (i = 0; i < N; i++) {
		ave += Acopy1[i];
	}

	printf("\nave=%d\n", ave);
}


//THIS IS SIMILAR TO VERSION1, BUT WE PRINT THE NUMBER OF THREADS USED
void version2() {

	int ave = 0;
	int i, threads;

#pragma omp parallel 
	{
		int t = omp_get_num_threads();
		int id = omp_get_thread_num();
		if (id == 0)
			threads = t;

#pragma omp for reduction(+:ave)
		for (i = 0; i < N; i++) {
			ave += Acopy1[i];
		}

	}
	printf("\nave=%d\n", ave);
	printf("\nthreads=%d\n", threads);
}


//THIS IS SIMILAR TO VERSION2, BUT INSTEAD OF USING THE REDUCTION CLAUSE, THE CRITICAL CLAUSE IS USED INSTEAD
void version3() {

	int average = 0;
	int i, threads;

#pragma omp parallel 
	{
		int t = omp_get_num_threads();
		int id = omp_get_thread_num();
		if (id == 0)
			threads = t;

		int ave = 0;
#pragma omp for
		for (i = 0; i < N; i++) {
			ave += Acopy1[i];
		}

#pragma omp critical
		average += ave;

	}
	printf("\nave=%d\n", average);
	printf("\nthreads=%d\n", threads);
}





