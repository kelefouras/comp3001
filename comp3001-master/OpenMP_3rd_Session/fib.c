/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//TO RUN ON WINDOWS, JUST COMMENT THE clock_gettime COMMANDS

//This program computes the fibonacci sequence

//gcc fib.c -o p -O2  -fopenmp


#include <math.h>
#include <stdio.h>
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>
 #include <sys/time.h>
#include <stdint.h>	/* for uint64 definition */


#define BILLION 1000000000L
#define INPUT 10

int Fibonacci(int n);
int kernel(int n);


int main(){
  
double start_time, run_time;
int n=10;
struct timespec start, end; //timers
uint64_t diff;


clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
start_time = omp_get_wtime();

n=Fibonacci(INPUT);

run_time = omp_get_wtime() - start_time;
clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
printf("\n omp time is %f seconds \n", run_time);

printf("\n n = %d\n",n);

}



int Fibonacci(int n){
int fib;

#pragma omp parallel 
{
#pragma omp single 
fib=kernel(n);
}

return fib;

}


int kernel (int n){

int x,y;

if (n<2)
 return n;

#pragma omp task shared(x) //x must be shared otherwise, it will be lost when the task ends. x is undefined outside the task
 x=kernel(n-1);

#pragma omp task shared(y)//y must be shared otherwise, it will be lost when the task ends. y is undefined outside the task
 y=kernel(n-2);

#pragma omp taskwait
return x+y;

}





