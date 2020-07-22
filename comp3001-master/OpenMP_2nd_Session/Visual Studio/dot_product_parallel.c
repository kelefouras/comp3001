#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
//#include <chrono>

void init(double *a, double *b);
double dot_prod_serial(double *a, double *b);
double dot_prod_parallel_ver1(double *a, double *b);
double dot_prod_parallel_ver2(double *a, double *b);
double dot_prod_parallel_ver3(double *a, double *b);
double dot_prod_parallel_ver4(double *a, double *b);

#define N 10000000 //array size
#define chunk 1000

#define BILLION 1000000000L
#define TIMES 1

int main () {
int i,it;
double *a, *b; // These pointers will hold the base addresses of the memory blocks created 
double sum,sum2;


/* Dynamically allocate memory storage for the arrays */
a = (double*) malloc (N * sizeof(double));
    if (a == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0); //terminates the process immediately
    } 

b = (double*) malloc (N * sizeof(double));
    if (b == NULL) { // Check if the memory has been successfully allocated by malloc or not 
        printf("\nMemory not allocated.\n"); 
        exit(0);  //terminates the process immediately
    } 
 

init(a,b); //initialize the arrays


//define the timers measuring execution time
clock_t start_1, end_1; //ignore this for  now
start_1 = clock(); //start the timer (THIS IS NOT A VERY ACCURATE TIMER) - ignore this for now

//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only

for (it=0; it< TIMES; it++)
//sum=dot_prod_serial(a,b); //execute the main routine
sum2=dot_prod_parallel_ver2(a,b); //execute the main routine

//auto finish = std::chrono::high_resolution_clock::now(); 
end_1 = clock(); //end the timer - ignore this for now


printf ("Done. Serial version:   sum  =  %f \n", sum);
printf ("Done. Parallel version: sum  =  %f \n", sum2);


free (a); //deallocate the memory
free (b); //deallocate the memory

printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
//std::chrono::duration<double> elapsed = finish - start;
//std::cout << "Elapsed time: " << elapsed.count() << " s\n";

system("pause"); //this command does not let the output window to close
return 0;

}   


void init(double *a, double *b){

double e=0.12134;
/* Initialize dot product vectors */
for (int i=0; i<N; i++) {
  a[i]=(double) (i%100)+e; 
  b[i]=a[i]+e; 
  }

}


/* Perform the dot product */
double dot_prod_serial(double *a, double *b){

double sum = 0.0;
int i;

for (i=0; i<N; i++) 
  {
    sum += (a[i] * b[i]);
  }


return sum;
}


//this implementation contains the OpenMP features we studied previous week
double dot_prod_parallel_ver1(double *a, double *b){


double sum = 0.0;
int i;

#pragma omp parallel for reduction(+:sum) 
for (i=0; i<N; i++) 
  {
    sum += (a[i] * b[i]);
  }


return sum;

}


//this implementation contains the schedule feature
double dot_prod_parallel_ver2(double *a, double *b){


double sum = 0.0;
int i;

#pragma omp parallel for reduction(+:sum) schedule(static)
for (i=0; i<N; i++) 
  {
    sum += (a[i] * b[i]);
  }


return sum;

}



double dot_prod_parallel_ver3(double *a, double *b){


double sum = 0.0;
int i;

#pragma omp parallel 
{

  int tid = omp_get_thread_num();
  if (tid == 0)
    {
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
   printf("Thread %d is starting...\n",tid);

#pragma omp for reduction(+:sum) schedule(dynamic, chunk)
for (i=0; i<N; i++) 
  {
    sum += (a[i] * b[i]);
    printf("Thread %d: executes iteration i= %d\n",tid,i);
  }

}

return sum;

}

//omp simd is supported in CS 2019 only (new technology)
//this implementation both multi-threading and vectorization
/*
double dot_prod_parallel_ver4(double *a, double *b) {


	double sum = 0.0;
	int i;

#pragma omp parallel for simd reduction(+:sum) schedule(static)
	for (i = 0; i < N; i++)
	{
		sum += (a[i] * b[i]);
	}


	return sum;

}
*/



