/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main () 
{
int procs, maxt, inpar, dynamic, nested;

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel 
  {

  int nthreads, tid;
  
  tid = omp_get_thread_num(); //get the number of each thread
  printf("Hello World from thread = %d\n", tid); //THE ORDER OF THE PRINTF() DIFFERS FROM RUN TO RUN

  
  if (tid == 0)  // Only master thread does this 
    {
    nthreads = omp_get_num_threads(); //returns the number of threads used inside #pragma omp parallel { }
    procs = omp_get_num_procs(); //returns the number of physical CPU cores
    maxt = omp_get_max_threads(); //returns the maximum number of threads available. by default this number will be set to the maximum number of available cores 
    inpar = omp_in_parallel(); //This function returns true if currently running in parallel, false otherwise.
    dynamic = omp_get_dynamic(); //This function returns true if enabled, false otherwise. 
    nested = omp_get_nested(); //This function returns true if nested parallel regions are enabled, false otherwise. If undefined, nested parallel regions are disabled by default. 

    printf("Number of threads = %d\n", nthreads);
    printf("Number of processors = %d\n", procs);
    printf("Max threads = %d\n", maxt);
    printf("In parallel? = %d\n", inpar);
    printf("Dynamic threads enabled? = %d\n", dynamic);
    printf("Nested parallelism enabled? = %d\n", nested);
    }

  }  // All threads join master thread and disband

}


