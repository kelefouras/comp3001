//Run this program is problematic. Can you identify the reason?

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100

int main () {

int nthreads, i, tid;
float total;

#pragma omp parallel 
  {
  tid = omp_get_thread_num();
  
  #pragma omp master
   {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier 

  total = 0.0;
  #pragma omp for schedule(static)
  for (i=0; i<N; i++) 
     total = total + i;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } 

return 0;
}


