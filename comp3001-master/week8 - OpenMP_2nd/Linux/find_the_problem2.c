/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/



#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4096

int A[N][N];



int main () {

int nthreads, tid, i, j;


#pragma omp parallel shared(nthreads) private(i,j,tid,A)
  {

  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      A[i][j] = (i-j)%100;


  printf("Thread %d done.\n",tid);
  printf("A[N-1][N-1] = %d \n",A[N-1][N-1]);

  }  

return 0;

}






