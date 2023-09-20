/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 100000000

float a[N], b[N], c[N], d[N];

void initialize(){
int i;
for (i=0; i<N; i++) {
  a[i] = i * 1.32;
  b[i] = i + 0.265;
  c[i] = 0.0;
  d[i] = 0.0;
  }
}



int main ( ) {

int i,j, nthreads, tid;

initialize();

omp_set_dynamic(0);
omp_set_nested(1);
//omp_set_num_threads(3);
#pragma omp parallel shared(a,b,c,d,nthreads) private(tid) num_threads(2)
  {
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads at first parallel region = %d\n", nthreads);
    }
  printf("Thread %d starting...\n",tid);

  #pragma omp sections 
    {
    #pragma omp section
      {
      printf("Thread %d doing section 1\n",tid);
      #pragma omp parallel for simd schedule(static) private(i) num_threads(2) 
      for (j=0; j<1000; j++)
      for (i=0; i<N; i++)
        {
        c[i] += (a[i] + b[i])/0.21f + sqrt(i)*0.033f;
        //printf("Thread %d: c[%d]= %f\n",omp_get_thread_num(),i,c[i]);
        }
      }

    #pragma omp section
      {
      printf("Thread %d doing section 2\n",tid);
      #pragma omp parallel for simd schedule(static) private(i) num_threads(2)	
       for (j=0; j<1000; j++)
       for (i=0; i<N; i++)
        {
        d[i] += (a[i] * b[i])/0.23f + sqrt(i)*0.023f;
        //printf("Thread %d: d[%d]= %f\n",omp_get_thread_num(),i,d[i]);
        }
      }

    }  /* end of sections */

    printf("Thread %d done.\n",tid); 

  }  /* end of parallel section */

}

