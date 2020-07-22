

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

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

int i, nthreads, tid;

initialize();

#pragma omp parallel shared(a,b,c,d,nthreads) private(i,tid) 
  {
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n",tid);

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d doing section 1\n",tid);
      #pragma omp simd	
      for (i=0; i<N; i++)
        {
        c[i] = a[i] + b[i];
       // printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
        }
      }

    #pragma omp section
      {
      printf("Thread %d doing section 2\n",tid);
      #pragma omp simd	
      for (i=0; i<N; i++)
        {
        d[i] = a[i] * b[i];
        //printf("Thread %d: d[%d]= %f\n",tid,i,d[i]);
        }
      }

    }  /* end of sections */

    printf("Thread %d done.\n",tid); 

  }  /* end of parallel section */

}

