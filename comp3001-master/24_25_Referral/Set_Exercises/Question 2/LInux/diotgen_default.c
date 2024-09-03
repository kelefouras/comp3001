//gcc diotgen_default.c -o p -march=native  -D_GNU_SOURCE -mavx2
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

#define N 128


#define Tq 128
#define Tp 8
#define Ts 32



float sum[N][N][N],A[N][N][N],C[N][N]   __attribute__((aligned(64)));
float sum2[N*N*N],A2[N*N*N],C2[N*N]  __attribute__((aligned(64)));

void writedata();
void init();
void default_kernel();


int main(){



init();

default_kernel();




return 0;
}


void init() {

	float e = 0.12, p = 0.72;
       unsigned int i,j,k;
	
	for ( i = 0; i < N; i++) { 
		for (j = 0; j < N; j++) {
			C[i][j] = (j % 9) + p; 
		}
	}
	   
	for ( i = 0; i < N; i++) { 
		for (j = 0; j < N; j++) {
                  for (k = 0; k < N; k++) {
			sum[i][j][k] = 0.0; 
			A[i][j][k]= ( ((i+j)%99) +e);
		}
		}
	      }
}
	   



void default_kernel(){


int r, q, p, s;


  for (r = 0; r < N; r++)
    for (q = 0; q < N; q++)  
      for (s = 0; s < N; s++)
       for (p = 0; p < N; p++)  
	  sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C[s][p];

}







