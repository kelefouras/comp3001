/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <pmmintrin.h>
#include <process.h>
//#include <chrono>
#include <iostream>
#include <immintrin.h>

#define N 128


#define Tq 128
#define Tp 8
#define Ts 32


__declspec(align(64)) float sum[N][N][N],A[N][N][N],C[N][N] ;
__declspec(align(64)) float sum2[N*N*N],A2[N*N*N],C2[N*N] ;


void init();
void default_kernel();//vectorize this routine


int main(){



init();

default_kernel();


system("pause"); //this command does not let the output window to close

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
	   


//vectorize this routine
void default_kernel(){


int r, q, p, s;


  for (r = 0; r < N; r++)
    for (q = 0; q < N; q++)  
      for (s = 0; s < N; s++)
       for (p = 0; p < N; p++)  
	  sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C[s][p];

}







