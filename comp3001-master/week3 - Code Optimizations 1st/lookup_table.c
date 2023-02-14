/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------EMBEDDED PROGRAMMING AND THE INTERNET OF THINGS-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc pow.c -o p -O2 -fopenmp -lm

#include <stdio.h>
#include <omp.h>
#include <math.h>
#define N 1000

double A[N][N];
//double table[7]={1.414213562373, 1.732050807569, 2.0, 2.236067977500,2.449489742783, 2.645751311065, 2.828427124746};




int main()
{
    int i,j,k;
        double start, end;

    printf("Hello \n");

start=omp_get_wtime();

    for (i=0; i<N; i++)  { 
        for (j=0;j<N;j++){
            A[i][j]=0.0;
          for (k=2;k<9;k++){
              A[i][j]+=i*sqrt((double) k) + j*sqrt((double) k);
             // A[i][j]+=i*table[k-2] + j*table[k-2];
          }
        }
    }

end=omp_get_wtime();
printf("\ntime elapsed is %f secs\n",end-start);


return 0;
}

