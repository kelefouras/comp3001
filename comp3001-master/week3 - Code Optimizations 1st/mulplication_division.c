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

float A[N][N];

int main()
{
    int i,j,times;
            double start, end;
    printf("Hello \n");

    for (i=0; i<N; i++) 
    {
        for (j=0;j<N;j++)
        A[i][j]=(float) (j+0.1);
    }

start=omp_get_wtime();

for (times=0;times<100;times++)//run many times the following code as otherwise the ex.time is tiny
    for (i=0; i<N; i++) 
    {
        for (j=0;j<N;j++)
        A[i][j]=A[i][j]/5.0f;
	//A[i][j]=A[i][j]*0.2f;
        //printf(" %d",A[i][j]);
    }

end=omp_get_wtime();
printf("\ntime elapsed is %f secs\n",end-start);


return 0;
}
