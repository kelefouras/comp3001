/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with gcc mmm.c -o p -O2 -D_GNU_SOURCE  

#include <stdio.h> //library for printf() 
#include <time.h>	/* for clock_gettime */
#include <stdint.h>	/* for uint64 definition */
#include <pthread.h> // for cpu_set_t 

#define BILLION 1000000000L

#define N 128  //arrays input size
#define TIMES 2000 //times to run
#define TILE 16 //tile size
#define ARITHMETICAL_OPS N*N*N*2

//In C, all the routines must be declared
void initialize();
void mmm();
void array_copying();
void mmm_reg_blocking_4();
void mmm_reg_blocking_8();
void mmm_tiling_bad();
void mmm_array_copying();

float A[N][N], B[N][N], C[N][N], Btranspose[N][N];


int main( ) {


struct timespec start, end; //timers
uint64_t diff;
double gflops;

//the following code binds this thread to code number 0. Without this code, the OS will tongle the thread among the cores, to reduce heat dissipation
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET(0,&mask);
if(sched_setaffinity(0,sizeof(mask),&mask) == -1)
   printf("WARNING: Could not set CPU Affinity, continuing...\n");


initialize();
//mmm_array_copying();

/* measure monotonic time */
clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (int t=0;t<TIMES;t++)
//mmm();
mmm_reg_blocking_8();
//mmm_tiling_bad();


clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
gflops = (double) ARITHMETICAL_OPS / (diff / TIMES); //ARITHMETICAL_OPS /(nanoseconds/TIMES)
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds \n%f GigaFLOPS achieved\n", (long long unsigned int) diff/1000000, gflops);



return 0; //normally, by returning zero, we mean that the program ended successfully. 
}



void initialize(){

int i,j;

for (i=0;i<N;i++)
 for (j=0;j<N;j++){
  A[i][j]=(float) (j%99 + 0.1);
  B[i][j]=(float) (j%65 - 0.1);
  C[i][j]=0.0;

}

}

void mmm(){

int i,j,k;

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  for (k=0;k<N;k++) 
 C[i][j]+=A[i][k]*B[k][j];
 

}

//assume that N%4==0
void mmm_reg_blocking_4(){

int i,j,k;
float c0,c1,c2,c3,a;

//register blocking has been applied to j loop by a factor of 4.
for (i=0;i<N;i++)
 for (j=0;j<N;j+=4){
 c0=C[i][j];
 c1=C[i][j+1];
 c2=C[i][j+2];
 c3=C[i][j+3];
  for (k=0;k<N;k++) {
 a=A[i][k];
 c0+=a*B[k][j];
 c1+=a*B[k][j+1];
 c2+=a*B[k][j+2];
 c3+=a*B[k][j+3];
}
C[i][j]=c0;
C[i][j+1]=c1;
C[i][j+2]=c2;
C[i][j+3]=c3;
}


}


//assume that N%8==0
void mmm_reg_blocking_8(){

int i,j,k;
float c0,c1,c2,c3,c4,c5,c6,c7,c8,a;

//register blocking has been applied to j loop by a factor of 4.
for (i=0;i<N;i++)
 for (j=0;j<N;j+=8){
 c0=C[i][j];
 c1=C[i][j+1];
 c2=C[i][j+2];
 c3=C[i][j+3];
 c4=C[i][j+4];
 c5=C[i][j+5];
 c6=C[i][j+6];
 c7=C[i][j+7];
  for (k=0;k<N;k++) {
 a=A[i][k];
 c0+=a*B[k][j];
 c1+=a*B[k][j+1];
 c2+=a*B[k][j+2];
 c3+=a*B[k][j+3];
 c0+=a*B[k][j+4];
 c1+=a*B[k][j+5];
 c2+=a*B[k][j+6];
 c3+=a*B[k][j+7];
}
C[i][j]=c0;
C[i][j+1]=c1;
C[i][j+2]=c2;
C[i][j+3]=c3;
C[i][j+4]=c4;
C[i][j+5]=c5;
C[i][j+6]=c6;
C[i][j+7]=c7;
}


}



//this is an optimized version of mmm()
void mmm_tiling(){ //array copying + tiling

int i,j,k,jj,e,c;



for (jj=0; jj<N; jj+=TILE)
 for (i=0;i<N;i++)
  for (j=jj;j<jj+TILE;j++){
   for (k=0;k<N;k++) {
 C[i][j]+=A[i][k]*Btranspose[j][k];
}
}


}


void mmm_array_copying(){

int i,j,k,jj,e,c;

//extra loop kernel introduced because of array copying. This introduces a performance overhead, but it pays off
for (i=0;i<N;i++)
 for (j=0;j<N;j++)
Btranspose[i][j]=B[j][i];


for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  for (k=0;k<N;k++) 
 C[i][j]+=A[i][k]*Btranspose[j][k];
 

}


//this a naive implementation of loop tiling. 
//more efficient loop tiling implementations are not discussed here, as this is a more advanced topic. 
void mmm_tiling_bad(){

int i,j,k,ii,jj,kk;

for (ii=0;ii<N;ii+=TILE)
for (jj=0;jj<N;jj+=TILE)
for (kk=0;kk<N;kk+=TILE)

for (i=ii;i<ii+TILE;i++)
 for (j=jj;j<jj+TILE;j++)
  for (k=kk;k<kk+TILE;k++) 
 C[i][j]+=A[i][k]*B[k][j];
 

}





