
#include <stdio.h>
#include <math.h>
#include <time.h>



#define N 1024

void q2_base();

float  X[N],Y[N] ;
float A[N][N], Atr[N*N] ;


int main(){

unsigned int i,j;

//-------------------initialize ---------------------------
for (i=0;i<N;i++){
Y[i]=0.0;
X[i]= ((i%99)/3);
}


for (i=0;i<N;i++)
for (j=0;j<N;j++)
A[i][j]=(((i+j)%99)/3);




//-------------------main kernel ---------------------------
q2_base();



return 0;
}


void q2_base(){
unsigned int i,j;
float alpha = 0.265f;

for (i=0;i<N;i++)
for (j=0;j<N;j++)
Y[i] += alpha * A[i][j] * X[j];

}




