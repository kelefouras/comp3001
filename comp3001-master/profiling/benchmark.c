/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//compile with 'gcc benchmark.c -o exec'
//run with './exec'

/*
How to use gprof profiler:
1. compile with -pg - 'gcc benchmark.c -o exec -pg'
2. run normally - './exec'
3. 'gprof exec' or  'gprof exec -b'
*/

#include <stdio.h> //library for fprintf() and fopen()

#define N 1000  //arrays input size

//In C, all the routines must be declared
void initialize();
void vec_add();
void MMM();
void MVM();
void vec_add_default();
void MMM_default();
void MVM_default();
void writedata_MVM();
void writedata_MMM();
void writedata_vec_add();

float A[N][N], B[N][N], C[N][N], X[N], Y[N];


int main( ) {

initialize();

vec_add();
//writedata_vec_add();

MVM();
//writedata_MVM();

MMM();
//writedata_MMM();


return 0; //normally, by returning zero, we mean that the program ended successfully. 
}



void initialize(){

int i,j;
float h=0.4563;

for (i=0;i<N;i++)
 for (j=0;j<N;j++){
  A[i][j]=(float) ((j%7)+h);
  B[i][j]=(float) ((j%19)+h);
  C[i][j]=0.0;
}

 for (j=0;j<N;j++){
  Y[j]=0.0;
  X[j]=j+0.0123;
}


}

void vec_add(){

int i,j;

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  C[i][j] = A[i][j] + B[i][j];

}


void MMM(){

int i,j,k;

for (i=0;i<N;i++){
 for (j=0;j<N;j++){
   C[i][j]=0.0;
  for (k=0;k<N;k++){
  C[i][j] += A[i][k] * B[k][j];
}}}

}


void MVM(){

int i,j;

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  Y[i] += A[i][j] * X[j];

}


void writedata_MVM(){
int l,i,j;
FILE *fp;

fp=fopen("MVM.txt","w"); //create a file to write

for (l=0;l<N;l++)
fprintf(fp,"%.3f  ",Y[l]); //write Y[] to the file

fclose(fp); //close the file



fp=fopen("MVM_default.txt","w"); //create a file to write


for (l=0;l<N;l++)  //re-initialize Y[] 
Y[l]=0;

 MVM_default();  

for (l=0;l<N;l++) 
fprintf(fp,"%.3f  ",Y[l]); //write Y[] to the file

fclose(fp); //close the file


}


void writedata_MMM(){
int l,i,j,k;
FILE *fp;

fp=fopen("MMM.txt","w"); //create a file to write

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
fprintf(fp,"%.3f  ",C[i][j]); //write C[][] to the file

fclose(fp); //close the file



fp=fopen("MMM_default.txt","w"); //create a file to write


for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  C[i][j]=0; //Re-initialize the C[][]

 MMM_default();

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
fprintf(fp,"%.3f  ",C[i][j]); //write C[][] to the file

fclose(fp); //close the file


}


void writedata_vec_add(){
int l,i,j;
FILE *fp;

fp=fopen("vec_add.txt","w");

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
fprintf(fp,"%.3f  ",C[i][j]);

fclose(fp);



fp=fopen("vec_add_default.txt","w");


for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  C[i][j]=0;

vec_add_default();

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
fprintf(fp,"%.3f  ",C[i][j]);

fclose(fp);


}


void vec_add_default(){

for (int i=0;i<N;i++)
 for (int j=0;j<N;j++)
  C[i][j]=A[i][j]+B[i][j];

}

void MMM_default(){

int i,j,k;

for (i=0;i<N;i++)
 for (j=0;j<N;j++)
  for (k=0;k<N;k++)
  C[i][j] += A[i][k] * B[k][j];

}

void MVM_default(){

int i,j;

for (i=0;i<N;i++)
for (j=0;j<N;j++){ 
Y[i]+=A[i][j]*X[j];
}

}






