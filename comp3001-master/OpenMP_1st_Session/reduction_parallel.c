

//gcc example2.c -o p -O2 -fopenmp


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>
#include <stdbool.h> 
 #include <sys/time.h>

#define NUM_THREADS 4
#define N 100

int A[N], B[N], Acopy1[N];

void init();
void un_opt();
void  version1();
void  version2();
void  version3();
void  version4();
bool compare();

int main(){
  
time_t start1, end1;
struct timeval start2, end2;
double pi1,pi2;

init();

start1 = clock();
gettimeofday(&start2, NULL);

for (int i=0; i<1; i++){
	//un_opt();
	//version1();
	//version2();
	version3();

}

end1 = clock();
gettimeofday(&end2, NULL);
printf(" clock() method: %ldms\n", (end1 - start1) / (CLOCKS_PER_SEC/1000));
printf(" gettimeofday() method: %ldms\n", (end2.tv_sec - start2.tv_sec) *1000 + (end2.tv_usec - start2.tv_usec)/1000);



}


void init(){

int i;

for (i=0; i<N; i++){
A[i]=rand()%50;
B[i]=rand()%1000;
Acopy1[i]=A[i];
}

}

//this is the serial version of the program you need to parallelize
void un_opt(){

int ave=0;
int i;

for (i=0; i<N; i++){
 ave+=A[i];
}

printf("\nave=%d\n",ave);
}


//THIS IS THE RECOMMENDED IMPLEMENTATION
void version1(){

int ave=0;
int i;

#pragma omp parallel for reduction(+:ave)
for (i=0; i<N; i++){
 ave+=Acopy1[i];
}

printf("\nave=%d\n",ave);
}


//THIS IS SIMILAR TO VERSION1, BUT WE PRINT THE NUMBER OF THREADS USED
void version2(){

int ave=0;
int i,threads;

#pragma omp parallel 
{
int t=omp_get_num_threads();
int id=omp_get_thread_num();
if (id==0)
  threads=t;

#pragma omp for reduction(+:ave)
for (i=0; i<N; i++){
 ave+=Acopy1[i];
}

}
printf("\nave=%d\n",ave);
printf("\nthreads=%d\n",threads);
}


//THIS IS SIMILAR TO VERSION2, BUT INSTEAD OF USING THE REDUCTION CLAUSE, THE CRITICAL CLAUSE IS USED INSTEAD
void version3(){

int average=0;
int i,threads;

#pragma omp parallel 
{
int t=omp_get_num_threads();
int id=omp_get_thread_num();
if (id==0)
  threads=t;

int ave=0;
#pragma omp for
for (i=0; i<N; i++){
 ave+=Acopy1[i];
}

#pragma omp critical
average+=ave;

}
printf("\nave=%d\n",average);
printf("\nthreads=%d\n",threads);
}





