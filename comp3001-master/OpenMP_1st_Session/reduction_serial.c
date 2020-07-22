

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

int main(){
  
time_t start1, end1;
struct timeval start2, end2;
double pi1,pi2;

init();

start1 = clock();
gettimeofday(&start2, NULL);

for (int i=0; i<1; i++){
	un_opt();


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







