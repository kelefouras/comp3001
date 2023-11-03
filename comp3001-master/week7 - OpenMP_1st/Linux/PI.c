/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//------------------------- PI PROGRAM ------------------------

//gcc PI.c -o p -O2  -fopenmp


#include <math.h>
#include <stdio.h>
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>
 #include <sys/time.h>
#include <stdint.h>	/* for uint64 definition */


#define NUM_THREADS 4
#define PAD 8

#define BILLION 1000000000L
static long num_steps = 1000000000;

double un_opt();
double version1();
double version2();
double version3();
double version4();
double version5();
double version6();

int main(){
  

double pi1,pi2;
struct timespec start, end; //timers
uint64_t diff;

clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

for (int i=0; i<5; i++){
	//pi1=un_opt();
	pi1=version1();
	//pi1=version2();
	//pi1=version3();
	//pi1=version6();
}

clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);


printf("\n pi=%.12f\n",pi1);

}


//this is the serial version of the PI program
double un_opt(){

int i;
double x, pi, sum=0.0;
double step;
  
  step=1.0/(double) num_steps;

 for (i=0; i<num_steps; i++){
  x=(i+0.5)*step;
  sum = sum + 4.0 / (1.0 + x*x);
}
pi = step * sum;

return pi;

}

//THIS IS THE 1ST PARALLEL VERSION - THIS IMPLEMENTATION IS NOT THE FASTEST, BECAUSE OF THE FALSE SHARING in sum[] array
double version1(){
   
double step;

int j, nthreads;   
double pi=0.0, sum[NUM_THREADS];

step=1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);

#pragma omp parallel 
{
   int i,id,nthrds; //local data
   double x;        //local data
   id=omp_get_thread_num(); 
   nthrds = omp_get_num_threads();
   if (id==0) nthreads=nthrds; //save a copy of num of threads as the enviroment might choose to give us less threads than requested. What if I ask 1000 threads?

   for (i=id, sum[id]=0.0; i<num_steps; i=i+nthrds){
      x=(i+0.5)*step;
      sum[id] = sum[id] + 4.0 / (1.0 + x*x);
   }
}

//after this point I have lost the local variables. So, for the sum to be visible, I must promote sum to an array.
for (j=0, pi=0.0; j<nthreads; j++)
  pi+=sum[j] * step;


return pi;

}

//THIS IS THE 2nd PARALLEL VERSION - It addresses false sharing, but we need to know the cache line size. What about changing machine?
double version2(){
   
double step;

int j, nthreads;   
double pi=0.0, sum[NUM_THREADS][PAD];

step=1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);

#pragma omp parallel 
{
   int i,id,nthrds; //local data
   double x;        //local data
   id=omp_get_thread_num(); 
   nthrds = omp_get_num_threads();
   if (id==0) nthreads=nthrds; //save a copy of num of threads as the enviroment might choose to give us less threads than requested. What if I ask 1000 threads?

   for (i=id, sum[id][0]=0.0; i<num_steps; i=i+nthrds){
      x=(i+0.5)*step;
      sum[id][0] = sum[id][0] + 4.0 / (1.0 + x*x);
   }
}

//after this point I have lost the local variables. So, for the sum to be visible, I promote sum to an array.
for (j=0, pi=0.0; j<nthreads; j++)
  pi+=sum[j][0] * step;


return pi;

}

//THIS IS THE 3rd PARALLEL VERSION - 
//better version, without using arrays. It is PORTABLE
double version3(){
   
double step;

int nthreads;   
double pi=0.0;

step=1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);

#pragma omp parallel 
{
   int i,id,nthrds; 		//local data
   double x, sum=0.0;        //sum is now local, each thread has its own copy
   id=omp_get_thread_num(); 
   nthrds = omp_get_num_threads();
   if (id==0) nthreads=nthrds; //save a copy of num of threads as the enviroment might choose to give us less threads than requested. What if I ask 1000 threads?

   for (i=id; i<num_steps; i=i+nthrds){
      x=(i+0.5)*step;
      sum = sum + 4.0 / (1.0 + x*x);
   }

   #pragma omp critical //mutual exclusion. only one thread at a time can enter this block. I could use 'atomic' instead of 'critical' it is the same
    {
    pi+=sum * step;
   }

}


return pi;

}

//THIS IS THE 4th PARALLEL VERSION 
//It is a SLOW 	alternative of version3, as the critical section includes most of the loop calculations, so it is almost serial computation. 
double version4(){
   
double step;

int nthreads;   
double pi=0.0;

step=1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);

#pragma omp parallel 
{
   int i,id,nthrds; 		//local data
   double x;        
   id=omp_get_thread_num(); 
   nthrds = omp_get_num_threads();
   if (id==0) nthreads=nthrds; //save a copy of num of threads as the enviroment might choose to give us less threads than requested. What if I ask 1000 threads?

   for (i=id; i<num_steps; i=i+nthrds){
      x=(i+0.5)*step;
      #pragma omp critical
      pi+= 4.0 / (1.0 + x*x);
   }


}

pi*=step;
return pi;

}

//THIS IS THE 5th PARALLEL VERSION - This version is easier to write. Here lies the power of OpenMP
double version5(){

int i;
double pi, sum=0.0;
double step;
  
  step=1.0/(double) num_steps;

#pragma omp parallel 
{
 double x;
#pragma omp for reduction(+:sum) //Each thread has its own copy of sum. Each thread does its own summation and when they are done, they are combined with the global copy of sum. 
 for (i=0; i<num_steps; i++){
  x=(i+0.5)*step;
  sum = sum + 4.0 / (1.0 + x*x);
}
}
pi = step * sum;


return pi;

}

//THIS IS THE 6th PARALLEL VERSION - Very similar to version5. This version is easier to write. Here lies the power of OpenMP
//Keep in mind that compilers skip a pragma when they do not understand it. So, this version runs perfectly fine even the pragma is not recognized. This makes OpenMP elegant
double version6(){

int i;
double x, pi, sum=0.0;
double step;
  

  step=1.0/(double) num_steps;
#pragma omp parallel for private(x) reduction(+:sum) //x is not defined inside the parallel region. Thus by default it is a shared variable. private(x) creates a private x variable in each thread. Be Careful: x is unitialized no matter what its previous value is. 
 for (i=0; i<num_steps; i++){
  x=(i+0.5)*step;
  sum = sum + 4.0 / (1.0 + x*x);
}
pi = step * sum;

return pi;

}





