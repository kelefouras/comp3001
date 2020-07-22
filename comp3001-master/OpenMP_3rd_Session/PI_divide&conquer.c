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


#define BULK 1000000
#define BILLION 1000000000L
static long num_steps = 1000000000;

double un_opt();
double divide_conquer();
inline double pi_kernel(int start, int finish, double step);
double pi_loop();


int main(){
  
double start_time, run_time;
double pi1,pi2;
struct timespec start, end; //timers
uint64_t diff;

pi1=un_opt();

clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
start_time = omp_get_wtime();

pi2=divide_conquer();
//pi2=pi_loop();

run_time = omp_get_wtime() - start_time;
clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);
printf("\n omp time is %f seconds \n", run_time);

printf("\n serial pi = %.12f - divide&conquer pi = %.12f\n",pi1,pi2);

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


double divide_conquer(){

int i;
double step, pi, sum=0.0;
  
step=1.0/(double) num_steps;

#pragma omp parallel 
{
#pragma omp single 
sum=pi_kernel(0,num_steps,step);
}

pi = step * sum;

return pi;

}

//it breaks the [0,num_steps] iterations into many tasks
double pi_kernel(int start, int finish, double step){
int i,blk;
double x,sum=0.0,sum1,sum2;

printf("Thread %d starting...\n", omp_get_thread_num());

if (finish-start < BULK){
 for (i=start; i<finish; i++){
  x=(i+0.5)*step;
  sum = sum + 4.0 / (1.0 + x*x);
	} 
}
else {
 blk=finish-start;

 #pragma omp task shared(sum1)
 sum1=pi_kernel(start,finish-blk/2, step);

 #pragma omp task shared(sum2)
 sum2=pi_kernel(finish-blk/2, finish, step);

 #pragma omp taskwait
 sum=sum1+sum2;
}

return sum;
}


double pi_loop(){

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





