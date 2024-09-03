/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/
//compile with gcc Question1.c -o p -O3 -lm -Wall -fopenmp

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

#define TRIALS 1000000000 //you can amend this value if you want routine1() to run slower/faster.
#define N 1024*16
#define MAX_THREADS 16

double drandom();
void seed(double low_in, double hi_in);
void routine1(long long int num_trials);
void routine2();
void execute();
void initialize();
double u_exact ( double x, double y );

double A[N][N];

long long int ADDEND      = 150889;
unsigned long long MULTIPLIER  = 764261123;
unsigned long long PMOD        = 2147483647;
unsigned long long mult_n=0;
double random_low, random_hi;
unsigned long long pseed[MAX_THREADS][4]; //[4] to padd to cache line
unsigned long long random_last = 0;
#pragma omp threadprivate(random_last)


int main ( ) {

  double wtime;


  wtime = omp_get_wtime ( );

  routine1(TRIALS);

  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n   Routine 1 took %f secs \n", wtime );
  
  //-----------------------------------------------------------------------------------------
  
  wtime = omp_get_wtime ( );

  routine2();

  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n   Routine 2 took %f secs \n", wtime );

  return 0;
}

void routine1(long long int num_trials){

   long long int i;  long long int acc = 0,no=0;
   double pi, x, y, test;
   double r = 1.0;   // radius of circle. Side of squrare is 2*r 

//START THE PARALLEL REGION

   seed(-r, r);  // The circle and square are centered at the origin
   
   //PARALLELIZE THIS LOOP
   for(i=0;i<num_trials; i++)
   {
      x = drandom(); //generates a random value; this is different every time
      y = drandom(); //generates a random value; this is different every time

      test = x*x + y*y;

      if (test <= r*r) {
       acc=acc+1;
       }
       else {
       no++;
       }
       
    }

//END THE PARALLEL REGION

    pi = 4.0 * ((double) acc / (double)num_trials);

    printf("\n Routine 1 : after %lld trials, pi is %lf - no is %lld\n",num_trials, pi, no);

}


//DO NOT AMEND THIS ROUTINE
double drandom()
{
    unsigned long long random_next;
    double ret_val;

// 
// compute an integer random number from zero to mod
//
    random_next = (unsigned long long)((mult_n  * random_last)% PMOD);
    random_last = random_next;

//
// shift into preset range
//
    ret_val = ((double)random_next/(double)PMOD)*(random_hi-random_low)+random_low;
    return ret_val;
}

//DO NOT AMEND THIS ROUTINE
// set the seed and the range
void seed(double low_in, double hi_in)
{
   int i, id, nthreads;
   unsigned long long iseed;
   id = omp_get_thread_num();

   #pragma omp single
   {
      if(low_in < hi_in)
      { 
         random_low = low_in;
         random_hi  = hi_in;
      }
      else
      {
         random_low = hi_in;
         random_hi  = low_in;
      }
  
//
// The Leapfrog method ... adjust the multiplier so you stride through
// the sequence by increments of "nthreads" and adust seeds so each 
// thread starts with the right offset
//

      nthreads = omp_get_num_threads();
      iseed = PMOD/MULTIPLIER;     // just pick a reasonable seed
      pseed[0][0] = iseed;
      mult_n = MULTIPLIER;
      for (i = 1; i < nthreads; ++i)
      {
	iseed = (unsigned long long)((MULTIPLIER * iseed) % PMOD);
	pseed[i][0] = iseed;
	mult_n = (mult_n * MULTIPLIER) % PMOD;
      }

   }
   random_last = (unsigned long long) pseed[id][0];
}

//PARALLELIZE AND VECTORIZE THIS ROUTINE USING OPENMP
void initialize(){

unsigned int i,j;
double x;

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
  x=(double) (i%77) * (j%83) + 0.024;
  A[i][j]=sqrt(x);
  }
  }
  
}

//PARALLELIZE AND VECTORIZE THIS ROUTINE USING OPENMP
void execute(){

double  x,y,u_true;
double u_true_norm=0.0,error_norm = 0.0,u_norm=0.0;

unsigned int i,j;

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
      u_norm += A[i][j]*A[i][j];
    }
  }
  
  u_norm = sqrt ( u_norm );
  

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
      x = ( double ) ( 2 * i - N + 1 ) / ( double ) ( N - 1 );
      y = ( double ) ( 2 * j - N + 1 ) / ( double ) ( N - 1 );
      u_true = u_exact ( x, y );
      error_norm += sqrt( ( A[i][j] - u_true ) * ( A[i][j] - u_true ) );
      u_true_norm += u_true * u_true;
    }
  }
  
    error_norm = sqrt ( error_norm );
  u_true_norm = sqrt ( u_true_norm );
  
  
  for (i=0;i<N;i++){
   for(j=0;j<N;j++){
    A[i][j]-=x;
  }
  }
  
  printf("\n Routine2 : output is %e, %e and %e\n",u_norm,error_norm,u_true_norm);
  
}


double u_exact ( double x, double y ) {
  double value;

  value = ( 1.0 - x * x ) * ( 1.0 - y * y );

  return value;
}

void routine2(){

initialize();
execute();

}



