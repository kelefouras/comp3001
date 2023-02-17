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

#define TRIALS 1000000000 //you can reduce this value if you want routine1() to run faster.
#define N 1024*16

double drandom();
void seed(double low_in, double hi_in);
void routine1(long long int num_trials);
void routine2();
void execute();
void initialize();
double u_exact ( double x, double y );

double A[N][N]  __attribute__((aligned(32)));

long long int MULTIPLIER  = 1366;
long long int ADDEND      = 150889;
long long int PMOD        = 714025;
long long int random_last = 0;
double random_low, random_hi;

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

   long long int i;  long long int Ncirc = 0;
   double pi, x, y, test;
   double r = 1.0;   // radius of circle. Side of squrare is 2*r 

   seed(-r, r);  // The circle and square are centered at the origin
   
   for(i=0;i<num_trials; i++)
   {
      x = drandom(); //generates a random value; this is different every time
      y = drandom(); //generates a random value; this is different every time

      test = x*x + y*y;

      if (test <= r*r) 
       Ncirc++;
    }

    pi = 4.0 * ((double)Ncirc/(double)num_trials);

    printf("\n Routine 1 : after %lld trials, pi is %lf \n",num_trials, pi);

}


double drandom()
{
    long long int random_next;
    double ret_val;

// 
// compute an integer random number from zero to mod
//
    random_next = (MULTIPLIER  * random_last + ADDEND)% PMOD;
    random_last = random_next;

//
// shift into preset range
//
    ret_val = ((double)random_next/(double)PMOD)*(random_hi-random_low)+random_low;
    return ret_val;
}

//
// set the seed and the range
//
void seed(double low_in, double hi_in)
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
   random_last = PMOD/ADDEND;  // just pick something

}


void initialize(){

unsigned int i,j;
double x;

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
  x=(double) (i%99) * (j%87) + 0.043;
  A[i][j]=sqrt(x);
  }
  }
  
}

void execute(){

double  x,y,u_true;
double u_true_norm=0.0,error_norm = 0.0,u_norm=0.0;

unsigned int i,j;

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
      u_norm = u_norm + A[i][j]*A[i][j];
    }
  }
  
  u_norm = sqrt ( u_norm );
  

for (i=0;i<N;i++){
 for(j=0;j<N;j++){
      x = ( double ) ( 2 * i - N + 1 ) / ( double ) ( N - 1 );
      y = ( double ) ( 2 * j - N + 1 ) / ( double ) ( N - 1 );
      u_true = u_exact ( x, y );
      error_norm = error_norm + sqrt( ( A[i][j] - u_true ) * ( A[i][j] - u_true ) );
      u_true_norm = u_true_norm + u_true * u_true;
    }
  }
  
    error_norm = sqrt ( error_norm );
  u_true_norm = sqrt ( u_true_norm );
  
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



