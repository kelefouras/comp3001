/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

//THIS IS AN IMAGE PROCESSING APPLICATION WRITTEN IN A VERY INEFFICIENT WAY...

#include "string.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>	/* for clock_gettime */
#include <stdint.h>	/* for uint64 definition */

#define BILLION 1000000000L

#define TIMES 1

#define N 1024
#define M 1024

#define GB (1)
#define NB (8)

void initialization();
int compare_images();

unsigned char image_in[N][M];
unsigned char image_out[N][M];
int x_offset[NB]={1,1,1,0,0,-1,-1,-1};
int y_offset[NB]={1,0,-1,1,-1,1,0,-1};
int Gauss[]={99,68,35,10};
unsigned char gauss_x_image[N][M];
unsigned char gauss_xy_image[N][M];
unsigned char comp_edge_image[N][M];
unsigned char maxdiff_compute[N][M][NB+1];
unsigned short gauss_x_compute[N][M][(2*GB)+2],
     gauss_xy_compute[N][M][(2*GB)+2];
	
unsigned char maxl3(unsigned char a, unsigned char b);





unsigned char maxl3 (unsigned char a, unsigned char b)
{
  return a>b ? a : b;
}


/*
In this practical you need to reduce the execution time of the ineffcient_routine(). This is a routine written in an inefficient way.
First, read the code and understand what it does. Then, try to simplify the code, by eliminating redundant operation and by applying 
as many optimization techniques you can (have a look at the lectures). 

Some of the simplifications you can apply are:
1. the loop kernel in line 64 can be simplified
2. the if condition in line 69 can be eliminated 
3. gauss_x_compute[][][] and gauss_xy_compute[][][] are redundant and can be eliminated 
*/

void inefficient_routine ()
{
  unsigned char out_compute;
  int x,y,k;
  unsigned short tot=0;

  /*  GaussBlur(); */
  for (k=-GB; k<=GB; ++k)  //this loop kernel can be simplified
	  tot += Gauss[abs(k)];

  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {    //this if condition can be eliminated
      gauss_x_compute[x][y][0]=0;
      for (k=-GB; k<=GB; ++k)
        gauss_x_compute[x][y][GB+k+1] = gauss_x_compute[x][y][GB+k]
           + image_in[x+k][y]*Gauss[abs(k)];
      gauss_x_image[x][y]= gauss_x_compute[x][y][(2*GB)+1]/tot;
      }
     else
      gauss_x_image[x][y] = 0;

  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
      gauss_xy_compute[x][y][0]=0;
      for (k=-GB; k<=GB; ++k)
        gauss_xy_compute[x][y][GB+k+1] = gauss_xy_compute[x][y][GB+k] +
           gauss_x_image[x][y+k]*Gauss[abs(k)];
      gauss_xy_image[x][y]= gauss_xy_compute[x][y][(2*GB)+1]/tot;
      }
     else
      gauss_xy_image[x][y] = 0;

 /*  ComputeEdges(g_image, c_image); */
  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
      maxdiff_compute[x][y][0] = 0;
      for (k=0; k<=NB-1; ++k)
        maxdiff_compute[x][y][k+1] =
          maxl3(abs(gauss_xy_image[x+x_offset[k]][y+y_offset[k]]
                    - gauss_xy_image[x][y]), maxdiff_compute[x][y][k]);
      comp_edge_image[x][y] = maxdiff_compute[x][y][NB];
      }
     else
      comp_edge_image[x][y] = 0;
  
  /*  DetectRoots(c_image, out_image); */
  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
        out_compute = 255; 
        k = 0;
        while ((out_compute == 255) && (k <= NB-1)) {
          if (comp_edge_image[x+x_offset[k]][y+y_offset[k]] <
              comp_edge_image[x][y]) out_compute = 0;
          ++k; }
        image_out[x][y] = out_compute;
        }
      else
        image_out[x][y] = 0;
}





int main() {
  
struct timespec start, end; //timers
uint64_t diff;
int outcome,i;

  printf("\n\r Programmed started \n\r");

/* measure monotonic time */
clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */


	initialization();
	

    
	//YOU WILL OPTIMIZE THE FOLLOWING function
//--------------------------------
for (i=0;i<TIMES;i++)
  inefficient_routine();
//------------------------------------------------------------------------------
	
clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
printf("elapsed time = %llu mseconds\n", (long long unsigned int) diff/1000000);

    
outcome=compare_images();
	
	if (outcome==0)
		printf("\n\n\r -----  output is correct -----\n\r");
	else 
		printf("\n\n\r ----- output is INcorrect -----\n\r");

	return 0;
}


void initialization(){
	int i,j;
	
//This is a greyscale image with random values
// your flash memory doesn't support fopen() function	
	for (i=0;i<N;i++)
	 for (j=0;j<M;j++)
	  image_in[i][j]=rand()%255;
	
		for (i=0;i<N;i++)
	   for (j=0;j<M;j++)
	    image_out[i][j]=0;
	
}



//returns false/true, when the output image is incorrect/correct, respectively
int compare_images(){
	

  unsigned char out_compute;
  int x,y,k;
  unsigned short tot=0;


/* start layer 2 code */

  /*  GaussBlur(in_image, g_image); */
  for (k=-GB; k<=GB; ++k)  tot += Gauss[abs(k)];

  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
      gauss_x_compute[x][y][0]=0;
      for (k=-GB; k<=GB; ++k)
        gauss_x_compute[x][y][GB+k+1] = gauss_x_compute[x][y][GB+k]
           + image_in[x+k][y]*Gauss[abs(k)];
      gauss_x_image[x][y]= gauss_x_compute[x][y][(2*GB)+1]/tot;
      }
     else
      gauss_x_image[x][y] = 0;

  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
      gauss_xy_compute[x][y][0]=0;
      for (k=-GB; k<=GB; ++k)
        gauss_xy_compute[x][y][GB+k+1] = gauss_xy_compute[x][y][GB+k] +
           gauss_x_image[x][y+k]*Gauss[abs(k)];
      gauss_xy_image[x][y]= gauss_xy_compute[x][y][(2*GB)+1]/tot;
      }
     else
      gauss_xy_image[x][y] = 0;

 /*  ComputeEdges(g_image, c_image); */
  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
      maxdiff_compute[x][y][0] = 0;
      for (k=0; k<=NB-1; ++k)
        maxdiff_compute[x][y][k+1] =
          maxl3(abs(gauss_xy_image[x+x_offset[k]][y+y_offset[k]]
                    - gauss_xy_image[x][y]), maxdiff_compute[x][y][k]);
      comp_edge_image[x][y] = maxdiff_compute[x][y][NB];
      }
     else
      comp_edge_image[x][y] = 0;
  
  /*  DetectRoots(c_image, out_image); */
  for (x=0; x<N; ++x)
    for (y=0; y<M; ++y)
     if (x>=GB && x<=N-1-GB && y>=GB && y<=M-1-GB) {
        out_compute = 255; 
        k = 0;
        while ((out_compute == 255) && (k <= NB-1)) {
          if (comp_edge_image[x+x_offset[k]][y+y_offset[k]] <
              comp_edge_image[x][y]) out_compute = 0;
          ++k; }
        if (image_out[x][y] != out_compute)
					return -1;
        }
      else
          if (image_out[x][y] != 0)
					  return -1;
		
					
					return 0;
	}
	
