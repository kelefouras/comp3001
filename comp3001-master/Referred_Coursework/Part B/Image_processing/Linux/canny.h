
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define IN   "rec.pgm"


#define OUT "out.pgm"
#define OUT_NAME1 "out1.pgm"
#define OUT_NAME2 "out2.pgm"
#define OUT_NAME3 "out3.pgm"
#define OUT_NAME4 "out4.pgm"

#define	N 1024
#define M 1024




#define	 UpThr  30	
#define	 LwThr  10	

void Gaussian_Blur_3x3();

void write_acc(char* filename,int angl);
void read_frame(char filename[80], unsigned char image[N][M]);
void read_image(char* filename, unsigned char **image);

void write_image2(char* filename, unsigned char **imag);
void write_image(char *filename, unsigned char image[N][M]);
void openfile(char *filename, FILE** finput);
int getint(FILE *fp);



void printt(int tem, int i, int j, int max1, int max2);

void trace_edges(int x0, int y0, int x, int y);
int sub_frames( int dtheta);



unsigned char **frame1;
unsigned char **print;




