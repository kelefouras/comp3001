#include "canny.h"

unsigned char filt[N][M];
unsigned char gaussianMask[3][3];

void Gaussian_Blur_3x3(){


int i,j;
	unsigned int    row, col;		
	int rowOffset;					
	int colOffset;					
	int Gx;						
	int Gy;							
	float thisAngle;				
	int newAngle;								
	int newPixel;				
			
        unsigned char temp;					



/* Declare Gaussian mask */
gaussianMask[0][0] = 1;
gaussianMask[0][1] = 2;
gaussianMask[0][2] = 1;


gaussianMask[1][0] = 2;
gaussianMask[1][1] = 4;
gaussianMask[1][2] = 2;


gaussianMask[2][0] = 1;
gaussianMask[2][1] = 2;
gaussianMask[2][2] = 1;





/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 1; row < N-1; row++) {
		for (col = 1; col < M-1; col++) {
			newPixel = 0;
			for (rowOffset=-1; rowOffset<=1; rowOffset++) {
				for (colOffset=-1; colOffset<=1; colOffset++) {
					
                   newPixel += frame1[row+rowOffset][col+colOffset] * gaussianMask[1 + rowOffset][1 + colOffset];
				}
			        }
		filt[row][col] = (unsigned char) (newPixel / 16);
		}
	}



for (i=0;i<N;i++)
 for (j=0;j<M;j++)
  print[i][j]=filt[i][j];

write_image2(OUT_NAME1,print);





}


/*
void Gaussian_Blur_default_unrolled() {

    short int row, col;
    short int newPixel;

    for (row = 1; row < N - 1; row++) {
        for (col = 1; col < M - 1; col++) {
            newPixel = 0;

            newPixel += in_image[row - 2][col - 1] * gaussianMask[0][0];
            newPixel += in_image[row - 2][col ] * gaussianMask[0][1];
            newPixel += in_image[row - 2][col+1] * gaussianMask[0][2];

            newPixel += in_image[row - 1][col - 1] * gaussianMask[1][0];
            newPixel += in_image[row - 1][col ] * gaussianMask[1][1];
            newPixel += in_image[row - 1][col+1] *  gaussianMask[1][2];

            newPixel += in_image[row][col - 1] * gaussianMask[2][0];
            newPixel += in_image[row][col ] * gaussianMask[2][1];
            newPixel += in_image[row][col+1] * gaussianMask[2][2];


            filt_image[row][col] = newPixel / 16;


        }
    }

} 
*/



