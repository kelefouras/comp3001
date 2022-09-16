#include "canny.h"

unsigned char filt[N][M], gradient[N][M],grad2[N][M],edgeDir[N][M];
unsigned char gaussianMask[5][5];
signed char GxMask[3][3],GyMask[3][3];

void GaussianBlur(){


int i,j;
	unsigned int    row, col;		
	int rowOffset;					
	int colOffset;					
						
	int newPixel;				
			
        unsigned char temp;					



/* Declare Gaussian mask */
gaussianMask[0][0] = 2;

gaussianMask[0][1] = 4;
gaussianMask[0][2] = 5;
gaussianMask[0][3] = 4;
gaussianMask[0][4] = 2;

gaussianMask[1][0] = 4;
gaussianMask[1][1] = 9;
gaussianMask[1][2] = 12;
gaussianMask[1][3] = 9;
gaussianMask[1][4] = 4;	

gaussianMask[2][0] = 5;
gaussianMask[2][1] = 12;
gaussianMask[2][2] = 15;
gaussianMask[2][3] = 12;
gaussianMask[2][4] = 5;	

gaussianMask[3][0] = 4;
gaussianMask[3][1] = 9;
gaussianMask[3][2] = 12;
gaussianMask[3][3] = 9;
gaussianMask[3][4] = 4;	

gaussianMask[4][0] = 2;
gaussianMask[4][1] = 4;
gaussianMask[4][2] = 5;
gaussianMask[4][3] = 4;
gaussianMask[4][4] = 2;	


/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 2; row < N-2; row++) {
		for (col = 2; col < M-2; col++) {
			newPixel = 0;
			for (rowOffset=-2; rowOffset<=2; rowOffset++) {
				for (colOffset=-2; colOffset<=2; colOffset++) {
					
                   newPixel += frame1[row+rowOffset][col+colOffset] * gaussianMask[2 + rowOffset][2 + colOffset];
				}
			        }
		filt[row][col] = (unsigned char) (newPixel / 159);
		}
	}


}



void Sobel(){


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



/* Declare Sobel masks */
	GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;
	
	GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	GyMask[1][0] =  0; GyMask[1][1] =  0; GyMask[1][2] =  0;
	GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;


	/*---------------------------- Start of Sobel  -------------------------------------------*/
	for (row = 1; row < N-1; row++) {
		for (col = 1; col < M-1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset=-1; rowOffset<=1; rowOffset++) {
				for (colOffset=-1; colOffset<=1; colOffset++) {
					
					Gx += filt[row+rowOffset][col+colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[row+rowOffset][col+colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			gradient[row][col] = (unsigned char) (sqrt(Gx*Gx + Gy*Gy));	
			
			thisAngle = ( (( atan2(Gx,Gy)) /3.14159) * 180.0);	

			/* Convert actual edge direction to approximate value */
			if ( ( (thisAngle >= -22.5) && (thisAngle <= 22.5) ) || (thisAngle >= 157.5) || (thisAngle <= -157.5) )
				newAngle = 0;
			else if ( ( (thisAngle > 22.5) && (thisAngle < 67.5) ) || ( (thisAngle > -157.5) && (thisAngle < -112.5) ) )
				newAngle = 45;
			else if ( ( (thisAngle >= 67.5) && (thisAngle <= 112.5) ) || ( (thisAngle >= -112.5) && (thisAngle <= -67.5) ) )
				newAngle = 90;
			else if ( ( (thisAngle > 112.5) && (thisAngle < 157.5) ) || ( (thisAngle > -67.5) && (thisAngle < -22.5) ) )
				newAngle = 135;
				

			edgeDir[row][col] = newAngle;		
		}
	}
	/*---------------------------- End of Sobel  -------------------------------------------*/

}




void image_detection(){


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



GaussianBlur();


for (i=0;i<N;i++)
 for (j=0;j<M;j++)
  print[i][j]=filt[i][j];

write_image2(OUT_NAME1,print);


Sobel();


/* write gradient to image*/

for (i=0;i<N;i++)
for (j=0;j<M;j++)
print[i][j]= gradient[i][j];

write_image2(OUT_NAME2,print);




}





