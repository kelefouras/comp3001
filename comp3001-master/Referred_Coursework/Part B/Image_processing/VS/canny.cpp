

#include "canny.h"

unsigned char filt[N][M];
unsigned char gaussianMask[3][3];




int Gaussian_Blur_3x3() {


	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;




	/* Declare the Gaussian mask */
	gaussianMask[0][0] = 1;
	gaussianMask[0][1] = 2;
	gaussianMask[0][2] = 1;


	gaussianMask[1][0] = 2;
	gaussianMask[1][1] = 4;
	gaussianMask[1][2] = 2;


	gaussianMask[2][0] = 1;
	gaussianMask[2][1] = 2;
	gaussianMask[2][2] = 1;


	


	/*---------------------- read 1st frame -----------------------------------*/
	frame1 = (unsigned char**)malloc(N * sizeof(unsigned char *));
	if (frame1 == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		frame1[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (frame1[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}



	print = (unsigned char**)malloc(N * sizeof(unsigned char *));
	if (print == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		print[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (print[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = 0;

	read_image(IN, frame1);

	   	  
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




	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = filt[i][j];

	write_image(OUT_NAME1, print);

	



	for (i = 0; i < N; i++)
		free(frame1[i]);
	free(frame1);



	for (i = 0; i < N; i++)
		free(print[i]);
	free(print);


	return 0;

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





void read_image(char filename[], unsigned char **image)
{
	int inint = -1;
	int c;
	FILE *finput;
	int i, j;

	printf("  Reading image from disk (%s)...\n", filename);
	//finput = NULL;
	openfile(filename, &finput);


	for (j = 0; j < N; j++)
		for (i = 0; i < M; i++) {
			c = getc(finput);


			image[j][i] = (unsigned char)c;
		}



	/* for (j=0; j<N; ++j)
	   for (i=0; i<M; ++i) {
		 if (fscanf(finput, "%i", &inint)==EOF) {
		   fprintf(stderr,"Premature EOF\n");
		   exit(-1);
		 } else {
		   image[j][i]= (unsigned char) inint; //printf("\n%d",inint);
		 }
	   }*/

	fclose(finput);

}





void write_image(char* filename, unsigned char **image)
{
	FILE* foutput;
	errno_t err;
	int i, j;


	printf("  Writing result to disk (%s)...\n", filename);
	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		printf("Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", image[j][i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}










void openfile(char *filename, FILE** finput)
{
	int x0, y0;
	errno_t err;
	char header[255];
	int aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		printf("Unable to open file %s for reading\n");
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	/*if (strcmp(header,"P2")!=0) {
	   fprintf(stderr,"\nFile %s is not a valid ascii .pgm file (type P2)\n",
			   filename);
	   exit(-1);
	 }*/

	x0 = getint(*finput);
	y0 = getint(*finput);





	if ((x0 != M) || (y0 != N)) {
		printf("Image dimensions do not match: %ix%i expected\n", N, M);
		exit(-1);
	}

	x0 = getint(*finput); /* read and throw away the range info */


}


int getint(FILE *fp) /* adapted from "xv" source code */
{
	int c, i, firstchar, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], *sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != ',') garbage = 1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c<'0' || c>'9') break;
	}
	return i;
}



