//compile with gcc main.c canny.c -o p -O2 -lm


#include "canny.h"




int main(){


int out,i,j;




       /*---------------------- read 1st frame -----------------------------------*/
            frame1 = malloc(N * sizeof(unsigned char *));
              if (frame1==NULL) {printf("\nerror with malloc fr"); return -1;}
                 for (i=0;i<N;i++){
                 frame1[i]=malloc(M * sizeof(unsigned char));
                 if (frame1[i]==NULL) {printf("\nerror with malloc fr"); return -1;}
                          	}

           

                    print = malloc(N * sizeof(unsigned char *));
                     if (print==NULL) {printf("\nerror with malloc fr"); return -1;}
                        for (i=0;i<N;i++){
                        	print[i]=malloc(M * sizeof(unsigned char));
                        if (print[i]==NULL) {printf("\nerror with malloc fr"); return -1;}
                                 	}

for (i=0;i<N;i++)
 for (j=0;j<M;j++)
  print[i][j]=0;

read_image(IN,frame1);

image_detection();


  //write_image2(OUT, print);

for (i=0;i<N;i++)
	free(frame1[i]);
free(frame1);



for (i=0;i<N;i++)
	free(print[i]);
free(print);


    return 0;
}





void read_image(char* filename, unsigned char **image)
{

 int c;
  FILE *finput;
  int i,j;

  printf("\nReading image from disk ...\n");
  finput=NULL;
  openfile(filename,&finput);


for (j=0; j<N; j++){

for (i=0; i<M; i++) {
c=getc(finput);

 image[j][i]= (unsigned char) c;
}
}


  fclose(finput);

}

void read_frame(char filename[80], unsigned char image[N][M])
{

 int c;
  FILE *finput;
  int i,j;

  printf("  Reading image from disk...\n");
  finput=NULL;
  openfile(filename,&finput);


for (j=0; j<N; j++)
for (i=0; i<M; i++) {
c=getc(finput);

 image[j][i]= (unsigned char) c;
}



  fclose(finput);

}




void write_image(char* filename, unsigned char image[N][M])
{
  FILE* foutput;
  int i,j;




  printf("  Writing result to disk ...\n");

  if ((foutput=fopen(filename,"wb"))==NULL) {
    fprintf(stderr,"Unable to open file %s for writing\n",filename);
    exit(-1);
  }

  fprintf(foutput,"P2\n");
  fprintf(foutput,"%d %d\n",M,N);
  fprintf(foutput,"%d\n",255);

  for (j=0; j<N; ++j) {
    for (i=0; i<M; ++i) {
      fprintf(foutput,"%3d ",image[j][i]);
      if (i%32==31) fprintf(foutput,"\n");
    }
    if (M%32!=0) fprintf(foutput,"\n");
  }
  fclose(foutput);


}




void write_image2(char* filename, unsigned char **image)
{

  FILE* foutput;
  int i,j;



  printf("  Writing result to disk ...\n");

  if ((foutput=fopen(filename,"wb"))==NULL) {
    fprintf(stderr,"Unable to open file %s for writing\n",filename);
    exit(-1);
  }

  fprintf(foutput,"P2\n");
  fprintf(foutput,"%d %d\n",M,N);
  fprintf(foutput,"%d\n",255);

  for (j=0; j<N; ++j) {
    for (i=0; i<M; ++i) {
      fprintf(foutput,"%3d ",image[j][i]);
      if (i%32==31) fprintf(foutput,"\n");
    }
    if (M%32!=0) fprintf(foutput,"\n");
  }
  fclose(foutput);


}


void write_frame(char *filename, unsigned char image[N][M])
{

  FILE* foutput;
  int i,j;



  printf("  Writing result to disk ...\n");

  if ((foutput=fopen(filename,"wb"))==NULL) {
    fprintf(stderr,"Unable to open file %s for writing\n",filename);
    exit(-1);
  }

  fprintf(foutput,"P2\n");
  fprintf(foutput,"%d %d\n",M,N);
  fprintf(foutput,"%d\n",255);

  for (j=0; j<N; ++j) {
    for (i=0; i<M; ++i) {
      fprintf(foutput,"%3d ",image[j][i]);
      if (i%32==31) fprintf(foutput,"\n");
    }
    if (M%32!=0) fprintf(foutput,"\n");
  }
  fclose(foutput);


}


void openfile(char *filename, FILE** finput)
{
  int x0, y0;
  char header[255];
int aa;

  if ((*finput=fopen(filename,"rb"))==NULL) {
    fprintf(stderr,"Unable to open file %s for reading\n",filename);
    exit(-1);
  }

  aa=fscanf(*finput,"%s",header);


  x0=getint(*finput);
  y0=getint(*finput);





  if ((x0!=M) || (y0!=N)) {
    fprintf(stderr,"Image dimensions do not match: %ix%i expected\n", N, M);
    exit(-1);
  }

  x0=getint(*finput); /* read and throw away the range info */


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
    if (c=='#') {
      /* if we're at a comment, read to end of line */
      char cmt[256], *sp;

      sp = cmt;  firstchar = 1;
      while (1) {
        c=getc(fp);
        if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
        else {
          if (c == '\n' || c == EOF) break;
          if ((sp-cmt)<250) *sp++ = c;
        }
      }
      *sp++ = '\n';
      *sp   = '\0';
    }

    if (c==EOF) return 0;
    if (c>='0' && c<='9') break;   /* we've found what we were looking for */

    /* see if we are getting garbage (non-whitespace) */
    if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',') garbage=1;

    c = getc(fp);
  }

  /* we're at the start of a number, continue until we hit a non-number */
  i = 0;
  while (1) {
    i = (i*10) + (c - '0');
    c = getc(fp);
    if (c==EOF) return i;
    if (c<'0' || c>'9') break;
  }
  return i;
}







float getfloat(FILE *fp) /* adapted from "xv" source code */
{
  int c, firstchar, garbage,dec,num_psifia;
  float i;


  /* note:  if it sees a '#' character, all characters from there to end of
     line are appended to the comment string */

  /* skip forward to start of next number */
  c = getc(fp);
  while (1) {
    /* eat comments */
    if (c=='#') {
      /* if we're at a comment, read to end of line */
      char cmt[256], *sp;

      sp = cmt;  firstchar = 1;
      while (1) {
        c=getc(fp);
        if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
        else {
          if (c == '\n' || c == EOF) break;
          if ((sp-cmt)<250) *sp++ = c;
        }
      }
      *sp++ = '\n';
      *sp   = '\0';
    }

    if (c==EOF) return 0;
    if ((c>='0' && c<='9') || (c == '.') || (c == '-') )break;   /* we've found what we were looking for */

    /* see if we are getting garbage (non-whitespace) */
    if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',') garbage=1;

    c = getc(fp);
  }


  if (c == '-') {
	  c=getc(fp);
	  i=-(c-'0');
	  return i;
  }


  if (c != '.'){
  /* we're at the start of a number, continue until we hit a non-number */
  i = 0;
  while (1) {
    i = (i*10) + (c - '0');
    c = getc(fp);
    if (c==EOF) return i;
    if (c<'0' || c>'9') break;
  }
  }

 if (c == '.'){
	  c = getc(fp);
	  dec = 0;
	  num_psifia=1;
	  while (1) {
	    dec = (dec*10) + (c - '0');
	    c = getc(fp);
	    if (c==EOF) return i;
	    if (c<'0' || c>'9') break;
	    num_psifia++;
	  }
i = i + (dec / ( (float) pow(10,num_psifia)) );

  }

 return i;

}



