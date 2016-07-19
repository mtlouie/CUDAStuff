#include "image.h"
#include <stdio.h>

void CompareImage(ImageData srcImage, ImageData ResampledTest, int img, 
		  int *index_array, long *min_rms_array, int nx_dim, int ny_dim, 
		  int do_grayscale)
/* int * orientation_array, */
{
  long score;
  long r, g, b;
  Pixel *srctmp, *testtmp;
  int isrc, itest;
  int nx_src=srcImage.xDim;
  int ny_src=srcImage.yDim;
  int nx_tile=ResampledTest.xDim;
  int ny_tile=ResampledTest.yDim;
  int ioffset;
  int icol, jrow;
  int i, j, itile, jtile;

  /* Loop over the tiles in the final image. */
  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      ioffset=i+j*nx_dim;
      score=0L;
      /* Loop over the contents of the tile. */
      for(jtile=0; jtile<ny_tile; jtile++) {
	for(itile=0; itile<nx_tile; itile++) {
	  /* Pixels are stored in row major order. */
  	  itest=itile+jtile*nx_tile;
	  jrow=jtile+j*ny_tile;
	  icol=itile+i*nx_tile;
	  isrc=icol+jrow*nx_src;
	  /* Compute score if in bounds. */
	  if(isrc<(nx_src*ny_src)) {
	    srctmp=(srcImage.pixels)+isrc;
	    testtmp=(ResampledTest.pixels)+itest;
	    r=(srctmp->R)-(testtmp->R);
	    g=(srctmp->G)-(testtmp->G);
	    b=(srctmp->B)-(testtmp->B);
	    score+=(do_grayscale==0)?(r*r+g*g+b*b):((r+g+b)*(r+g+b));
	  }
	}
      }
      /* If the current tile is a better fit, save it and its score. */
      if(score<min_rms_array[ioffset]) {
	index_array[ioffset]=img;
	min_rms_array[ioffset]=score;
      }
    }
  }
}

void ReplaceInImage(int img, int *index_array,  
		    int nx_dim, int ny_dim, 
		    ImageData FinalImage, ImageData ResampledTest,
		    int do_grayscale) 
/* orientation_array, */ 
{
  int nx_final=FinalImage.xDim;
  int ny_final=FinalImage.yDim;
  int nx_test=ResampledTest.xDim;
  int ny_test=ResampledTest.yDim;
  int i, j;
  int ioffset;
  Pixel *finaltmp, *testtmp;
  int itest, jtest, ifinal, isample;
  int jrow, icol;
  if(img==2) {
    /* Initialize final image to zeroes on first call.*/
    for(ifinal=0; ifinal<nx_final*ny_final; ifinal++) {
      finaltmp=(FinalImage.pixels)+ifinal;
      finaltmp->R=0;
      finaltmp->G=0;
      finaltmp->B=0;
    }
  }
  /* Loop over blocks (tiles) of the smaller image in the larger */
  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      ioffset=i+j*nx_dim; /* Offset in index_array of an image sample */
      if(index_array[ioffset]==img) {
      /* Replace a block of the image with the resampled image 
	 if the corresponding index_array value matches img */
	for(jtest=0; jtest<ny_test; jtest++) {
	  for(itest=0; itest<nx_test; itest++) {
	    isample=itest+jtest*nx_test;
	    jrow=jtest+j*ny_test;
	    icol=itest+i*nx_test;
	    ifinal=icol+jrow*nx_final;
	    /* Replace pixel if in bounds in the output image */
	    if(ifinal<(nx_final*ny_final)) {
	      finaltmp=(FinalImage.pixels)+ifinal;
	      testtmp=(ResampledTest.pixels)+isample;
	      if(do_grayscale==0) {
		/* Color */
		finaltmp->R=testtmp->R;
		finaltmp->G=testtmp->G;
		finaltmp->B=testtmp->B;
	      }
	      else {
		/* Grayscale */
		finaltmp->R=(testtmp->R+testtmp->G+testtmp->B)/3;
		finaltmp->G=finaltmp->R;
		finaltmp->B=finaltmp->R;
	      }
	    }
	  }
	}
      }
    }
  }
  FinalImage.valid=1;
}

