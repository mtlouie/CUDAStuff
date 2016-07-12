#include "image.h"
#include <stdio.h>

void CompareImage(ImageData srcImage, ImageData ResampledTest, int img, int *index_array, long *min_rms_array, /* int * orientation_array, */
		  int nx_dim, int ny_dim)
{
  long sum;
  long r, g, b;
  Pixel *srctmp, *testtmp;
  int isrc, itest;
  int nx_src=srcImage.xDim;
  int ny_src=srcImage.yDim;
  int nx_tile=ResampledTest.xDim;
  int ny_tile=ResampledTest.yDim;
  int ioffset;
  int i, j, itile, jtile;
  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      sum=0;
      for(jtile=0; jtile<ny_tile; jtile++) {
	for(itile=0; itile<nx_tile; itile++) {
	  itest=itile+nx_tile*jtile;
          isrc=itile+i*nx_tile+(jtile+j*ny_tile)*nx_src;
	  if(isrc<nx_src*ny_src) {
	    srctmp=&srcImage.pixels[isrc];
	    testtmp=&ResampledTest.pixels[itest];
	    r=srctmp->R-testtmp->R;
	    g=srctmp->G-testtmp->G;
	    b=srctmp->B-testtmp->B;
	    sum=r*r+g*g+b*b;
	  }
	}
      }
      ioffset=i+j*nx_dim;
      printf("%ld %ld\n", sum, min_rms_array[ioffset]);
      if(sum<min_rms_array[ioffset]) {
	index_array[ioffset]=img;
	min_rms_array[ioffset]=sum;
      }
      printf("%ld %ld %d\n", sum, min_rms_array[ioffset], index_array[ioffset]);
    }
  }
}


void ReplaceInImage(int img, int *index_array,  /* orientation_array, */ 
		    int nx_dim, int ny_dim, 
		    ImageData FinalImage, ImageData ResampledTest) {
  int nx_final=FinalImage.xDim;
  int ny_final=FinalImage.yDim;
  int nx_test=ResampledTest.xDim;
  int ny_test=ResampledTest.yDim;
  int i, j;
  int ioffset;
  long skipcount=0L;
  Pixel *finaltmp, *testtmp;
  int itest, jtest, ifinal;
  /* Initialize final image to zeroes */
  for(ifinal=0; ifinal<nx_final*ny_final; ifinal++) {
    finaltmp=&FinalImage.pixels[ifinal];
    finaltmp->R=128;
    finaltmp->G=0;
    finaltmp->B=0;
  }
  printf("Tiling with image %d\n", img);
  /* Loop over blocks (tiles) of the smaller image in the larger */
  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      ioffset=i+j*nx_dim; /* Offset in index_array of an image sample */
      /* Replace a block of the image if the corresponding index_array value matches img */
      if(index_array[ioffset]==img) {
        printf("Replacing tile at %d %d with image %d\n", i, j, img);
	for(jtest=0; jtest<ny_test; jtest++) {
	  for(itest=0; itest<nx_test; itest++) {
	    itest=itest+nx_test*jtest;
	    ifinal=itest+i*nx_test+(jtest+j*ny_test)*nx_final;
	    if(ifinal<nx_final*ny_final) {
	      finaltmp=&FinalImage.pixels[ifinal];
	      testtmp=&ResampledTest.pixels[itest];
	      printf("Replacing pixel at %d with pixel %d of image %d (%d, %d, %d)\n", ifinal, itest, img,
		     (int)testtmp->R,(int)testtmp->G,(int)testtmp->B);
	      finaltmp->R=testtmp->R;
	      finaltmp->G=testtmp->G;
	      finaltmp->B=testtmp->B;
	    }
	    else {
	      skipcount++;
	      printf("Skipping itest=%d jtest=%d skipcount=%ld\n", itest, jtest, skipcount);
            }
	  }
	}
      }
    }
  }
  FinalImage.valid=1;
}

