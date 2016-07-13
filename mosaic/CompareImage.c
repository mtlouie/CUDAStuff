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
      sum=0L;
      for(jtile=0; jtile<ny_tile; jtile++) {
	for(itile=0; itile<nx_tile; itile++) {
#ifdef ROWMAJOR
	  /* Pixels are stored in row major order */
  	  itest=itile+nx_tile*jtile;
	  isrc=itile+i*nx_tile+(jtile+j*ny_tile)*nx_src;
#else
	  /* Pixels are stored in column major order */
	  itest=jtile+ny_tile*itile;
          isrc=jtile+j*ny_tile+(itile+i*nx_tile)*ny_src;
#endif
	  /* Compute sum if in bounds */
	  if(isrc<nx_src*ny_src) {
	    srctmp=&srcImage.pixels[isrc];
	    testtmp=&ResampledTest.pixels[itest];
	    r=srctmp->R-testtmp->R;
	    g=srctmp->G-testtmp->G;
	    b=srctmp->B-testtmp->B;
	    sum+=r*r+g*g+b*b;
	  }
	}
      }
      ioffset=j+i*ny_dim;
      //      printf("%ld %ld\n", sum, min_rms_array[ioffset]);
      if(sum<min_rms_array[ioffset]) {
	index_array[ioffset]=img;
	min_rms_array[ioffset]=sum;
      }
      //      printf("%ld %ld %d\n", sum, min_rms_array[ioffset], index_array[ioffset]);
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
  int itest, jtest, ifinal, isample;
  int ipixelin, ipixelout;
  if(img==2) {
    /* Initialize final image to zeroes on first call.*/
    for(ifinal=0; ifinal<nx_final*ny_final; ifinal++) {
      finaltmp=&FinalImage.pixels[ifinal];
      finaltmp->R=0;
      finaltmp->G=0;
      finaltmp->B=0;
    }
  }
  // printf("Tiling with image %d\n", img);
  /* Loop over blocks (tiles) of the smaller image in the larger */
  for(j=0; j<ny_dim; j++) {
    printf("\n");
    for(i=0; i<nx_dim; i++) {
      ioffset=j+i*ny_dim; /* Offset in index_array of an image sample */
      /* Replace a block of the image if the corresponding index_array value matches img */
      printf("%d ", index_array[ioffset]);
      if(index_array[ioffset]==img) {
	//        printf("Replacing tile at %d %d with image %d\n", i, j, img);
	// printf(".");
	//	printf("(i,j)=(%d,%d)\n",i,j);
	ipixelin=nx_test*ny_test;
	ipixelout=0;
	for(jtest=0; jtest<ny_test; jtest++) {
	  for(itest=0; itest<nx_test; itest++) {
#ifdef ROWMAJOR
	    isample=itest+nx_test*jtest;
	    ifinal=itest+i*nx_test+(jtest+j*ny_test)*nx_final;
#else
	    isample=jtest+ny_test*itest;
	    ifinal=jtest+j*ny_test+(itest+i*nx_test)*ny_final;
#endif
	    /* Replace pixel if in bounds in the output image */
	    if(ifinal<nx_final*ny_final) {
	      ipixelout++;
	      finaltmp=&FinalImage.pixels[ifinal];
	      testtmp=&ResampledTest.pixels[isample];
	      // printf("Replacing pixel at %d with pixel %d of image %d (%d, %d, %d)\n", ifinal, itest, img,
	      //		     (int)testtmp->R,(int)testtmp->G,(int)testtmp->B);
	      // printf("%d %d %d %d (%ld skipped)\n", itest, jtest, itest+i*nx_test, jtest+j*ny_test, skipcount);
	      finaltmp->R=testtmp->R;
	      finaltmp->G=testtmp->G;
	      finaltmp->B=testtmp->B;
	    }
	    else {
	      skipcount++;
	      //	      printf("Skipping itest=%d jtest=%d skipcount=%ld\n", itest, jtest, skipcount);
            }
	  }
	}
	// printf("pixels in: %d, pixels out: %d\n", ipixelin, ipixelout);
      }
    }
  }
  //  printf("\n");
  FinalImage.valid=1;
}

