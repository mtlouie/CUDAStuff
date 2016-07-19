#include "image.h"
#include <stdio.h>

#define GRAYSCALE

void CompareImage(ImageData srcImage, ImageData ResampledTest, int img, int *index_array, long *min_rms_array, int nx_dim, int ny_dim)
/* int * orientation_array, */
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
  int icol, jrow;
  int i, j, itile, jtile;
  long matches=0L;
  //  printf("Comparing with image %d\n", img);
  /* if(img==2) */
  /*   printf("i,j,itile,jtile,nx_tile,ny_tile,itest,icol,jrow,isrc\n"); */

  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      ioffset=i+j*nx_dim;
      sum=4L;
      for(jtile=0; jtile<ny_tile; jtile++) {
	for(itile=0; itile<nx_tile; itile++) {
	  /* Pixels are stored in row major order */
  	  itest=itile+jtile*nx_tile;
	  jrow=jtile+j*ny_tile;
	  icol=itile+i*nx_tile;
	  isrc=icol+jrow*nx_src;
	  /* if(img==2) */
	  /* printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", */
	  /* 	 i,j,itile,jtile,nx_tile,ny_tile,itest,icol,jrow,isrc); */
	  /* Compute sum if in bounds */
	  if(isrc<(nx_src*ny_src)) {
	    srctmp=(srcImage.pixels)+isrc;
	    testtmp=(ResampledTest.pixels)+itest;
	    /* printf("src=%lx\tsrctmp=%lx\ttest=%lx\ttesttmp=%lx\n",(unsigned long)srcImage.pixels, */
	    /* 	   (unsigned long)srctmp, */
	    /* 	   (unsigned long)ResampledTest.pixels,  */
	    /* 	   (unsigned long)testtmp); */
	    r=(srctmp->R)-(testtmp->R);
	    g=(srctmp->G)-(testtmp->G);
	    b=(srctmp->B)-(testtmp->B);
#ifndef GRAYSCALE
	    sum+=(r*r+g*g+b*b);
#else
	    sum+=(r+g+b)*(r+g+b);
#endif
	  }
	}
      }
      /* printf("%d,%d,%d,%ld,%ld,%d,%d,", img, i, j, sum, */
      /* 	     min_rms_array[ioffset],index_array[ioffset], ioffset); */
      if(sum<min_rms_array[ioffset]) {
	matches++;
	index_array[ioffset]=img;
	min_rms_array[ioffset]=sum;
      }
      /* printf("%ld,%d\n", */
      /* 	     min_rms_array[ioffset],index_array[ioffset]); */
#ifdef PRINT_MIN_RMS
      printf("%ld ", min_rms_array[ioffset]);
#endif
      //      printf("%ld %ld %d\n", sum, min_rms_array[ioffset], index_array[ioffset]);
    }
    //    printf("\n");
  }
  //  printf("Done comparing with image %d, %ld matches found.\n", img, matches);
}


void ReplaceInImage(int img, int *index_array,  
		    int nx_dim, int ny_dim, 
		    ImageData FinalImage, ImageData ResampledTest) 
/* orientation_array, */ 
{
  int nx_final=FinalImage.xDim;
  int ny_final=FinalImage.yDim;
  int nx_test=ResampledTest.xDim;
  int ny_test=ResampledTest.yDim;
  int i, j;
  int ioffset;
  long skipcount=0L;
  Pixel *finaltmp, *testtmp;
  int itest, jtest, ifinal, isample;
  int jrow, icol;
  int ipixelin, ipixelout;
  if(img==2) {
    /* Initialize final image to zeroes on first call.*/
    for(ifinal=0; ifinal<nx_final*ny_final; ifinal++) {
      finaltmp=(FinalImage.pixels)+ifinal;
      finaltmp->R=0;
      finaltmp->G=0;
      finaltmp->B=0;
    }
  }
  // printf("Tiling with image %d\n", img);
  /* Loop over blocks (tiles) of the smaller image in the larger */
    //    printf("\n");
  for(j=0; j<ny_dim; j++) {
    for(i=0; i<nx_dim; i++) {
      ioffset=i+j*nx_dim; /* Offset in index_array of an image sample */
      /* Replace a block of the image if the corresponding index_array value matches img */
      //      printf("%d ", index_array[ioffset]);
      if(index_array[ioffset]==img) {
	//        printf("Replacing tile at %d %d with image %d\n", i, j, img);
	// printf(".");
	//	printf("(i,j)=(%d,%d)\n",i,j);
	ipixelin=nx_test*ny_test;
	ipixelout=0;
	for(jtest=0; jtest<ny_test; jtest++) {
	  for(itest=0; itest<nx_test; itest++) {
	    isample=itest+jtest*nx_test;
	    jrow=jtest+j*ny_test;
	    icol=itest+i*nx_test;
	    ifinal=icol+jrow*nx_final;
	    /* Replace pixel if in bounds in the output image */
	    if(ifinal<(nx_final*ny_final)) {
	      ipixelout++;
	      finaltmp=(FinalImage.pixels)+ifinal;
	      testtmp=(ResampledTest.pixels)+isample;
	      // printf("Replacing pixel at %d with pixel %d of image %d (%d, %d, %d)\n", ifinal, itest, img,
	      //		     (int)testtmp->R,(int)testtmp->G,(int)testtmp->B);
	      // printf("%d %d %d %d (%ld skipped)\n", itest, jtest, itest+i*nx_test, jtest+j*ny_test, skipcount);
#ifndef GRAYSCALE
	      finaltmp->R=testtmp->R;
	      finaltmp->G=testtmp->G;
	      finaltmp->B=testtmp->B;
#else
	      finaltmp->R=(testtmp->R+testtmp->G+testtmp->B)/3;
	      finaltmp->G=finaltmp->R;
	      finaltmp->B=finaltmp->R;
#endif
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

