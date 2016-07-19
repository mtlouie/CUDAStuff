#include <stdio.h>
#include <stdlib.h>   // For malloc
#include "image.h"
#include "CompareImage.h"

#include <limits.h>  /* Defines LONG_MAX */

ImageData Resample(ImageData testImage, size_t sampled_x, size_t sampled_y);

int main (int argc, char **argv) {

    char buffer[256];
    int nx_src, ny_src;
    int nx_sample, ny_sample;
    int nx_dim, ny_dim;
    int *index_array;
    long *min_rms_array;
    int i;
    int nx_test, ny_test;
    int nx_tile, ny_tile;
    int nx_final, ny_final;
    /* Read source image into memory
     * Get filename from command line */
    nx_sample=ny_sample=8;
    nx_tile=ny_tile=24;
    if (argc>1)
    {	
	fprintf(stderr,"argc = %d\n", argc);
	ImageData srcImage = ReadImage(argv[1]);
	if (srcImage.valid)
	{
	    nx_src=srcImage.xDim;
	    ny_src=srcImage.yDim;
	    nx_dim=nx_src/nx_sample;
	    ny_dim=ny_src/ny_sample;

	    /* Allocate memory for Mosaic data
	     * index_array is array of indices into library of images
	     * orientation_array is array of orientations of library images
	     * (only for square test images) */
	    index_array=(int *)malloc(nx_dim*ny_dim*sizeof(int));
            for(int k=0; k<(nx_dim*ny_dim); k++) 
		index_array[k]=0;
		
        /* if(nx_test==ny_test) {
	orientation_array=(char *)malloc(nx_dim*ny_dim*sizeof(char));
	}*/
	    /* min_rms_array is metric for difference between test and original images
	       at each block of the original */
	    min_rms_array=(long *)malloc(nx_dim*ny_dim*sizeof(long));
            for(int k=0; k<(nx_dim*ny_dim); k++) 
		min_rms_array[k]=LONG_MAX;
	    /* test_image is buffer for library images to be compared against original */
	    /* Loop over library images */
	    for(i=2; i<argc; i++) {
		/*   Get library image */
		ImageData testImage=ReadImage(argv[i]);
		ImageData ResampledTest;
		nx_test=testImage.xDim;
		ny_test=testImage.yDim;
		/*   Resample library image to tile size */
		ResampledTest=Resample(testImage, nx_sample, ny_sample);
		ReleaseImage(&testImage);
		/*   Compare tile with source image by tiling over source image
		 *    index_array, orientation_array, comparison_array */
		CompareImage(srcImage, ResampledTest, i, index_array, min_rms_array /*, orientation_array */, nx_dim, ny_dim);
		ReleaseImage(&ResampledTest);
	    }
	
	    /* Free memory for source image. */
	    ReleaseImage(&srcImage);
	    /*
	     * Constructing the final image
	     */
	    nx_final=((nx_src+nx_sample-1)/nx_sample)*nx_tile;
	    ny_final=((ny_src+ny_sample-1)/ny_sample)*ny_tile;
	    ImageData FinalImage;
	    FinalImage.xDim=nx_final;
	    FinalImage.yDim=ny_final;
	    FinalImage.pixels=malloc(nx_final*ny_final*sizeof(Pixel));
	    /* Loop over library images */
	    for(i=2; i<argc; i++) {
		/* Insert logic to skip unused library images */
		ImageData testImage=ReadImage(argv[i]);
		ImageData ResampledTest;
#ifdef DEBUG
		sprintf(buffer,"original_tiles-%d.jpg", i); 
		WriteImage(&testImage, buffer); 
#endif
		ResampledTest=Resample(testImage, nx_tile, ny_tile);
		ReleaseImage(&testImage);
#ifdef DEBUG
		sprintf(buffer,"Resampled-%d.jpg", i);
		WriteImage(&ResampledTest, buffer); 
#endif
		ReplaceInImage(i, index_array, nx_dim, ny_dim, 
			       FinalImage, ResampledTest); /* orientation_array, */ 
		ReleaseImage(&ResampledTest);
	    }
	    /*   Loop through index_array, replacing locations in final image with library image if it matches index value. */
	    sprintf(buffer,"tiled-%s", argv[1]);
	    WriteImage(&FinalImage, buffer);
	    ReleaseImage(&FinalImage);
	}
    }
}

#ifdef RANDOMIMAGES
#include <stdlib.h>

void GetImage(Pixel* testImage, int nx_test, int ny_test, int i, int pixel_depth) {
    /* This version produces a random image */
    int j;
    float scale;
    Pixel *ptr=testImage;
    if(i==0) {
        srandom(12);
    }
    for(j=0; j<nx_test*ny_test; j++) {
        scale=(float)random()/(float)RAND_MAX;
        ptr->R=(unsigned int)(pixel_depth*scale);
        scale=(float)random()/(float)RAND_MAX;
        ptr->G=(unsigned int)(pixel_depth*scale);
        scale=(float)random()/(float)RAND_MAX;
        ptr->B=(unsigned int)(pixel_depth*scale);
        ptr++;
    }
}  

#endif /* RANDOMIMAGES */

ImageData Resample(ImageData testImage, size_t sampled_x, size_t sampled_y) {
    ImageData resampled;
    resampled.xDim = sampled_x;
    resampled.yDim = sampled_y;
    resampled.pixels = (Pixel *)malloc(sampled_x * sampled_y * sizeof(Pixel));

    if (!resampled.pixels)
    {
	/* If we couldn't get memory, skip the rest of the processing. */
        resampled.valid = false;
        return resampled;
    }

    /* Now do the hard work */
    int nx_test = testImage.xDim;
    int ny_test = testImage.yDim;

#ifdef DEBUG
    /* Insert red pixels for debugging */
    for(int i = 0; i < sampled_x; i++) {
        for(int j = 0; j < sampled_y; j++) {
	  resampled.pixels[i*sampled_y+j].R = 128;
	  resampled.pixels[i*sampled_y+j].G = 0;
	  resampled.pixels[i*sampled_y+j].B = 0;
	}
    }
#endif

    /* Note: The resampling scheme below will distort the resampled image if
     * the resampled image does not have the same aspect ratio as the original image.
     * For the general case, we will need to truncate or fill an image. */
    for(int i = 0; i < sampled_x; i++) {
        for(int j = 0; j < sampled_y; j++) {
	  double x = (double)i / (double)(sampled_x-1);
	  double y = (double)j / (double)(sampled_y-1);
	  int ix = x*(nx_test-1);
	  int jy = y*(ny_test-1);
	  double dx = x*(double)(nx_test-1)-(double)ix;
	  double dy = y*(double)(ny_test-1)-(double)jy;
	  Pixel *top_left, *top_right, *bottom_left, *bottom_right, *dest;
	  int xstride, ystride;
	  /* The next pair of if statements allow us to interpolate all the way to the edges of the image. */
	  xstride=((ix+1)<nx_test)?1:0;
	  ystride=((jy+1)<ny_test)?nx_test:0;
	  /* Locations of pixels to be interpolated in input image */
	  top_left=&testImage.pixels[jy*nx_test+ix];
	  top_right=top_left+xstride;
	  bottom_left=top_left+ystride; 
          bottom_right=top_left+xstride+ystride;
	  /* Location of destination pixel in output image */
	  dest=&resampled.pixels[j*sampled_x+i];
	  /* Standard resampling scheme - interpolate along two lines parallel to the x axis, then interpolate vertically. */
	  dest->R = 
	    (unsigned int)((1.0e0-dy)*((1.e0-dx)*top_left->R+dx*top_right->R)+dy*((1.e0-dx)*bottom_left->R+dx*bottom_right->R));
	  dest->G = 
	    (unsigned int)((1.0e0-dy)*((1.e0-dx)*top_left->G+dx*top_right->G)+dy*((1.e0-dx)*bottom_left->G+dx*bottom_right->G));
	  dest->B = 
	    (unsigned int)((1.0e0-dy)*((1.e0-dx)*top_left->B+dx*top_right->B)+dy*((1.e0-dx)*bottom_left->B+dx*bottom_right->B));
        }
    }

    return resampled;
}

/* Local Variables: */
/* mode:c           */
/* c-file-style: "stroustrup" */
/* End:             */
