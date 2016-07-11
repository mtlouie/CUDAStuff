#include <stdio.h>
#include <stdlib.h>   // For malloc
#include "image.h"

ImageData Resample(ImageData testImage, size_t sampled_x, size_t sampled_y);

int main (int argc, char **argv) {

    /* Read source image into memory
     * Get filename from command line */
    ImageData srcImage = ReadImage(argv[1]);
    if (srcImage.valid)
    {
        //ImageData finalImage = CloneImage(srcImage);
        ImageData sampledFinal = Resample(srcImage, 256, 256);

#if 0
	nx_dim=nx_src/nx_sample;
	ny_dim=ny_src/ny_sample;

	/* Allocate memory for Mosaic data
	 * index_array is array of indices into library of images
	 * orientation_array is array of orientations of library images
	 * (only for square test images) */
	index_array=(int *)malloc(nx_dim*ny_dim*sizeof(int));
	if(nx_test==ny_test) {
	    orientation_array=(char *)malloc(nx_dim*ny_dim*sizeof(char));
	}
	/* min_rms_array is metric for difference between test and original images
	   at each block of the original */
	min_rms_array=(long *)malloc(nx_dim*ny_dim*sizeof(long));
	/* test_image is buffer for library images to be compared against original */
	test_image=(Pixel *) malloc(nx_test*ny_test*sizeof(Pixel));

	/* Loop over library images */
	for(i=0; i< nlib_images; i++) {
	    /*   Get library image */
	    GetImage(testImage, nx_test, ny_test, i);
	    /*   Resample library image to tile size */
	    Resample(testImage, nx_test, ny_test, ResampledTest, nx_sample, ny_sample);
	    /*   Compare tile with source image by tiling over source image
	     *    index_array, orientation_array, comparison_array */
	    CompareImage(SRCrgbArray, nx_src, ny_src, 
			 ResampledTest, nx_sample, ny_sample, 
			 index_array, min_rms_array, orientation_array);
	}
	
	/* Free memory for source image. */
	
	free(SRCrgbArray);
	/*
	 * Constructing the final image
	 *
	 * Allocate memory for final image
	 */
	nx_tile=nx_test*nx_scale;
	ny_tile=ny_test*ny_scale;
	nx_final=(nx_src/nx_sample)*nx_tile;
	ny_final=(ny_src/ny_sample)*ny_tile;
	FinalImage=malloc(nx_final*ny_final*n_color_depth*npixel_depth);
	
	/* Loop over library images */
	for(i=0; i< nlib_images; i++) {
	    /* Insert logic to skip unused library images */
	    GetImage(testImage, nx_test, ny_test, i);
	    Resample(testImage, nx_test, ny_test, ResampledTest, nx_tile, ny_tile);
	    ReplaceInImage(i, 
			   index_array,  orientation_array, nx_dim, ny_dim, 
			   FinalImage, nx_final, ny_final, 
			   ResampledTest, nx_tile, ny_tile);
	}
#endif /* #if 0 */

	/*   Loop through index_array, replacing locations in final image with library image if it matches index value. */
	WriteImage(&sampledFinal, argv[2]);
	/* ReleaseImage(&finalImage); */
	ReleaseImage(&srcImage);
    }
}

#if 0
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

#endif /* #if 0 */

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

    /* Now do the hard work
     * "Cheat" to simplify editing */
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
	  double dx = x*(nx_test-1)-ix;
	  double dy = y*(ny_test-1)-jy;
	  Pixel *top_left, *top_right, *bottom_left, *bottom_right, *dest;
	  int xstride, ystride;
	  /* The next pair of if statements allow us to interpolate all the way to the edges of the image. */
	  xstride=((ix+1)<nx_test)?ny_test:0;
	  ystride=((jy+1)<ny_test)?1:0;
	  /* Locations of pixels to be interpolated in input image */
	  top_left=&testImage.pixels[ix*ny_test+jy];
	  top_right=top_left+xstride;
	  bottom_left=top_left+ystride; 
          bottom_right=top_left+xstride+ystride;
	  /* Location of destination pixel in output image */
	  dest=&resampled.pixels[i*sampled_y+j];
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
