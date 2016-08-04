#include <stdio.h>
#include <strings.h>  /* For rindex */
#include <stdlib.h>   /* For malloc and atoi */
#include <unistd.h>   /* For getopt */
#include <getopt.h>
#include "image.h"
#include "CompareImage.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* Default tile dimensions used in comparisons */
#ifndef SAMPLESIZE
#define SAMPLESIZE 8
#endif
/* Default tile dimensions used in replacements */
#ifndef TILESIZE
#define TILESIZE  16
#endif

#include <limits.h>  /* Defines LONG_MAX */

ImageData Resample(ImageData testImage, size_t sampled_x, size_t sampled_y);

int main (int argc, char **argv) {

    char buffer[256];
    char *btmp;
    int nx_src, ny_src;
    int nx_sample, ny_sample;
    int nx_dim, ny_dim;
    int *index_array;
    long *min_rms_array;
    int i;
    int image_label;
    int iorientation;
    int k;
    int nx_test, ny_test;
    int nx_tile, ny_tile;
    int nx_final, ny_final;
    int flag;
    int do_grayscale=0;
    int enable_verbose=0;
    int do_counts=0;
    int *counts;
    int max_orientations=1;
#ifdef _OPENMP
    int ithreads;
#endif
    extern char *optarg;
    extern int optind, opterr, optopt;

    nx_sample=ny_sample=SAMPLESIZE;
    nx_tile=ny_tile=TILESIZE;

    /* parse optional command line arguments:
	 -c              compute counts of images used
         -g              enables grayscale processing
	 -h              print help and exit
         -s VALUE        sets nx_sample=ny_sample=VALUE
         -t VALUE        sets nx_tile=ny_tile=VALUE
	 -v              enable verbose output
    */

    while((flag=getopt(argc, argv, "cghrs:t:v"))!=-1) {
	switch(flag) {
	case 'c':
	    printf("Computing counts of images used\n");
	    do_counts=1;
	    break;
	case 'g':
	    printf("Enabling grayscale image processing\n");
	    do_grayscale=1;
	    break;
	case 'h':
	    printf("Usage: %s [-c] [-g] [-h] [-s VALUE] [-t VALUE] [-v] target_image list_of_tile_images\n", argv[0]);
	    printf("FLAGS\n");
	    printf("\t-c              compute counts of images used\n");
	    printf("\t-g              enables grayscale processing\n");
	    printf("\t-h              print this help and exit\n");
	    printf("\t-r              include rotated images in mosaic\n");
	    printf("\t-s VALUE        sets nx_sample=ny_sample=VALUE\n");
	    printf("\t-t VALUE        sets nx_tile=ny_tile=VALUE\n");
	    printf("\t-v              enable verbose output\n");
	    exit(0);
	    break;
	case 'r':
	    max_orientations=4;
	    break;
	case 's':
	    nx_sample=ny_sample=atoi(optarg);
	    printf("Setting nx_sample=ny_sample=%d\n", nx_sample);
	    break;
	case 't':
	    nx_tile=ny_tile=atoi(optarg);
	    printf("Setting nx_tile=ny_tile=%d\n", nx_tile);
	    break;
	case 'v':
	    enable_verbose=1;	   
	    break;
	}
    }

#ifdef _OPENMP
	// If OpenMP, set up number of threads
    ithreads=omp_get_max_threads();
    printf("OpenMP enabled with a maximum of %d threads\n",ithreads);
#endif

    /* Read source image into memory
     * Get filename from command line after parsing command line args */
    if ((argc-optind)>1) {	

	if(enable_verbose==1) {
	    fprintf(stderr,"Files to process = %d\n", argc-optind+1);
	    fprintf(stderr,"Target = %s\n", argv[optind]);
	}

	ImageData srcImage = ReadImage(argv[optind]);
	if (srcImage.valid==1) {
	    nx_src=srcImage.xDim;
	    ny_src=srcImage.yDim;
	    nx_dim=nx_src/nx_sample;
	    ny_dim=ny_src/ny_sample;

	    /* Allocate memory for Mosaic data
	     * index_array is array of indices into library of images */

	    index_array=(int *)malloc(nx_dim*ny_dim*sizeof(int));
            for(k=0; k<(nx_dim*ny_dim); k++) 
		index_array[k]=0;
		
	    /* min_rms_array is metric for difference between test and original images
	       at each block of the original */
	    min_rms_array=(long *)malloc(nx_dim*ny_dim*sizeof(long));
            for(k=0; k<(nx_dim*ny_dim); k++) 
		min_rms_array[k]=LONG_MAX;

	    printf("Scanning tiles over source image");

	    /* test_image is buffer for library images to be compared against original */
	    /* Loop over library images to compare against target  */
	    for(i=optind+1; i<argc; i++) {

		if(enable_verbose==1) {
		    printf("*");
		    if((i-optind)%50==0) 
			printf("\n");
		}

		/* Open library image */
		ImageData testImage=ReadImage(argv[i]);
		ImageData ResampledTest;
		nx_test=testImage.xDim;
		ny_test=testImage.yDim;
		/* Resample library image to tile size */
		ResampledTest=Resample(testImage, nx_sample, ny_sample);
		ReleaseImage(&testImage);
		for(iorientation=0; iorientation<max_orientations; iorientation++) {
		    image_label=max_orientations*i+iorientation;
		    /* Compare tile with source image by tiling over source image index_array,  comparison_array */
		    CompareImage(srcImage, ResampledTest, image_label, index_array, min_rms_array, nx_dim, ny_dim, do_grayscale);
		    if(max_orientations>1)
			RotateImageCCW(&ResampledTest);
		}
		ReleaseImage(&ResampledTest);
	    }
	
	    /* Free memory for source image. */
	    ReleaseImage(&srcImage);

	    printf("Constructing final image");

	    if(enable_verbose==1)
		printf("\n");

	    /*
	     * Constructing the final image
	     */

	    nx_final=((nx_src+nx_sample-1)/nx_sample)*nx_tile;
	    ny_final=((ny_src+ny_sample-1)/ny_sample)*ny_tile;
	    ImageData FinalImage;
	    FinalImage.xDim=nx_final;
	    FinalImage.yDim=ny_final;
	    FinalImage.pixels=malloc(nx_final*ny_final*sizeof(Pixel));
	    /* Initialize image with zeros */
	    for(i=0; i<nx_final*ny_final; i++) {
		FinalImage.pixels[i].R=0;
		FinalImage.pixels[i].G=0;
		FinalImage.pixels[i].B=0;
	    }

	    /* Loop over library images to build final image */
	    for(i=optind+1; i<argc; i++) {

		if(enable_verbose==1) {
		    printf("-");
		    if((i-optind)%50==0) 
			printf("\n");
		}

		/* Insert logic to skip unused library images */
		ImageData testImage=ReadImage(argv[i]);
		ImageData ResampledTest;
#ifdef DEBUG
		sprintf(buffer,"original_tiles-%d.jpg", i); 
		WriteImage(&testImage, buffer); 
#endif
		ResampledTest=Resample(testImage, nx_tile, ny_tile);
		ReleaseImage(&testImage);
		for(iorientation=0; iorientation<max_orientations; iorientation++) {
		    image_label=max_orientations*i+iorientation;
#ifdef DEBUG
		    sprintf(buffer,"Resampled-%d.jpg", image_label);
		    WriteImage(&ResampledTest, buffer); 
#endif
		    ReplaceInImage(image_label, index_array, nx_dim, ny_dim, FinalImage, ResampledTest, do_grayscale);
		    if(max_orientations>1)
			RotateImageCCW(&ResampledTest);
		}
		ReleaseImage(&ResampledTest);
	    }

	    if(enable_verbose==1)
		printf("\n");

	    if((btmp=rindex(argv[optind],'/'))==(char *)NULL)
		sprintf(buffer,"tiled-%s", argv[optind]);
	    else 
		sprintf(buffer,"tiled-%s", btmp+1);

	    printf("Printing final image on %s\n", buffer);
	    WriteImage(&FinalImage, buffer);

	    ReleaseImage(&FinalImage);
	    free(min_rms_array);
	    if(do_counts>0) {
		counts=(int *)malloc(max_orientations*argc*sizeof(int));
		for(k=0;k<max_orientations*argc;k++)
		    counts[k]=0;
		for(k=0;k<nx_dim*ny_dim;k++) 
		    counts[index_array[k]]++;
		printf("Image #\tCount\tOrientation\tImage\n");
		for(k=optind+1;k<argc;k++) {
		    for(iorientation=0; iorientation<max_orientations; iorientation++)
			printf("%d\t%d\t%d\t%s\n",
			       max_orientations*k+iorientation,
			       counts[max_orientations*k+iorientation], 
			       iorientation, 
			       argv[k]);
		}
		free(counts);
	    }
	    free(index_array);
	}
	else {
	    printf("%s is not valid\n", argv[optind]);
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
