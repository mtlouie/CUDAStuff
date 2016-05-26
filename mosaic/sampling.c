#include "sampling.h"
#include <stdlib.h>

//#define ANDY

ImageData Resample(ImageData testImage, size_t sampled_x, size_t sampled_y) {
#ifdef ANDY
    ImageData resampled;
    resampled.xDim = sampled_x;
    resampled.yDim = sampled_y;
    resampled.pixels = (Pixel *)malloc(sampled_x * sampled_y * sizeof(Pixel));

    // If we couldn't get memory, skip the rest of it
    if (!resampled.pixels)
    {
        resampled.valid = false;
        return resampled;
    }

    // Now do the hard work
    // "Cheat" to simplify editing
    int ny_test = testImage.yDim;

    // This geometry assumes the two axes to have unit length.  Should it be
    // a square image???
    // x and y are offsets to the upper left corner of the current target pixel
    double x;
    double y;

    Pixel* ti = testImage.pixels;

    for(int i = 0; i < sampled_x; i++) {
        x = (double)i / sampled_x;
        for(int j = 0; j < sampled_y; j++) {
            y = (double)j / sampled_y;
            // ??? ix and jy will always be zero ???
            int ix = x;
            int jy = y;
            double dx = x-ix;
            double dy = y-jy;
            int ixp1 = ix+1;
            int jyp1 = jy+1;

            resampled.pixels[j * sampled_x + i].R = 
                (unsigned char)(0.5 * ((1.0 - dx) * ti[ix * ny_test + jy].R
                     + dx * ti[ixp1 * ny_test + jy].R
                     + (1.0 - dy) * ti[ix * ny_test + jy].R
                     + dy * ti[ix * ny_test + jyp1].R));
            resampled.pixels[j * sampled_x + i].G = 
                (unsigned char)0.5d * ((1.e0 - dx) * ti[ix * ny_test + jy].G
                     + dx * ti[ixp1 * ny_test + jy].G
                     + (1.e0 - dy) * ti[ix * ny_test + jy].G
                     + dy * ti[ix * ny_test + jyp1].G);
            resampled.pixels[j * sampled_x + i].B = 
                (unsigned char)0.5d * ((1.e0 - dx) * ti[ix * ny_test + jy].B
                     + dx * ti[ixp1 * ny_test + jy].B
                     + (1.e0 - dy) * ti[ix * ny_test + jy].B
                     + dy * ti[ix * ny_test + jyp1].B);
        }
    }
#else // BILATERAL
    ImageData resampled;
// From biliniar.c
//
//inline float 
//BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y) 
//{
//	float x2x1, y2y1, x2x, y2y, yy1, xx1;
//	x2x1 = x2 - x1;
//	y2y1 = y2 - y1;
//	x2x = x2 - x;
//	y2y = y2 - y;
//	yy1 = y - y1; 
//	xx1 = x - x1;
//	return 1.0 / (x2x1 * y2y1) * (
//		q11 * x2x * y2y +
//		q21 * xx1 * y2y +
//		q12 * x2x * yy1 +
//		q22 * xx1 * yy1
//	);
//}

// Reference:
// en.wikipedia.org/wiki/Bilinear_Interpolation
// Overview of approach:
//   The source and target images are each viewed as three functions of 
//   two variables:
//      fr(x, y) is the Red channel mapping the center point of each
//         pixel to the range [0, 255]; similarly for fb(x,y) and fg(x,y).
//      To account for non-square source images, the larger of the 
//         two sides is taken as the unit of width.  Thus, the longest
//         side is 1.0 units.
//      The target function tf() is based on the nearest four neigbors
//         of the source data sf().  Note that these centers can be computed
//         from the target center:
//      TODO: Incorporate notes from hand-written derivation of the
//         algorithm.
//      DISCLAIMER: In the name of getting something posted, these comments
//         are incomplete.
//
//      for the source pixel at i, j, the centerpoint x, y is given by
//         x = pixel_size/2 + i*pixel_size = pixel_size(i + 0.5)
//         y = pixel_size/2 + j*pixel_size = pixel_size(j + 0.5)
//      Similarly for the target pixel centers, but with different
//         pixel_counts.
//

#endif
    return resampled;
}

