// File handlers for displayable images

#include "image.h"
#include <stdlib.h>    // For exit()
#include <stdio.h>
#include <string.h>    // For memcpy
#include <stdarg.h>    // for va_list and friends
#include "jpeglib.h"   // On Ubuntu, apt-get install libjpeg-dev
#include <png.h>       //   ... libpng-dev
#ifndef _SETJMP_H_
#include <setjmp.h>    // For libjpeg error handling
#endif

void abort_(const char * s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}
//
// The handling of PNG images is based on code at http://zarb.org/~gc/html/libpng.html
// The classic PNG text, "PNG: The Definitive Ghide" (the vole book), available online
// at http://www.libpng.org/pub/png/pngbook.html, uses a stale version of libpng

// Based on https://gist.github.com/niw/5963798, 2016/06/22
#define SIG_SIZE 8
ImageData ReadPNG(char* fileName)
{
    ImageData meta;
    meta.valid = false;

    // Step 0: Make sure we can open the input file
    FILE* inFile = fopen(fileName, "rb");
    if (!inFile) {
        abort_("ReadPNG: Can't open %s for reading\n", fileName);
    }

    // Step 1: initialize the reader
    png_structp png = png_create_read_struct(
            PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        abort_("ReadPNG: png_create_read_struct() failed\n");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        abort_("ReadPNG: png_create_info_struct() failed\n");
    }

    if (setjmp(png_jmpbuf(png))) {
        abort_("ReadPNG: Error during initialization.\n");
    }
    
    png_init_io(png, inFile);
    png_read_info(png, info);

    // Step 2: Collect image statistics
    int width           = png_get_image_width(png, info);
    int height          = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Make some adjustments
    if (bit_depth == 16) {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if(png_get_valid(png, info, PNG_INFO_tRNS))
            png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }
    
    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    // Step 3: Read the data
    if (setjmp(png_jmpbuf(png))) {
        abort_("ReadPNG: Error during image data reading.\n");
    }

    png_bytep row_pointers[height];

    // This method gives a discontiguous array of rows
    // Each row is later copied to the destination image, RGB only.
    // There should be a more streamlined method, but this method works for lena.png
    for (int y = 0; y < height; y++) {
       row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
       if (!row_pointers[y]) {
           abort_("ReadPNG: Couldn't allocate row %d\n", y);
        }
    }
    
    // Now we can read the image
    int row_bytes = png_get_rowbytes(png, info);
    png_read_image(png, row_pointers);

    // Convert the image from discontiguous rows to contiguous rows, discarding alpha
    Pixel* image = (Pixel*)malloc(height * width * 3);
    if (!image) {
        abort_("ReadPNG: Couldn't allocate image (%d, %d)\n", height, row_bytes);
    }

    // Brute force copy, skip over ALPHA channel
    for (int row = 0; row < height; row++) {
        for (int pixel = 0; pixel < width; pixel++)
        {
            image[(row * width + pixel)] = 
                *(Pixel*)(row_pointers[row] + pixel * 4);
        }
        // As we use them, delete the pointers to the image rows.
        free(row_pointers[row]);
    }

   
    meta.xDim = width;
    meta.yDim = height;
    meta.valid = true;
    meta.pixels = (Pixel*)image;

    // Step 4: Tidy up.
    fclose(inFile);

    // Done.
    return meta;

}

// See github.com/ellzey/libjpeg/blob/master/example.c for an example
//   of using libjpeg
ImageData ReadJPG(char* fileName)
{
  ImageData meta;
  struct jpeg_decompress_struct cinfo;  // Workspace for JPEG
  struct jpeg_error_mgr jerr;           // Error handler

  // Step 0: Make sure we can open the input file
  FILE* inFile = fopen(fileName, "rb");
  if (!inFile) {
    fprintf(stderr, "Can't open %s for reading\n", fileName);
    exit(1);
  }

  // Step 1: Allocate and initalize JPEG decompression object
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Step 2: Link the file
  jpeg_stdio_src(&cinfo, inFile);

  // Step 3: Get file header
  jpeg_read_header(&cinfo, TRUE);

  // Step 4: Set decompression parameters, if necessary
  // Not necessary

  // Step 5: Start decompression
  jpeg_start_decompress(&cinfo);

  meta.xDim = cinfo.output_width;
  meta.yDim = cinfo.output_height;

  int data_size = meta.xDim * meta.yDim * 3;
  meta.pixels = (Pixel *)malloc(data_size);
  
  // Step 6: Process the rows
  unsigned char* rowptr[1];
  while (cinfo.output_scanline < cinfo.output_height) {
    rowptr[0] = (unsigned char*)meta.pixels +
                   (3 * meta.xDim * cinfo.output_scanline);
    jpeg_read_scanlines(&cinfo, rowptr, 1);
  }

  // Step 7: Finish decompression
  jpeg_finish_decompress(&cinfo);

  // Step 8: Clean up
  jpeg_destroy_decompress(&cinfo);
  fclose(inFile);

  return meta;
}

// Also informed by https://gist.github.com/niw/5963798
bool WritePNG(ImageData* outImage, char* fileName)
{
    // Step 0: Make sure we can open the input file
    FILE* outFile = fopen(fileName, "wb");
    if (!outFile) {
        abort_("WritePNG: Can't open %s for writing\n", fileName);
    }

    int width = outImage->xDim;
    int height = outImage->yDim;

    // The image we get has RGB-only data, in 3-byte pixels
    // We need an array of rows, with pointers to them, each row having RGBA data
    png_bytep row_pointers[height];
    for (int row = 0; row < height; row++) {
        // Output image rows are RGBA
        row_pointers[row] = (png_bytep)malloc(width * 4);
        if (!row_pointers[row]) {
            abort_("WritePNG: Can't allocate memory for row %d\n", row);
        }
    
        // Set all to 0xff, for initializing alpha channel
        memset(row_pointers[row], 0xff, width * 4);

        // Now copy pixels from input image to output
        for (int pixel = 0; pixel < width; pixel++) {
            *(Pixel*)(row_pointers[row] + pixel * 4) =
                *(outImage->pixels + row * width + pixel);
        }
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, outFile);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
            png,
            info,
            width, height,
            8,
            PNG_COLOR_TYPE_RGBA,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
            );
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format, Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    for(int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    fclose(outFile);
    return false;
}

bool WriteJPG(ImageData* outImage, char* fileName)
{
  struct jpeg_compress_struct cinfo;  // Workspace for JPEG
  struct jpeg_error_mgr jerr;         // Error handler
  FILE* outFile;
  JSAMPROW rowPointer[1];
 
  // Step 1: Initialize error handler and compression object
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // Step 2: Link to output file
  outFile = fopen(fileName, "wb");
  if (!outFile) {
    fprintf(stderr, "Can't open %s for writing\n", fileName);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outFile);

  // Step 3: Set compression parameters
  cinfo.image_width = outImage->xDim;
  cinfo.image_height = outImage->yDim;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  // jpeg_set_quality(&cinfo, JPEG_QUALITY, TRUE);

  // Step 4: Start compressor
  jpeg_start_compress(&cinfo, TRUE);

  // Step 5: Process scanlines
  while (cinfo.next_scanline < cinfo.image_height) {
    rowPointer[0] = (JSAMPLE *)&outImage->pixels[cinfo.next_scanline * outImage->xDim];
     jpeg_write_scanlines(&cinfo, rowPointer, 1);
  }
  
  // Step 6: Finish compression
  jpeg_finish_compress(&cinfo);
  fclose(outFile);

  // Step 7: Release compression object
  jpeg_destroy_compress(&cinfo);

  return true;
}

ImageData ReadImage(char* fileName)
{
    // Check that fileName ends in either jpg or png, and dispatch if so
    if (strcmp("png", fileName + strlen(fileName) - 3) == 0)
        return ReadPNG(fileName);
    else if (strcmp("jpg", fileName + strlen(fileName) - 3) == 0)
        return ReadJPG(fileName);
    else {
        ImageData meta;
        meta.valid = false;
        return meta;
    }
}

bool WriteImage(ImageData* outImage, char* fileName)
{
    // Check that fileName ends in either jpg or png, and dispatch if so
    if (strcmp("png", fileName + strlen(fileName) - 3) == 0)
        return WritePNG(outImage, fileName);
    else if (strcmp("jpg", fileName + strlen(fileName) - 3) == 0)
        return WriteJPG(outImage, fileName);
    else {
        return false; 
    }
}

// Clone an image
ImageData CloneImage(ImageData source) {
    ImageData clone;
    clone.valid = false;
    clone.xDim = source.xDim;
    clone.yDim = source.yDim;
    size_t size = clone.xDim * clone.yDim * sizeof(Pixel);
    clone.pixels = (Pixel *)malloc(size);
    if (clone.pixels) {
        memcpy(clone.pixels, source.pixels, size);
        clone.valid = true;
    }
    return clone;
}

// Release an image's pixel data
void ReleaseImage(ImageData* image)
{
    if (image->valid) {
        free(image->pixels);
    }
    image->valid = false;
    image->pixels = 0;
}

// Make some useful solid-color block images
ImageData MakeColorRectangle(int w, int h, 
        unsigned char r, unsigned char g, unsigned char b)
{
    ImageData image;
    image.valid = false;
    image.xDim = w;
    image.yDim = h;
    size_t size = w * h * sizeof(Pixel);
    image.pixels = (Pixel *)malloc(size);
    if (image.pixels) {
        unsigned char* ptr = (unsigned char*) image.pixels;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                *ptr++ = r;
                *ptr++ = g;
                *ptr++ = b;
            }
        }
        image.valid = true;
    }
    return image;
}

ImageData MakeColorSquare(int side,
        unsigned char r, unsigned char g, unsigned char b)
{
    return MakeColorRectangle(side, side, r, g, b);
}

ImageData MakeMonoRectangle(int w, int h, unsigned char gray)
{
    return MakeColorRectangle(w, h, gray, gray, gray);
}

ImageData MakeMonoSquare(int side, unsigned char gray)
{
    return MakeColorRectangle(side, side, gray, gray, gray);
}
