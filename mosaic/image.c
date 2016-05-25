// File handlers for displayable images

#include "image.h"
#include <stdlib.h>    // For exit()
#include <stdio.h>
#include <setjmp.h>    // For libjpeg error handling
#include <string.h>    // For memcpy
#include "jpeglib.h"   // On Ubuntu, apt-get install libjpeg-dev

// See github.com/ellzey/libjpeg/blob/master/example.c for an example
//   of using libjpeg
ImageData ReadImage(char* fileName)
{
  ImageData meta;
  struct jpeg_decompress_struct cinfo;  // Workspace for JPEG
  struct jpeg_error_mgr jerr;           // Error handler
  FILE* inFile;

  // Step 0: Make sure we can open the input file
  inFile = fopen(fileName, "rb");
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

bool WriteImage(ImageData* outImage, char* fileName)
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
    free(image->pixels);
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
