#ifndef IMAGE_H
#define IMAGE_H

#include <stdbool.h>   // For 'bool', 'true', 'false'

typedef struct _Pixel 
{ 
  unsigned char R;
  unsigned char G;
  unsigned char B;
} Pixel;

typedef struct _ImageData {
  // Pointer to allocated image data
  int     valid;
  Pixel * pixels;
  int     xDim;
  int     yDim;
} ImageData;

// Read an image from the specified filename, returning a struct containing
//   useful data about the image.  If there is any error, valid = 0.
ImageData ReadImage(char* fileName);

// Write an image to the specified filename, returning true (1) on success,
//   false (0) on failure.
bool WriteImage(ImageData* outImage, char* fileName);

// Clone an image
ImageData CloneImage(ImageData source);

// Release an image's pixel data
void ReleaseImage(ImageData* image);

// Make some useful solid-color block images
ImageData MakeColorRectangle(int w, int h, 
        unsigned char r, unsigned char g, unsigned char b);
ImageData MakeColorSquare(int side,
        unsigned char r, unsigned char g, unsigned char b);
ImageData MakeMonoRectangle(int w, int h, unsigned char gray);
ImageData MakeMonoSquare(int side, unsigned char gray);

#endif // IMAGE_H
