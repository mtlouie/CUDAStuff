# Makefile for mosaic app and associated testers

CC = gcc
#CC = nvcc

#OPTS = -Wall -Werror  -std=c99
OPTS = -Wall  -std=c99 -O3

mosaic: mosaic.o image.o CompareImage.o
	$(CC) mosaic.o image.o CompareImage.o -o mosaic -ljpeg -lpng -lz

mosaic.o: main.c image.h CompareImage.h
	$(CC) -c $(OPTS) main.c -o mosaic.o

image.o: image.h image.c
	$(CC) -c $(OPTS) image.c -o image.o

CompareImage.o: CompareImage.h CompareImage.c
	$(CC) -c $(OPTS) CompareImage.c -o CompareImage.o

mosaic_test: mosaic
	rm -f mosaicLena.jpg
	./mosaic lena.png mosaicLena.png

clean:
	rm -f mosaic *.o

very_clean: clean
	rm -rf Resampled* original*
