# Makefile for mosaic app and associated testers

CC = gcc
#CC = nvcc

OPTS = -Wall -Werror  -std=c99

mosaic: mosaic.o image.o
	$(CC) mosaic.o image.o -o mosaic -ljpeg -lpng -lz

mosaic.o: main.c image.h
	$(CC) -c $(OPTS) main.c -o mosaic.o

image.o: image.h image.c
	$(CC) -c $(OPTS) image.c -o image.o

mosaic_test: mosaic
	rm -f mosaicLena.jpg
	./mosaic lena.png mosaicLena.png

clean:
	rm -f mosaic *.o
