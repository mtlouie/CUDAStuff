# Makefile for mosaic app and associated testers

CC = gcc
#CC = nvcc

OPTS = -Wall -Werror -std=c99

mosaic: mosaic.o image.o
	$(CC) mosaic.o image.o -o mosaic -ljpeg

mosaic.o: main.c image.h
	$(CC) -c $(OPTS) main.c -o mosaic.o

image.o: image.h image.c
	$(CC) -c $(OPTS) image.c -o image.o

mosaic_test: mosaic
	rm -f mosaicAlan.jpg
	./mosaic AlanTuring.jpg mosaicAlan.jpg

clean:
	rm -f mosaic *.o
