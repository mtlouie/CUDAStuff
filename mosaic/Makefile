# Makefile for mosaic app and associated testers
CC = gcc
#CFLAGS = -Wall -Werror  -std=c99

ifndef NO_OPENMP
OPENMPFLAG = -fopenmp
else
OPENMPFLAG = 
endif

ifndef DEBUG
DEBUGFLAGS=
else
ifeq ($(DEBUG),full)
# Debug and profiling
DEBUGFLAGS= -g -pg
else
# Debug only
DEBUGFLAGS= -g
endif
endif

CFLAGS = -Wall -std=c99 -O3 $(OPENMPFLAG) $(DEBUGFLAGS)

#CC = nvcc

LIBS = -ljpeg -lpng -lz

OBJECTS = mosaic.o image.o CompareImage.o

all: mosaic

$(OBJECTS) : image.h CompareImage.h


mosaic: $(OBJECTS)
	$(CC) $(CFLAGS)  -o $@ $^ $(LIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) -o $@ $<

mosaic.o: main.c 
	$(CC) -c $(CFLAGS) -o $@ $<

mosaic_test: mosaic
	rm -f mosaicLena.jpg
	./mosaic lena.png mosaicLena.png

clean:
	rm -f mosaic *.o

very_clean: clean
	rm -rf Resampled* original*
