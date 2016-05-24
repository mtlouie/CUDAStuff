CFLAGS=
CFLAGS_OMP= $(CFLAGS)
TARGETS=cudatest cudazctest openmptest
NVCC=nvcc
CC=gcc
all: $(TARGETS)
cudatest: cudatest.cu
	$(NVCC) $(CFLAGS) -o cudatest cudatest.cu -lm
cudazctest: cudatest.cu
	$(NVCC) $(CFLAGS) -DZEROCOPY -o cudazctest cudatest.cu -lm

openmptest: openmptest.c
#	$(CC) $(CFLAGS_OMP) -fopenmp -o $@ $^
	$(CC) $(CFLAGS_OMP) -o $@ $^
clean: 
	-rm -f $(TARGETS) *.o
veryclean: 
	-rm -f $(TARGETS) *.o *~ *.log

