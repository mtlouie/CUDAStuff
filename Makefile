# FLAGS to pass to the top level compiler

NVCC=nvcc
CC=gcc

# Common flags for both compilers are set using CFLAGS
CFLAGS=
# 
GCCFLAGS=$(CFLAGS)
NVCCFLAGS=$(CFLAGS)

# Flags to enable profiling
GCC_PROFILE= -g -pg 
#CUDA_PROFILE= -G  -Xcompiler ' -v -g -pg -fno-omit-frame-pointer -marm -funwind-tables ' 
CUDA_PROFILE= -G  -Xcompiler ' -v -g -pg -fno-omit-frame-pointer -funwind-tables ' 

# XCFLAGS are extra flags to pass to the external compiler, gcc, under nvcc
XCFLAGS=

ifdef do_profile
GCCFLAGS += $(GCC_PROFILE)
NVCCFLAGS += $(CUDA_PROFILE)
endif

CUDA_TARGETS= cudatest cudazctest CUDA1 cudatest-omp cudazctest-omp
CPU_TARGETS= openmptest serialtest


all: cuda_targets cpu_targets

cuda_targets: $(CUDA_TARGETS)
cpu_targets: $(CPU_TARGETS)

# Default rules for simple C and CUDA codes
%: %.c
	$(CC) $(GCCFLAGS) -o $@ $^ -lm
%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lm

cudazctest: cudatest.cu
	$(NVCC) $(NVCCFLAGS) -DZEROCOPY -o $@ $^ -lm

cudatest-omp: cudatest.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler " -fopenmp $(XCFLAGS) "  -o $@ $^ -lm

cudazctest-omp: cudatest.cu
	$(NVCC) $(NVCCFLAGS)  -DZEROCOPY -Xcompiler " -fopenmp $(XCFLAGS) " -o $@ $^ -lm

serialtest: openmptest.c
	$(CC) $(GCCFLAGS) -o $@ $^

openmptest: openmptest.c
	$(CC) $(GCCFLAGS) -fopenmp -o $@ $^

clean:
	-rm -f $(CPU_TARGETS) $(CUDA_TARGETS) *.o

veryclean: 
	-rm -f $(CPU_TARGETS) $(CUDA_TARGETS) *.o *~ *.log

