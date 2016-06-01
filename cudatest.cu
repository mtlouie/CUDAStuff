#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* NVCC: Build with -Xcompiler -fopenmp to enable OpenMP directives. */
/* Build with -DDEBUG to see memory available on GPU at start of calculation. */
/* Build with -DZEROCOPY to take advantage of memory shared between CPU and GPU. */

#define NUMTHREADS 1024.0
#define INUMTHREADS 1024

/* A single routine to read a CUDA error status variable and print a message to the screeen */

void report_error(char *string, cudaError_t status) {
  switch(status) {
  case cudaSuccess:
#ifdef DEBUG
    printf("%s returned cudaSuccess\n", string); 
#endif
    break;
  case cudaErrorInvalidHostPointer:
    printf("%s returned cudaErrorInvalidHostPointer\n", string);
    exit(1);
    break;
  case cudaErrorInvalidDevicePointer:
    printf("%s returned cudaErrorInvalidDevicePointer\n", string);
    exit(1);
    break;
  case cudaErrorMemoryAllocation:
    printf("%s returned cudaErrorMemoryAllocation\n", string);
    exit(1);
    break;
  case cudaErrorInvalidValue:
    printf("%s returned cudaErrorInvalidValue\n", string);
    exit(1);
    break;
  }
}

/* Nanosecond resolution timer, need to test what the actual granularity of the clock is. */
double nanosecond_timer(void) {
  double rresult;
  struct timespec curr_value;
  (void)clock_gettime(CLOCK_REALTIME, &curr_value);
  rresult=((double)curr_value.tv_sec + 1.0e-9*(double)curr_value.tv_nsec);
  return( rresult  );
}

/* CUDA device kernel function to add vectors A and B and store the result in C. */
__global__ void vecAddKernel(float *A, float *B, float *C, size_t n) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i<n) C[i] = A[i] + B[i];
}

#ifndef ZEROCOPY
/* Standard driver routine for GPU calls */
void vecAdd (float *h_A, float *h_B, float *h_C, size_t n) {
  cudaError_t status;
  size_t size = n*sizeof(float);
  float *d_A, *d_B, *d_C;
#ifdef VERBOSE
  char string1[]="cudaMalloc of d_A";
  char string2[]="cudaMalloc of d_B";
  char string3[]="cudaMalloc of d_C";
  char string4[]="cudaMemcpy from h_A to d_A";
  char string5[]="cudaMemcpy from h_B to d_B";
  char string9[]="cudaMemcpy of d_C";
  char string10[]="cudaFree of d_A";
  char string11[]="cudaFree of d_B";
  char string12[]="cudaFree of d_C";
#endif

  /* Allocate memory on GPU device */
  status=cudaMalloc((void **) &d_A, size);
#ifdef VERBOSE
  report_error(string1, status);
#endif
  status=cudaMalloc((void **) &d_B, size);
#ifdef VERBOSE
  report_error(string2, status);
#endif
  status=cudaMalloc((void **) &d_C, size);
#ifdef VERBOSE
  report_error(string3, status);
#endif

  /* Copy arrays to GPU device */
  status=cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
#ifdef VERBOSE
  report_error(string4, status);
#endif
  status=cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
#ifdef VERBOSE
  report_error(string5, status);
#endif
  /* Execute kernel on GPU */
  vecAddKernel <<< ceil(n/NUMTHREADS),INUMTHREADS >>>(d_A, d_B, d_C, n);

  /* Copy result back to host */
  status=cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
  report_error(string9, status);
#endif
  /* Free memory on GPU */
  status=cudaFree(d_A);
#ifdef VERBOSE
  report_error(string10, status);
#endif
  status=cudaFree(d_B);
#ifdef VERBOSE
  report_error(string11, status);
#endif
  status=cudaFree(d_C);
#ifdef VERBOSE
  report_error(string12, status);
#endif
}

#else

/* Zero Copy/Pinned memory driver routine for GPU calls. 
   The GPU and CPU share RAM, so there is no need to do data copies or allocate extra memory. 
   Just need to map device pointers to CPU pointers. */

void vecAdd (float *h_A, float *h_B, float *h_C, size_t n) {
  float *d_A, *d_B, *d_C;
  cudaError_t status;
#ifdef VERBOSE
  char string6[]="cudaHostGetDevicePointer of d_A";
  char string7[]="cudaHostGetDevicePointer of d_B";
  char string8[]="cudaHostGetDevicePointer of d_C"; 
#endif

  /* Map CPU memory locations to GPU memory locations */
  status=cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
#ifdef VERBOSE
  report_error(string6, status);
#endif
  status=cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
#ifdef VERBOSE
  report_error(string7, status);
#endif
  status=cudaHostGetDevicePointer((void **)&d_C, (void *)h_C, 0);
#ifdef VERBOSE
  report_error(string8, status);
#endif

  /* Execute kernel on GPU */
  vecAddKernel <<< ceil(n/NUMTHREADS),INUMTHREADS >>>(d_A, d_B, d_C, n);

}

#endif
  
int main(int argc, char **argv) {
  float *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_D=NULL;
  size_t n;
  int j;
  int maxitG, maxitC;
#ifdef _OPENMP
  int maxthreads;
#endif
  size_t i;
  size_t size;
  double startC, startG, endC, endG;
  double startP, endP, timeP;
  double timeC, timeG;
  float result_test;
#if (defined(DEBUG)||defined(ZEROCOPY))
  cudaError_t status;
#endif

#ifdef DEBUG
  FILE *fp;
  size_t freemem, totalmem;
#endif

#ifdef DEBUG
  char string1[]="cudaMemGetInfo";
#endif

#ifdef ZEROCOPY
  char string2[]="cudaSetDeviceFlags";
  char string3[]="cudaHostAlloc of h_A";
  char string4[]="cudaHostAlloc of h_B";
  char string5[]="cudaHostAlloc of h_C";
  char string6[]="cudaFreeHost of h_A";
  char string7[]="cudaFreeHost of h_B";
  char string8[]="cudaFreeHost of h_C";
#endif
  startP=nanosecond_timer(); 
  /* Read size of array in elements from command line */
  if(argc>3) {
    n = atoi(argv[1]);
    maxitG = atoi(argv[2]);
    maxitC = atoi(argv[3]);
  }
  else {
    printf("Usage: %s ndim_array max_iterations_GPU max_iterations_CPU\n", argv[0]);
    exit(1);
  }

#ifdef 	_OPENMP
  maxthreads = omp_get_max_threads ();
  printf("OpenMP functionality enabled with %d threads\n", maxthreads);
#endif

  /* Compute size in bytes of a single array */
  size=n*sizeof(float);

#ifdef DEBUG
  status=cudaMemGetInfo(&freemem, &totalmem);
  report_error(string1, status);
  printf("GPU memory at program start free=%lu total=%lu\n", (unsigned long)freemem, (unsigned long)totalmem);
#ifndef ZEROCOPY
  printf("GPU memory required=%lu\n", (unsigned long)3*size);
  printf("Total RAM required=%lu\n", (unsigned long)7*size);
  if (freemem<7*size) {
    printf("Insufficient memory to run calculation\n");
    exit(1);
  }
#else
  printf("Total Shared RAM required=%lu\n", (unsigned long)4*size);
  if (freemem<4*size) {
    printf("Insufficient memory to run calculation\n");
    exit(1);
  }
#endif
#endif

  /* GPU processing block */
#ifndef ZEROCOPY
  /* Host and Device memory are distinct */
  /* Allocate host memory using standard C calls*/
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);
#else
  /* Host and Device memory reside in a shared block of memory.
     On the Tegra TK1, the CPUs and GPU share memory,
     so we can save memory and skip several large memory copy
     operations. */
  status=cudaSetDeviceFlags(cudaDeviceMapHost);
  report_error(string2, status);
  status=cudaHostAlloc((void **)&h_A, size, cudaHostAllocMapped);
  report_error(string3, status);
  status=cudaHostAlloc((void **)&h_B, size, cudaHostAllocMapped);
  report_error(string4, status);
  status=cudaHostAlloc((void **)&h_C, size, cudaHostAllocMapped);
  report_error(string5, status);
#endif

  /* Initialize host memory vectors with random data or with index value. */
#ifndef NORANDOM
  srandom(0); /* Initialize the random number generator. */
  for(i=0; i<n; i++) {
    h_A[i] = (float)n*(float)random()/(float)RAND_MAX;
    h_B[i] = (float)n*(float)random()/(float)RAND_MAX;
  }
#else
  for(i=0; i<n; i++) {
    h_A[i] = (float)i;
    h_B[i] = (float)i;
  }
#endif

  /* Compute on GPU and time */
  startG = nanosecond_timer(); 
  for (j = 0; j<maxitG; j++) {
    vecAdd(h_A, h_B, h_C, n);   
  }
  endG = nanosecond_timer();

  /* CPU processing block. This repeats the above calculation on the CPU side,
     with or without OpenMP parallelization. */

#ifdef ZEROCOPY

  /* We must allocate memory for CPU side calculation using malloc() rather than cudaHostAlloc().
     This is because for some mysterious and as yet undetermined reason the CPU side 
     calculation runs ~5-6x slower when the memory is allocated with cudaHostAlloc(), regardless of
     optimization level. This has been tested on CUDA 6.5 (Jetson TK1) and CUDA 7.0 (Jetson TX1). */

  /* Free host memory allocated with cudaHostAlloc() */
  status=cudaFreeHost(h_A);
  report_error(string6, status);
  status=cudaFreeHost(h_B);
  report_error(string7, status);

  /* Allocate memory with malloc() */
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
	
  /* Re-initialize the vectors with the same values as before. */
#ifndef NORANDOM
  srandom(0);
  for(i=0; i<n; i++) {
    h_A[i] = (float)n*(float)random()/(float)RAND_MAX;
    h_B[i] = (float)n*(float)random()/(float)RAND_MAX;
  }
#else
  for(i=0; i<n; i++) {
    h_A[i] = (float)i;
    h_B[i] = (float)i;
  }
#endif
#endif

  h_D = (float *)malloc(size);

  /* Compute on host and time */
  startC = nanosecond_timer(); 
  for (j = 0; j<maxitC; j++) {
#pragma omp parallel for 
    for (i=0; i<n; i++) {
      h_D[i] = h_A[i] + h_B[i];
    }
  }
  endC = nanosecond_timer(); 	
	
  /* Report timing */
  timeC=(endC-startC)/(double)maxitC;
  timeG=(endG-startG)/(double)maxitG;

  printf("Time on CPU=%g (sec) aggregate time=%g (sec)\n", timeC, (endC-startC));
  printf("CPU Performance = %g MFLOPS\n", (double)n/1.0e6/timeC);
  printf("Time on GPU=%g (sec) aggregate time=%g (sec)\n", timeG, (endG-startG));
  printf("GPU Performance = %g MFLOPS\n", (double)n/1.0e6/timeG);
  printf("Speedup = %g\n", timeC/timeG);

  /* Check GPU results against CPU results. */
  result_test = 0.0;
  for (i=0; i<n; i++) {
    result_test += fabsf((h_C[i]-h_D[i]));
  }
  printf("Agreement check, result_test=%f\n", result_test);

#ifdef DEBUG
  /* Output first and last elements of each array on stdout. */
  printf("h_A[%lu]=%g h_B[%lu]=%g h_C[%lu]=%g\n h_D[%lu]=%g", 
	 (unsigned long)0, h_A[0],
	 (unsigned long)0, h_B[0],
	 (unsigned long)0, h_C[0],
	 (unsigned long)0, h_D[0]);
  printf("h_A[%lu]=%g h_B[%lu]=%g h_C[%lu]=%g h_D[%lu]=%g\n", 
	 (unsigned long)(n-1), h_A[n-1],
	 (unsigned long)(n-1), h_B[n-1],
	 (unsigned long)(n-1), h_C[n-1],
	 (unsigned long)(n-1), h_D[n-1]);
#ifdef VERBOSE
  fp=fopen("cudatest.log","w");
  for (i=0; i<n; i++) {
    fprintf(fp,"%u %g %g %g %g\n", i, h_A[i], h_B[i], h_C[i], h_D[i]);
  } 
  fclose(fp);
#endif
#endif

  /* Free memory and exit */
  free(h_A);
  free(h_B);
  free(h_D);
#ifndef ZEROCOPY
  free(h_C);
#else
  status=cudaFreeHost(h_C);
  report_error(string8, status);
#endif
  endP=nanosecond_timer(); 
  timeP= endP-startP;
  printf ("Total time = %lf seconds. \n ", timeP);
}

/* Major mode settings for GNU Emacs */
/* This coerces the editor to treat this as C code. */

/* Local Variables: */
/* mode:c           */
/* End:             */

