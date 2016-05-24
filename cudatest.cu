#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Build with -DDEBUG to see memory available on GPU at start of calculation. */
/* Build with -DZEROCOPY to take advantage of memory shared between CPU and GPU. */

#define NUMTHREADS 1024.0
#define INUMTHREADS 1024

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


/* Nanosecond resolution timer */
double nanosecond_timer(void) {
  double rresult;
  struct timespec curr_value;
  (void)clock_gettime(CLOCK_REALTIME, &curr_value);
  rresult=((double)curr_value.tv_sec + 1.0e-9*(double)curr_value.tv_nsec);
  return( rresult  );
}

/* CUDA device kernel function */
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
  /*char string1[]="cudaMalloc of d_A";
  char string2[]="cudaMalloc of d_B";
  char string3[]="cudaMalloc of d_C";
  char string4[]="cudaMemcpy from h_A to d_A";
  char string5[]="cudaMemcpy from h_B to d_B";
  char string9[]="cudaMemcpy of d_C";
  char string10[]="cudaFree of d_A";
  char string11[]="cudaFree of d_B";
  char string12[]="cudaFree of d_C";*/

  /* Allocate memory on GPU device */
  status=cudaMalloc((void **) &d_A, size);
  //report_error(string1, status);
  status=cudaMalloc((void **) &d_B, size);
  //report_error(string2, status);
  status=cudaMalloc((void **) &d_C, size);
  //report_error(string3, status);

  /* Copy arrays to GPU device */
  status=cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //report_error(string4, status);
  status=cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  //report_error(string5, status);

  /* Execute kernel on GPU */
  vecAddKernel <<< ceil(n/NUMTHREADS),INUMTHREADS >>>(d_A, d_B, d_C, n);

  /* Copy result back to host */
  status=cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  //report_error(string9, status);

  /* Free memory on GPU */
  status=cudaFree(d_A);
  //report_error(string10, status);
  status=cudaFree(d_B);
  //report_error(string11, status);
  status=cudaFree(d_C);
  //report_error(string12, status);
}

#else

/* Zero Copy Driver routine for GPU calls. */
/* The GPU and CPU share RAM, so there is no need to do data copies or allocate extra memory. */
void vecAdd (float *h_A, float *h_B, float *h_C, size_t n) {
  cudaError_t status;
  float *d_A, *d_B, *d_C;
  /*char string6[]="cudaHostGetDevicePointer of d_A";
  char string7[]="cudaHostGetDevicePointer of d_B";
  char string8[]="cudaHostGetDevicePointer of d_C";*/

  /* Map CPU memory locations to GPU memory locations */
  status=cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
  //report_error(string6, status);
  status=cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
  //report_error(string7, status);
  status=cudaHostGetDevicePointer((void **)&d_C, (void *)h_C, 0);
  //report_error(string8, status);

  /* Execute kernel on GPU */
  vecAddKernel <<< ceil(n/NUMTHREADS),INUMTHREADS >>>(d_A, d_B, d_C, n);
}

#endif
  
int main(int argc, char **argv) {
  float *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_D=NULL;
  size_t n;
  int j;
  int maxitG;
  int maxitC;
#ifdef _OPENMP
  int maxthreads;
#endif

  size_t i;
  size_t size;
  double start1, start2, end1, end2;
  float myfolly;
  cudaError_t status;

#ifdef DEBUG
  FILE *fp;
  size_t freemem, totalmem;
#endif

#ifdef DEBUG
  char string1[]="cudaMemGetInfo";
#endif


  char string2[]="cudaSetDeviceFlags";
  char string3[]="cudaHostAlloc of h_A";
  char string4[]="cudaHostAlloc of h_B";
  char string5[]="cudaHostAlloc of h_C";
  char string6[]="cudaFreeHost of h_A";
  char string7[]="cudaFreeHost of h_B";
  char string8[]="cudaFreeHost of h_C";


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

  /* Allocate host memory using standard C calls for CPU side test*/
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  /* Initialize host memory with random data. */
  srandom(0);
  for(i=0; i<n; i++) {
#ifndef NORANDOM
    h_A[i] = (float)n*(float)random()/(float)RAND_MAX;
    h_B[i] = (float)n*(float)random()/(float)RAND_MAX;
#else
    h_A[i] = (float)i;
    h_B[i] = (float)i;
#endif
  }
  /* Memory for CPU side test */
  h_D = (float *)malloc(size);

  /* Compute on host and time */
  start1 = nanosecond_timer(); 

  for (j = 0; j<maxitC; j++) {
#pragma omp parallel for 
    for (i=0; i<n; i++) {
      h_D[i] = h_A[i] + h_B[i];
    }
  }

  end1 = nanosecond_timer(); 	

  free(h_A);
  free(h_B);

  /* On the Tegra TK1, the CPUs and GPU share memory,
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
	
  /* Initialize host memory with random data. */
  srandom(0);
  for(i=0; i<n; i++) {
#ifndef NORANDOM
    h_A[i] = (float)n*(float)random()/(float)RAND_MAX;
    h_B[i] = (float)n*(float)random()/(float)RAND_MAX;
#else
    h_A[i] = (float)i;
    h_B[i] = (float)i;
#endif
  }
  /* Compute on GPU and time */
  start2 = nanosecond_timer();

  for (j = 0; j<maxitG; j++) {
    vecAdd(h_A, h_B, h_C, n);
  }

  end2 = nanosecond_timer();
	
  /* Report timing */
  printf("Time on CPU=%g (sec)\n", (end1-start1)/(double)maxitC);
  printf("CPU Performance = %g MFLOPS\n", (double)n*(double)maxitC/(end1-start1)/1.0e6);
  printf("Time on GPU=%g (sec)\n", (end2-start2)/(double)maxitG);
  printf("GPU Performance = %g MFLOPS\n", (double)n*(double)maxitG/(end2-start2)/1.0e6);
  printf("Speedup = %g\n", ((end1-start1)*(double)maxitG/(end2-start2)/(double)maxitC));

  /*  Check GPU result against CPU. */
  myfolly = 0.0;
  for (i=0; i<n; i++) {
    myfolly += fabsf((h_C[i]-h_D[i]));
  }
  printf("Agreement check, myfolly=%f\n", myfolly);

#ifdef DEBUG
  /* Detailed check of 1st and last elements of each array */
  printf("h_A[%lu]=%g\n", (unsigned long)0, h_A[0]);
  printf("h_B[%lu]=%g\n", (unsigned long)0, h_B[0]);
  printf("h_C[%lu]=%g\n", (unsigned long)0, h_C[0]);
  printf("h_D[%lu]=%g\n", (unsigned long)0, h_D[0]);
  printf("h_A[%lu]=%g\n", (unsigned long)(n-1), h_A[n-1]);
  printf("h_B[%lu]=%g\n", (unsigned long)(n-1), h_B[n-1]);
  printf("h_C[%lu]=%g\n", (unsigned long)(n-1), h_C[n-1]);
  printf("h_D[%lu]=%g\n", (unsigned long)(n-1), h_D[n-1]);
  fp=fopen("cudatest.log","w");
  for (i=0; i<n; i++) {
    fprintf(fp,"%u %g %g %g %g\n", i, h_A[i], h_B[i], h_C[i], h_D[i]);
  }
  fclose(fp);
#endif

  /* Free host memory allocated with cudaHostAlloc() */
  status=cudaFreeHost(h_A);
  report_error(string6, status);
  status=cudaFreeHost(h_B);
  report_error(string7, status);
  status=cudaFreeHost(h_C);
  report_error(string8, status);
  /* Free host memory allocated with malloc() */
  free(h_D);
}

/* Major mode settings for GNU Emacs */
/* This coerces the editor to treat this as C code. */

/* Local Variables: */
/* mode:c           */
/* End:             */

