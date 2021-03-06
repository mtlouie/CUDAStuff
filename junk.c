#include <stdio.h>
#include <stdlib.h>

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


/* Nanosecond resolution timer */
double nanosecond_timer(void) {
  double rresult;
  struct timespec curr_value;
  (void)clock_gettime(CLOCK_REALTIME, &curr_value);
  rresult=((double)curr_value.tv_sec + 1.0e-9*(double)curr_value.tv_nsec);
  return( rresult  );
}




  
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


#ifdef DEBUG
  FILE *fp;
  size_t freemem, totalmem;
#endif



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
  printf("openmp functionality enabled with %d threads\n", maxthreads);
#endif

  /* Compute size in bytes of a single array */
  size=n*sizeof(float);




  /* Allocate host memory using standard C calls*/
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  /* Memory for CPU side test */
  h_D = (float *)malloc(size);

  /* Initialize host memory with random data. */
  for(i=0; i<n; i++) {
#ifndef NORANDOM
    h_A[i] = (float)n*(float)random()/(float)RAND_MAX;
    h_B[i] = (float)n*(float)random()/(float)RAND_MAX;
#else
    h_A[i] = (float)i;
    h_B[i] = (float)i;
#endif
  }

  /* Compute on host and time */
  start1 = nanosecond_timer(); 
  for (j = 0; j<maxitC; j++) {
#pragma omp for 
	  for(i=0; i<n; i++) {
		h_D[i] = h_A[i] + h_B[i];
		}
  }

  start2 = end1 = nanosecond_timer(); 

  end2 = nanosecond_timer();
	
	
  /* Report timing */
  printf("Time on CPU=%g (sec)\n", (end1-start1)/maxitC);
 
  sleep(1);
  



  /* Free host memory allocated with malloc() */
  free(h_A);
  free(h_B);
  free(h_C);

  /* Free host memory allocated with malloc() */
  free(h_D);
}

/* Major mode settings for GNU Emacs */
/* This coerces the editor to treat this as C code. */

/* Local Variables: */
/* mode:c           */
/* End:             */

