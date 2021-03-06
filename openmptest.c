#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Build with -DDEBUG to see memory available on GPU at start of calculation. */

/* Nanosecond resolution timer */
double nanosecond_timer(void) {
  double rresult;
  struct timespec curr_value;
  (void)clock_gettime(CLOCK_REALTIME, &curr_value);
  rresult=((double)curr_value.tv_sec + 1.0e-9*(double)curr_value.tv_nsec);
  return( rresult  );
}




  
int main(int argc, char **argv) {
  float *h_A=NULL, *h_B=NULL, *h_C=NULL;
  size_t n;
  int j;
  int maxitC;
#ifdef _OPENMP
  int maxthreads;
  int nthreads;
#endif
  size_t i;
  size_t size;
  double startC, endC;
  double timeC;
  float myfolly;


#ifdef DEBUG
  FILE *fp;
  size_t freemem, totalmem;
#endif

  /* Read size of array in elements from command line */
  if(argc>2) {
    n = atoi(argv[1]);
    maxitC = atoi(argv[2]);
    /*    maxitG = atoi(argv[2]); */
  }
  else {
    printf("Usage: %s ndim_array max_iterations_CPU\n", argv[0]);
    exit(1);
  }

#ifdef 	_OPENMP
  maxthreads = omp_get_max_threads ();
  printf("OpenMP functionality enabled with a maximum of %d threads\n", maxthreads);
#endif

  /* Compute size in bytes of a single array */
  size=n*sizeof(float);

  /* Allocate host memory using standard C calls*/
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);

  /* Memory for CPU side test */
  h_C = (float *)malloc(size);

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

#ifdef _OPENMP
  for(nthreads=1; nthreads<=maxthreads; nthreads++) {
    omp_set_num_threads(nthreads);
#endif
    /* Compute on host and time */
    startC = nanosecond_timer(); 
    for (j=0; j<maxitC; j++) {
#pragma omp parallel for private(i)
      for(i=0; i<n; i++) {
#ifdef VERBOSE
	printf("%lu: Thread %d of %d\n", i, omp_get_thread_num(), omp_get_num_threads());
#endif
	h_C[i] = h_A[i] + h_B[i];
      }
    }
    endC = nanosecond_timer(); 
	
#ifdef 	_OPENMP
    printf("Results for %d threads\n", nthreads);
#endif
	
    /* Report timing */
    timeC=(endC-startC)/(double)maxitC;
    printf("Time on CPU=%g (sec)\n", timeC);
    printf("Performance = %g MFLOPS\n", (double)n/1.0e6/timeC);
#ifdef _OPENMP
  }
#endif

 /* Free host memory allocated with malloc() */
 free(h_A);
 free(h_B);
 free(h_C);
}

/* Major mode settings for GNU Emacs */
/* This coerces the editor to treat this as C code. */

/* Local Variables: */
/* mode:c           */
/* End:             */

