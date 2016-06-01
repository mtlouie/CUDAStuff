// CUDA to start with...
// Operation:  B = parMatTranspose(A)
#include <stdio.h>
#include <unistd.h>

#define BLOCK_SIZE 32

__global__ void parMatTranspose (float *B, float *A, int ndim) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
 
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
    
  // Index into global array A.
  int i = blockDim.y*by+ty;
  int j = blockDim.x*bx+tx;
	
  // Working memory;
  __shared__ float block[BLOCK_SIZE][BLOCK_SIZE];
	
  if((i<ndim)&&(j<ndim)) {
    block[ty][tx] = A[j*ndim+i];
  }
  // Synchronize to make sure the matrices are loaded
  __syncthreads();
  if((i<ndim)&&(j<ndim)) {
    B[i*ndim+j]=block[ty][tx];
  }
  // Synchronize to make sure the matrices are loaded
  __syncthreads();
}



int main(int argc, char **argv) {
  int ndim;
  int threaddim=BLOCK_SIZE;
  int blockdim;
  float *h_A, *h_B;
  float *d_A, *d_B;
  int i,j,k;
  if(argc>1) {
    ndim=atoi(argv[1]);
  }
  else {
   ndim=1024;
  }

  blockdim=(ndim+BLOCK_SIZE-1)/BLOCK_SIZE;

while (1)
  {

  h_A=(float *)malloc(ndim*ndim*sizeof(float));
  h_B=(float *)malloc(ndim*ndim*sizeof(float));
  for(k=0; k<ndim*ndim; k++)
    h_A[k]=(float)k;

  cudaMalloc((void **)&d_A,ndim*ndim*sizeof(float));
  printf("%lX\n", (unsigned long)d_A);
  
  cudaMalloc((void **)&d_B,ndim*ndim*sizeof(float));

  cudaMemcpy(d_A,h_A, ndim*ndim*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(blockdim,blockdim);
  dim3 threads(threaddim,threaddim);	
  parMatTranspose <<<grid, threads>>>(d_B, d_A, ndim);
  cudaMemcpy(h_B,d_B, ndim*ndim*sizeof(float), cudaMemcpyDeviceToHost);

  int count=0;
  for(i=0; i<ndim; i++) {
    for(j=0; j<ndim; j++) {
      if(h_B[ndim*j+i]!=(float)(ndim*i+j)) count++;
    }
  }

  printf("Count=%d\n", count);
  if(ndim<10) {
    for(i=0; i<ndim*ndim; i++)
      printf("a[%d]=%g b[%d]=%g\n", i, h_A[i], i, h_B[i]);
  }
  sleep (1);
  //cudaFree(d_A);
  //cudaFree(d_B);

  free(h_A);
  free(h_B);
  }
}

/* Major mode settings for GNU Emacs */
/* This coerces the editor to treat this as C code. */

/* Local Variables: */
/* mode:c           */
/* End:             */
