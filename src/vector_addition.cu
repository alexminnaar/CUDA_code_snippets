#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 10

__global__
void add(int *a, int *b,int *c){
	//get unique thread id
	int tid = blockIdx.x;

	if (tid < N){
		c[tid] = a[tid]+b[tid];
	}
}


int main()
{

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	//allocate memory on device for the three arrays
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMalloc((void**)&dev_b,N*sizeof(int));
	cudaMalloc((void**)&dev_c,N*sizeof(int));

	//fill arrays on host
	for(int i = 0 ; i<N;i++){
		a[i]= -i;
		b[i]= i*i;
	}


	//copy arrays to GPU
	cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c,c,N*sizeof(int),cudaMemcpyHostToDevice);


	//call cuda kernel
	add<<<N,1>>>(dev_a,dev_b,dev_c);

	//copy c array back to host
	cudaMemcpy(c,dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);


	//print results
	for(int i=0; i<N;i++){
		printf("%d + %d = %d\n",a[i],b[i],c[i]);
	}

	//free allocated memory on device
	cudaFree(dev_a);
	cudaFree(dev_a);
	cudaFree(dev_a);

	printf("done");
	return EXIT_SUCCESS;
}
