/* Examples from book "CUDA by example" by Jason Sanders */
// to make the example: nvcc helloworld.cu -o hello.out
// to execute the example ./hello.out
#include <stdio.h>

#define N	50000

__global__ void kernel(void) {
}

__global__ void addScalar(int a, int b, int *c) {
	*c = a + b;
}

__global__ void addVector(int *a, int *b, int *c) {
	int tid = blockIdx.x; 
	if (tid<N)
		c[tid] = a[tid] + b[tid];
}

int main_1 (void) {
	kernel<<<1,1>>>();
        printf("Hello, World!\n");
        return 0;
}

int main_2 (void) {
	int c;
        int *dev_c;
        cudaMalloc( (void**)&dev_c, sizeof(int) );
        
	addScalar<<<1,1>>>(2, 7, dev_c);
	
	cudaMemcpy (&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("2+7=%d\n",c);
	cudaFree(dev_c);

	return 0;
}

int main(void) {
	int a[N]; int b[N]; int c[N];
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc( (void**)&dev_a, N*sizeof(int));
	cudaMalloc( (void**)&dev_b, N*sizeof(int));
	cudaMalloc( (void**)&dev_c, N*sizeof(int));

	for (int i=0; i<N; i++) {
		a[i] = -i,
		b[i] = i*i;
	}

        // copy the arrays a and b to the GPU
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	addVector<<<N,1>>>(dev_a, dev_b, dev_c);

	// copy the array c back from the GPU to the CPU
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}
