/* (c) 2016 Livio and Tiziano Mengotti */
/* n-dipoles simulation */
// to make it: nvcc n-dipoles.cu -o n-dipoles.out
// to execute the example ./n-dipoles.out
#include <stdio.h>
#define N 640 // number of dipoles
//TODO: adjust these constants
#define t 0.0001 // timestamp
#define r 10e-15 // rutheford radius
#define q 1e-10 //electric quant

__global__ void simulate_dipoles(double *x, double *y, double *omega, double *E_pot) {
	double ax[N*2]; double ay[N*2]; 
        	
	int tid = blockIdx.x; 
	//if (tid<N)
	//	c[tid] = a[tid] + b[tid];
}

/*
__device__ double getElectricForce(int p, int q) {
	/*	
	double E_force=0;	
	if ((p<N) && (q<N)) {
		E_force=1;
	}
	return E_force;
	*/
//	return 0;
//}

/*
__device__ double getDistance(int p, int q, double *x, double *y) {
	double distance = 0;
	if ((p<N) && (q<N)) {
		distance = sqrt((x[p]-x[q])*(x[p]-x[q])+(y[p]-y[q])*(y[p]-y[q]));		
	}
	return distance;
}
*/

int main(void) {
	double x[N*2]; double y[N*2]; 
	double omega[N];
        double E_pot[N]; // potential energy of the system

        double *dev_x, *dev_y, *dev_omega;
	double *dev_E_pot;

	cudaMalloc( (void**)&dev_x,     N*2*sizeof(double));
        cudaMalloc( (void**)&dev_y,     N*2*sizeof(double));
        cudaMalloc( (void**)&dev_omega, N*sizeof(int));
        cudaMalloc( (void**)&dev_E_pot, N*sizeof(int));

	//TODO: init variables with Box-Muller
	for (int i=0; i<N*2; i++) {
		x[i] = -i,
		y[i] = i*i;
		omega[i] = 0;
		E_pot[i] = 0;
	}

        cudaMemcpy(dev_x, x, N*2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, N*2*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_omega, omega, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_E_pot, E_pot, N*sizeof(double), cudaMemcpyHostToDevice);
	
	simulate_dipoles<<<N,1>>>(dev_x, dev_y, dev_omega, dev_E_pot);

	cudaMemcpy(omega, dev_omega, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(E_pot, dev_E_pot, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<N; i++) {
		printf("i: %d omega: %g E_pot %g\n", i, omega[i], E_pot[i]);
	}
	
	cudaFree(dev_x);
        cudaFree(dev_y);
	cudaFree(dev_omega);
	cudaFree(dev_E_pot);

}




