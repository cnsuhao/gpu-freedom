/* (c) 2016 Livio and Tiziano Mengotti */
/* n-dipoles simulation */
// to make it: nvcc n-dipoles.cu -o n-dipoles.out
// to execute the example ./n-dipoles.out
#include <stdio.h>
#define N 640 // number of dipoles
//TODO: adjust these constants
#define T 0.0001 // timestamp
#define R 10e-15 // rutheford radius
#define Q 1 //electric quant
#define PI 3.1415926535
#define COULOMB 1/(4*PI)
#define Mp 1  // proton mass
#define Me 1  // electron mass

__device__ double sqr(double x) {
	return x*x;
}

__device__ double getDistanceSquared(int p1, int p2, double *x, double *y) {
	double distance = 0;
	if ((p1<N) && (p2<N)) {
		distance = sqr(x[p1]-x[p2])+sqr((y[p1]-y[p2]));		
	}
	return distance;
}


// Projects a magnitude along x and y axis
__device__ void projectVector(double magnitude,
                              double x1, double x2, double y1, double y2, 
                              double *px, double* py) {

	double distance;
	double deltax; double deltay;
	
	deltax = (x2-x1);
	deltay = (y2-y1);
	distance = sqrt(sqr(deltax) + sqr(deltay));	

	// TODO: double check if projection formula is correct
	*px = magnitude * deltax / distance; 
	*py = magnitude * deltay / distance;
	
}  

__device__ void getElectricAcceleration(int p1, int p2, 
                                        double *x, double *y, double *ax, double *ay) {
	
	// acceleration is calculated on p1	
	double acceleration=0;
	double mass;
	if ((p1 % 2) == 0) // even indexes are proton
		mass = Mp;
	else    mass = Me; // odd indexes are electrons
		

	if ((p1<N) && (p2<N)) {
		acceleration = COULOMB * Q * Q * getDistanceSquared(p1, p2, x, y) / mass;  
		// now we need to project acceleration along (x1-x2) and (y1-y2)
		projectVector(acceleration, x[p1], x[p2], y[p1], y[p2], ax, ay);
	}

}


__global__ void simulate_dipoles(double *x, double *y, double *omega, double *E_pot) {
	double ax[N*2]; double ay[N*2]; 
        	
	int tid = blockIdx.x; 
        int iselectron;
	double ax_temp; double ay_temp;

	   if (tid<N) {
		// 1. calculate acceleration on tid particle
		ax[tid]=0; ay[tid]=0;

		for (int i=0; i<N; i++) {
			if (i==tid) continue; // we do not calculate on ourself
			iselectron = (i % 2);
			// we do not calculate acceleration on the partner particle on the dipole
			if ((iselectron==1) && (i==tid-1)) continue;
			if ((iselectron==0) && (i==tid+1)) continue;	
			
			getElectricAcceleration(tid, i, x, y, &ax_temp, &ay_temp);
			ax[tid]=ax[tid]+ax_temp;
			ay[tid]=ay[tid]+ay_temp;
		}
	   }
}




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




