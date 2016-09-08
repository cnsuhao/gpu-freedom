/* 
   n-dipoles simulation on CUDA architecture 
   (c) 2016 Livio and Tiziano Mengotti 

   to compile it: nvcc n-dipoles.cu -o n-dipoles.out
   to execute the example ./n-dipoles.out

   arrangement of arrays is
   index      0 1 2 3 4 5 ...  n-2 n-1
   particle   p e p e p e      p   e      with p proton and e electron
   dipole nb  0 0 1 1 2 2      n-1 n-1    this is the dipole number

*/
#include <stdio.h>
#define Np 1280 // number of particles (dipoles are half of them), should be CUDA core count and even
#define Nd 640 // number of dipoles, has to be Np/2
//TODO: adjust these constants
#define TIMESTEP   1E-15 // timestep in seconds
#define R          5.291772106712E−11  // Bohr radius in meter
#define Q          1.6021773349E-19 //elementar charge in Coulomb
#define Q2         Q*Q
#define PI         3.1415926535
#define PIHALF     PI/2
#define DIELECTRIC 1
#define COULOMB    1/(4*PI*DIELECTRIC)
#define Mp         1.672623110E-27  // proton mass in kg
#define Me         9.109389754E-31  // electron mass in kg

__device__ double sqr(double x) {
	return x*x;
}

__device__ double getDistanceSquared(int p1, int p2, double *x, double *y) {
	double distance = 0;
	if ((p1<Np) && (p2<Np)) {
		distance = sqr(x[p1]-x[p2])+sqr((y[p1]-y[p2]));		
	}
	return distance;
}


// Projects a magnitude along x and y axis
__device__ void projectVectorXY(double magnitude,
                              double x1, double x2, double y1, double y2, 
                              double *px, double *py) {

	double distance;
	double deltax; double deltay;
	
	deltax = (x2-x1);
	deltay = (y2-y1);
	distance = sqrt(sqr(deltax) + sqr(deltay));	

	// TODO: double check if projection formula is correct
	*px = magnitude * deltax / distance; 
	*py = magnitude * deltay / distance;
	
}  

__device__ void projectAonB(double a_x, double a_y, double b_x, double b_y, 
                            double *ab_x, double *ab_y) {
	// a_proj_on_b = dotproduct(a,b)/length(b)^2 * b	
	double dotproduct = a_x*b_x + a_y*b_y; //4
	double len_b_2 = sqr(b_x)+sqr(b_y); //16
	double factor = dotproduct/len_b_2; //0.25	

	*ab_x = b_x * factor;
	*ab_y = b_y * factor;
}

__device__ void getElectricAcceleration(int p1, int p2, 
                                        double *x, double *y, double *ax, double *ay) {
	
	// acceleration is calculated on p1	
	double acceleration=0;
	double mass;
	if ((p1 % 2) == 0) // even indexes are proton
		mass = Mp;
	else    mass = Me; // odd indexes are electrons
		

	if ((p1<Np) && (p2<Np)) {
		acceleration = COULOMB * Q2 * getDistanceSquared(p1, p2, x, y) / mass;  
		// now we need to project acceleration along (x1-x2) and (y1-y2)
		projectVectorXY(acceleration, x[p1], x[p2], y[p1], y[p2], ax, ay);
	}

}


__global__ void simulate_dipoles(double *x, double *y, double *omega, 
				 double *ax, double *ay, double *angle,
                                 double *E_pot, int steps) {
	int tid = blockIdx.x; // particle number (thread id)
        
        int iselectron;
	double ax_temp; double ay_temp;
        
        double a_on_b_1_x;
	double a_on_b_1_y;
	double a_on_b_2_x;
	double a_on_b_2_y;

	
	for (int j=0; j<steps; j++) {
		
	   	if (tid<Np) {
			// 1. calculate acceleration on tid particle
			ax[tid]=0; ay[tid]=0;

			for (int i=0; i<Np; i++) {
				if (i==tid) continue; // we do not calculate on ourself
				iselectron = (i % 2);
				// we do not calculate acceleration on the partner particle on the dipole
				if ((iselectron==1) && (i==tid-1)) continue;
				if ((iselectron==0) && (i==tid+1)) continue;	
			
				getElectricAcceleration(tid, i, x, y, &ax_temp, &ay_temp);
				ax[tid]=ax[tid]+ax_temp;
				ay[tid]=ay[tid]+ay_temp;
			}

                
			__syncthreads();


			// 2. update omega (angular velocity) with the projected acceleration,
			//    we do it only on half of the cores
			if (tid%2==0) {
                        	int did = (int)tid / 2; // dipole identifier

				// the axis of projection is perpendicular
				// of (x1,y1)<-->(x2,y2)
		        	// we take the angle and add 90 degrees
                        	// see definition of atan2 on wikipedia for polar coordinates conversion
				// https://en.wikipedia.org/wiki/Polar_coordinate_system
				double deltax = x[tid+1]-x[tid];
				double deltay = y[tid+1]-y[tid];			
			
				angle[did] = atan2f(deltax, deltay);

				double r_did = sqrt(sqr(deltax)+sqr(deltay))/2;
                        	double center_x = deltax/2 + x[tid];
                        	double center_y = deltay/2 + y[tid];

				// now we need to project a=(ax[tid], ay[tid]) on 
                        	double tan_x_1 = center_x + cos(angle[did]+PIHALF) * r_did;
				double tan_y_1 = center_y + sin(angle[did]+PIHALF) * r_did;
								
				projectAonB(ax[tid],ay[tid],tan_x_1,tan_y_1,&a_on_b_1_x,&a_on_b_1_y);

						
				double len_proj_acc_1 = sqrt(sqr(a_on_b_1_x)+sqr( a_on_b_1_y));

				// now we need to project a=(ax[tid+1], ay[tid+2]) on 
                        	double tan_x_2 = center_x - cos(angle[did]+PIHALF) * r_did;
				double tan_y_2 = center_y - sin(angle[did]+PIHALF) * r_did;
			                        
				projectAonB(ax[tid+1],ay[tid+1],tan_x_2,tan_y_2,&a_on_b_2_x,&a_on_b_2_y);
				double len_proj_acc_2 = sqrt(sqr(a_on_b_2_x)+sqr( a_on_b_2_y));

				// if they have the same sign, they slow down the dipole,
				// if they have opposite sign, they accelerate it
				double len_proj_acc = len_proj_acc_1 - len_proj_acc_2;

				// now we add this acceleration to the current angular velocity omega
				// TODO: double check if /2*PI is necessary
			
			
				/*
		 		// This is a test for method projectAonB
				double t_ax=1;
				double t_ay=1;
				double t_bx=4;
				double t_by=0;
				double res_x=0;
				double res_y=0;
			
				projectAonB(t_ax, t_ay, t_bx, t_by, &res_x, &res_y);
				angle[did]=res_x;
				omega[did]=res_y;
				*/

			
                        	omega[did] = omega[did] + (len_proj_acc* TIMESTEP)/(r_did);//2*PI);
				angle[did] = angle[did] + omega[did]*TIMESTEP;

				//TODO: we could calculate this part on all cores
				// we calculate now the new position for the dipoles
				x[tid] = center_x + cos(angle[did])*r_did;
				y[tid] = center_y + sin(angle[did])*r_did;
			
				x[tid+1] = center_x - cos(angle[did])*r_did;
				y[tid+1] = center_y - sin(angle[did])*r_did;

				// and we clear accelerations as we will recalculate them in the next round
				ax[tid]=0;
				ay[tid]=0;
				ax[tid+1]=0;
				ay[tid+1]=0;
					
			} // tid%2==0
		
			__syncthreads();
	   	}  // tid<Np
	} // for j<steps

}




int main(void) {
	double x[Np]; double y[Np]; double omega[Nd];
        double ax[Np]; double ay[Np]; double angle[Nd];
        double E_pot[Nd]; // potential energy of the system
        
        double *dev_x, *dev_y, *dev_omega;
	double *dev_ax, *dev_ay, *dev_angle;
	double *dev_E_pot;

	cudaMalloc( (void**)&dev_x,     Np*sizeof(double));
        cudaMalloc( (void**)&dev_y,     Np*sizeof(double));
        cudaMalloc( (void**)&dev_omega, Nd*sizeof(double));
	cudaMalloc( (void**)&dev_ax,    Np*sizeof(double));
	cudaMalloc( (void**)&dev_ay,    Np*sizeof(double));
	cudaMalloc( (void**)&dev_angle, Nd*sizeof(double));
        cudaMalloc( (void**)&dev_E_pot, Nd*sizeof(double));

	//TODO: init variables with Box-Muller and 2D gauss curve
        //      for two bodies with different centers and radia
	for (int i=0; i<Np; i++) {
		x[i] = (i*i)/1E6;
		y[i] = i/1000.0;
		ax[i] = 0;
		ay[i] = 0;
	}
	for (int i=0; i<Nd; i++) {
		omega[i] = 0;
		angle[i] = 0;
		E_pot[i] = 0;
	}

        cudaMemcpy(dev_x, x, Np*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, Np*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_omega, omega, Nd*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ax, ax, Np*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ay, ay, Np*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_angle, angle, Nd*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_E_pot, E_pot, Nd*sizeof(double), cudaMemcpyHostToDevice);
	
	simulate_dipoles<<<Np,1>>>(dev_x, dev_y, dev_omega, dev_ax, dev_ay, dev_angle, dev_E_pot, 1000);
	
	cudaMemcpy(x, dev_x, Np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dev_y, Np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(omega, dev_omega, Nd*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ax, dev_ax, Np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ay, dev_ay, Np*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(angle, dev_angle, Nd*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(E_pot, dev_E_pot, Nd*sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<Np; i++) {
		printf("i: %d x: %g y: %g ax: %g ay %g angle %g omega %g \n", 
                        i, x[i], y[i], ax[i], ay[i], angle[(int)i/2], omega[(int)i/2]);
	}
	
	cudaFree(dev_x);
        cudaFree(dev_y);
	cudaFree(dev_omega);
	cudaFree(dev_ax);
	cudaFree(dev_ay);
	cudaFree(dev_angle);
	cudaFree(dev_E_pot);

}




