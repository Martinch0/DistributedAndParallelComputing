/// COMMENT

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
using namespace std;

float* h_ellipse_vertex;
float* h_torus_vertex;
float* h_torus_surface;

float sweep_radius;
int number_sweep_steps;
int number_ellipse_points;

// Read from file
void readFromFile(char *filename, float *arr)
{
}

// Write to file
void writeToFile(char *filename, float* arr)
{
}

// Matrix multiplication
void matrix_mul(float *a, float *b, float *res)
{
}

// Sweep the ellypse
void sweep()
{
}

// Generate the surface table
void generateSurfaceTable()
{
}

int main(int argc, char** argv)
{
	//CUDA properties
	int devID;
	cudaDeviceProp props;

	// get number of SMs on this GPU
	cutilSafeCall(cudaGetDevice(&devID));
	cutilSafeCall(cudaGetDeviceProperties(&props, devID));

        // create a timer
	unsigned int timer = 0;
        cutilCheckError(cutCreateTimer(&timer));
	// start the timer
        cutilCheckError(cutStartTimer(timer));

	//
	//
	// DO STUFF HERE
	//
	//

	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");

	// wait for device to finish
	cudaThreadSynchronize();

	cutilCheckError(cutStopTimer(timer));

	// exit and clean up device status
	cudaThreadExit();

	return 0;
}
