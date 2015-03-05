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
float* sweep_origin;

int number_sweep_steps;
int number_ellipse_points;

// Read from file
void readFromFile(char *filename, float *arr)
{
  //Martin
}

// Write to file
void writeToFile(char *filename, float* arr)
{
  //Martin
}

// Matrix multiplication
float* matrix_mul(float *a, float *b, int ax, int ay, int bx, int by)
{
  float* result = new float[ax][by];
  //Yu-Yang
}

// Sweep the ellypse
void sweep()
{
  //Martin: do one ring after the other.
}

// Generate the surface table
void generateSurfaceTable()
{
  //Yu-Yang: assumming rings are one after the other.
}

int main(int argc, char** argv)
{
	//CUDA properties
	int devID;
	cudaDeviceProp props;

	// get number of SMs on this GPU
	cutilSafeCall(cudaGetDevice(&devID));
	cutilSafeCall(cudaGetDeviceProperties(&props, devID));
  
  //
  // INIT DATA HERE
  //
  
  // print information
  cout << "Number of ellipse vertices : " << number_ellipse_points << endl;
  cout << "Number of rotational sweep steps : " << number_sweep_steps << endl;
  cout << "Rotational sweep origin : " << "[" << sweep_origin[0] << ", " << sweep_origin[1] << ", " << sweep_origin[2] << "]" << endl;
  
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
