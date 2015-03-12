/// COMMENT

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
using namespace std;

float** h_ellipse_vertex;
float* h_torus_vertex;
float* h_torus_surface;
float* sweep_origin;

int number_sweep_steps;
int number_ellipse_points;

// Count the lines in a file
int countLines(char *filename) {
	ifstream fin(filename);
	int input_size = count(istreambuf_iterator<char>(fin), istreambuf_iterator<char>(), '\n');
	fin.seekg(ios::beg);
	return input_size;
}

// Read from file
float** readFromFile(char *filename)
{
	ifstream fin(filename);
	int len = countLines(filename);
	number_ellipse_points = len;
	float **arr = new float*[len];
	for(int i=0; i<len; i++) {
		arr[i] = new float[4];
		fin>>arr[i][0]>>arr[i][1]>>arr[i][2];
		arr[i][3] = 1.0f;
	}
	return arr;
}

// Write to file
void writeToFile(char *filename, float** arr, int x, int y)
{
	ofstream fout(filename);
	for(int i = 0; i<y; i++) {
		for(int j=0; j<x; j++) {
			fout<<arr[i][j]<<' ';
    }
		fout<<endl;
  }
	fout.flush();
	fout.close();
}

// Write to console
void writeToConsole(float** arr, int x, int y) {
	for(int i = 0; i<y; i++) {
		for(int j=0; j<x; j++) {
			cout<<arr[i][j]<<' ';
    }
		cout<<endl;
  }
}

// Matrix multiplication
// X is the number of ROWS, Y is the number of COLS.
float** matrix_mul(float **a, float **b, int ax, int ay, int bx, int by)
{
	float** result = new float * [ax];
  //init array
  for(int i=0; i < ax; i++) {
    result[i] = new float[by];
  }
  //for every row in the result
  for(int i=0; i < ax; i++) {
    //for every column in the result
    for(int j=0; j < by; j++) {
      float sum = 0;
      //find the sum of the multiplied row and column
      for(int k=0; k < ay; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
	return result;
}

// Rotation Transformation Matrix
float** rotation_matrix(int angle)
{
	float** rotation = new float*[4];
	rotation[0] = new float[4];
  rotation[0][0] = cos(angle);
	rotation[0][1] = 0;
  rotation[0][2] = sin(angle);
  rotation[0][3] = 0;
	rotation[1] = new float[4];
  rotation[1][0] = rotation[1][2] = rotation[1][3] = 0;
	rotation[1][1] = 1;
	rotation[2] = new float[4];
  rotation[2][0] = -sin(angle);
	rotation[2][1] = rotation[2][3] = 0;
  rotation[2][2] = cos(angle);
	rotation[3] = new float[4];
  rotation[3][0] = rotation[3][1] = rotation[3][2] = 0;
  rotation[3][3] = 1;
  return rotation;
}

// Sweep the ellipse
void sweep()
{
	
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
  
	cout<<"Start reading"<<endl;
	float **h_ellipse_vertex = readFromFile("../ellipse_matrix.txt");
	//for(int i=0; i<number_ellipse_points; i++) {
	//	cout<<h_ellipse_vertex[i][0]<<' '<<h_ellipse_vertex[i][1]<<' '<<h_ellipse_vertex[i][2]<<' '<<h_ellipse_vertex[i][3]<<endl;
	//}


  float ** arr1 = new float*[2];
  arr1[0] = new float[2];
  arr1[1] = new float[2];

  arr1[0][0] = 1;
  arr1[0][1] = 2;
  arr1[1][0] = 3;
  arr1[1][1] = 4;


  float ** arr2 = new float*[2];
  arr2[0] = new float[2];
  arr2[1] = new float[2];

  arr2[0][0] = 4;
  arr2[0][1] = 3;
  arr2[1][0] = 2;
  arr2[1][1] = 1;


  float ** arr3 = matrix_mul(arr1, arr2, 2, 2, 2, 2);


  writeToConsole(rotation_matrix(45), 2, 2);

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
