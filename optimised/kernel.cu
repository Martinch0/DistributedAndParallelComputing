
// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "cutil_inline.h"
#include <iostream>
#include "../custom_matrix.cpp"
#define PI 3.14159265
#define num_rpb 128

using namespace std;

///////////////////////////
// MATRIX MULTIPLICATION //
///////////////////////////
// acols and bcols is the number of columns of the two matrices.
__global__ void matrix_mulk(float* a, float* b, float* c, long acols, long bcols, float* out1, float* out2)
{
  int row = (blockIdx.x*num_rpb) + (threadIdx.x>>2);
  int col = threadIdx.x % 4;

  //transformed B matrix
  out1[getIndex(row, col, 4)] =  a[getIndex(col, 0, acols)] * b[getIndex(row, 0, bcols)];
  out1[getIndex(row, col, 4)] += a[getIndex(col, 1, acols)] * b[getIndex(row, 1, bcols)]; 
  out1[getIndex(row, col, 4)] += a[getIndex(col, 2, acols)] * b[getIndex(row, 2, bcols)]; 
  out1[getIndex(row, col, 4)] += a[getIndex(col, 3, acols)] * b[getIndex(row, 3, bcols)];

  //transformed C matrix
  out2[getIndex(row, col, 4)] =  a[getIndex(col, 0, acols)] * c[getIndex(row, 0, bcols)]; 
  out2[getIndex(row, col, 4)] += a[getIndex(col, 1, acols)] * c[getIndex(row, 1, bcols)];
  out2[getIndex(row, col, 4)] += a[getIndex(col, 2, acols)] * c[getIndex(row, 2, bcols)]; 
  out2[getIndex(row, col, 4)] += a[getIndex(col, 3, acols)] * c[getIndex(row, 3, bcols)];
}

/////////////////////////////////////////
// FORWARD DEFINITIONS FOR USE IN HOST //
/////////////////////////////////////////
void rotmat_X(float angle, float* rotation);
void rotmat_Y(float angle, float* rotation);
void rotmat_Z(float angle, float* rotation);
void matrix_mul(float* a, float* b, long arows, long acols, long brows, long bcols, float* out);

////////////////////////////
// KERNEL LAUNCH FUNCTION //
////////////////////////////
void launch_rotate_kernel(float* h_torus_vertex, float* h_torus_normals, float* d_torus_vertex, float* d_torus_normals, long numPoints, float *d_out1, float *d_out2) {

  // Preallocate memory for the rotation matrices.
  float rotmatX[16];
  float rotmatY[16];
  float rotmatZ[16];
  float rotTemp[16];
  float rotCombined[16];

  // Generate the rotation matrices based on the current elapsed time.
  double time = glutGet(GLUT_ELAPSED_TIME);

  rotmat_X(cos(time/5051)*3, &rotmatX[0]);
  rotmat_Y(cos(time/2063)*3, &rotmatY[0]);
  rotmat_Z(cos(time/1433)*2, &rotmatZ[0]);

  // Calculate the combined matrix and allocate memory for it on the device.
  matrix_mul(&rotmatZ[0], &rotmatX[0], 4, 4, 4, 4, &rotTemp[0]);
  matrix_mul(&rotmatY[0], &rotTemp[0], 4, 4, 4, 4, &rotCombined[0]);
  float *d_rotmat;
  
  cutilSafeCall(cudaMalloc((void **) &d_rotmat, sizeof(float) * 16));

  // Calculate the number of operations required.
  long size = sizeof(float) * getSize(numPoints, 4);

	// Copy host memory to device
	cutilSafeCall(cudaMemcpy(d_torus_vertex, h_torus_vertex, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_torus_normals, h_torus_normals, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_rotmat, rotCombined, sizeof(float) * 16, cudaMemcpyHostToDevice));
  cudaThreadSynchronize();

  // Start the kernel
  matrix_mulk<<<numPoints/num_rpb + 1, 4*num_rpb>>>(d_rotmat, d_torus_vertex, d_torus_normals, 4, 4, d_out1, d_out2);
  cudaThreadSynchronize();

  // Copy device memory to host.
  cutilSafeCall(cudaMemcpy(h_torus_vertex, d_out1, size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(h_torus_normals, d_out2, size, cudaMemcpyDeviceToHost));

  cutilSafeCall(cudaFree(d_rotmat));

}
