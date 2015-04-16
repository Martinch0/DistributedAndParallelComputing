/////////////////////////////////////////////////////////////////////////////
// 
//  NAIVE IMPLEMENTATION OF TORUS ROTATION
//  DISTRIBUTED AND PARALLEL COMPUTATION MINIPROJECT
//  2014-2015
//  
//  This is the naive device implementation of torus rotation.
//  The torus is generated using a sweep function and a matrix file it reads.
//  For more information, refer to the README file attached.
//  
//  authors: Martin Mihov (1229174) and Yu-Yang Lin (1228863)
//
//////////////////////////////////////////////////////////////////////////////

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
using namespace std;


///////////////////////////
// Matrix multiplication //
///////////////////////////
// arows and brows is the number of ROWS in a and b, respectively
// acols and bcols is the number of COLS.
// The resulting matrix is saved in out, which needs to be preallocated.
__device__ void matrix_mulk(float* a, float* b, long arows, long acols, long brows, long bcols, float* out)
{
  //matrix_mul(&a[0], &b[0], 4, 1, 1, 4, &c[0]);
  //for every row in the result
  for(long i=0; i < arows; i++) {
    //for every column in the result
    for(long j=0; j < bcols; j++) {
      float sum = 0;
      //find the sum of the multiplied row and column
      for(long k=0; k < acols; k++) {
        sum += a[getIndex(i, k, acols)] * b[getIndex(k, j, bcols)];
      }
      out[getIndex(i, j, bcols)] = sum;
    }
  }
}

/////////////////////////////////////////
// Rotation Transformation Matrix on X //
/////////////////////////////////////////
__device__ void rotmat_Xk(float angle, float* rotation)
{
  float angle_rad = PI * angle / 180.0;

  //First row.
  rotation[getIndex(0, 0, 4)] = 1;
  rotation[getIndex(0, 1, 4)] = 0;
  rotation[getIndex(0, 2, 4)] = 0;
  rotation[getIndex(0, 3, 4)] = 0;

  //Second row.
  rotation[getIndex(1, 0, 4)] = 0;
  rotation[getIndex(1, 1, 4)] = cos(angle_rad);
  rotation[getIndex(1, 2, 4)] = -sin(angle_rad);
  rotation[getIndex(1, 3, 4)] = 0;

  //Third row.
  rotation[getIndex(2, 0, 4)] = 0;
  rotation[getIndex(2, 1, 4)] = sin(angle_rad);
  rotation[getIndex(2, 2, 4)] = cos(angle_rad);
  rotation[getIndex(2, 3, 4)] = 0;

  //Fourth row.
  rotation[getIndex(3, 0, 4)] = 0;
  rotation[getIndex(3, 1, 4)] = 0;
  rotation[getIndex(3, 2, 4)] = 0;
  rotation[getIndex(3, 3, 4)] = 1;
}

/////////////////////////////////////////
// Rotation Transformation Matrix on Y //
/////////////////////////////////////////
__device__ void rotmat_Yk(float angle, float* rotation)
{
  float angle_rad = PI * angle / 180.0;

  //First row.
  rotation[getIndex(0, 0, 4)] = cos(angle_rad);
  rotation[getIndex(0, 1, 4)] = 0;
  rotation[getIndex(0, 2, 4)] = sin(angle_rad);
  rotation[getIndex(0, 3, 4)] = 0;

  //Second row.
  rotation[getIndex(1, 0, 4)] = 0;
  rotation[getIndex(1, 1, 4)] = 1;
  rotation[getIndex(1, 2, 4)] = 0;
  rotation[getIndex(1, 3, 4)] = 0;

  //Third row.
  rotation[getIndex(2, 0, 4)] = -sin(angle_rad);
  rotation[getIndex(2, 1, 4)] = 0;
  rotation[getIndex(2, 2, 4)] = cos(angle_rad);
  rotation[getIndex(2, 3, 4)] = 0;

  //Fourth row.
  rotation[getIndex(3, 0, 4)] = 0;
  rotation[getIndex(3, 1, 4)] = 0;
  rotation[getIndex(3, 2, 4)] = 0;
  rotation[getIndex(3, 3, 4)] = 1;
}

/////////////////////////////////////////
// Rotation Transformation Matrix on Z //
/////////////////////////////////////////
__device__ void rotmat_Zk(float angle, float* rotation)
{
  float angle_rad = PI * angle / 180.0;

  //First row.
  rotation[getIndex(0, 0, 4)] = cos(angle_rad);
  rotation[getIndex(0, 1, 4)] = -sin(angle_rad);
  rotation[getIndex(0, 2, 4)] = 0;
  rotation[getIndex(0, 3, 4)] = 0;

  //Second row.
  rotation[getIndex(1, 0, 4)] = sin(angle_rad);
  rotation[getIndex(1, 1, 4)] = cos(angle_rad);
  rotation[getIndex(1, 2, 4)] = 0;
  rotation[getIndex(1, 3, 4)] = 0;

  //Third row.
  rotation[getIndex(2, 0, 4)] = 0;
  rotation[getIndex(2, 1, 4)] = 0;
  rotation[getIndex(2, 2, 4)] = 1;
  rotation[getIndex(2, 3, 4)] = 0;

  //Fourth row.
  rotation[getIndex(3, 0, 4)] = 0;
  rotation[getIndex(3, 1, 4)] = 0;
  rotation[getIndex(3, 2, 4)] = 0;
  rotation[getIndex(3, 3, 4)] = 1;
}

// The kernel for rotating the torus by one step.
__global__ void rotate_torus_kernel(float* d_torus_vertex, float* d_torus_normals, long number_torus_points, float time) {

  // Preallocate memory for the rotation matrices
  float rotmatX[16];
  float rotmatY[16];
  float rotmatZ[16];

  // Calculate the rotation matrices based on the elapsed time.
  rotmat_Xk(cos(time/5051)*3, &rotmatX[0]);
  rotmat_Yk(cos(time/2063)*3, &rotmatY[0]);
  rotmat_Zk(cos(time/1433)*2, &rotmatZ[0]);

  float point[4];
  float normal[4];

  long i = blockIdx.x * blockDim.x + threadIdx.x;
  // Rotate the point with id corresponding to the current thread and block, and its normal
  if (i < number_torus_points) { // If invalid point, do nothing
    point[getIndex(0, 0, 1)] = d_torus_vertex[getIndex(i, 0, 4)];
    point[getIndex(1, 0, 1)] = d_torus_vertex[getIndex(i, 1, 4)];
    point[getIndex(2, 0, 1)] = d_torus_vertex[getIndex(i, 2, 4)];
    point[getIndex(3, 0, 1)] = d_torus_vertex[getIndex(i, 3, 4)];

    normal[getIndex(0, 0, 1)] = d_torus_normals[getIndex(i, 0, 4)];
    normal[getIndex(1, 0, 1)] = d_torus_normals[getIndex(i, 1, 4)];
    normal[getIndex(2, 0, 1)] = d_torus_normals[getIndex(i, 2, 4)];
    normal[getIndex(3, 0, 1)] = d_torus_normals[getIndex(i, 3, 4)];

    float temp[16];
    float combo[16]; 
    float newpoint[4]; 
    float newNormal[4];

    // Calculate the combined transformation matrix
    matrix_mulk(&rotmatZ[0], &rotmatX[0], 4, 4, 4, 4, &temp[0]);
    matrix_mulk(&rotmatY[0], &temp[0], 4, 4, 4, 4, &combo[0]);

    // Rotate the point and its normal
    matrix_mulk(&combo[0], &point[0], 4, 4, 4, 1, &newpoint[0]);
    matrix_mulk(&combo[0], &normal[0], 4, 4, 4, 1, &newNormal[0]);

    d_torus_vertex[getIndex(i, 0, 4)] = newpoint[getIndex(0, 0, 1)];
    d_torus_vertex[getIndex(i, 1, 4)] = newpoint[getIndex(1, 0, 1)];
    d_torus_vertex[getIndex(i, 2, 4)] = newpoint[getIndex(2, 0, 1)];
    d_torus_vertex[getIndex(i, 3, 4)] = newpoint[getIndex(3, 0, 1)];

    d_torus_normals[getIndex(i, 0, 4)] = newNormal[getIndex(0, 0, 1)];
    d_torus_normals[getIndex(i, 1, 4)] = newNormal[getIndex(1, 0, 1)];
    d_torus_normals[getIndex(i, 2, 4)] = newNormal[getIndex(2, 0, 1)];
    d_torus_normals[getIndex(i, 3, 4)] = newNormal[getIndex(3, 0, 1)];
  }
}

// A function which initializes and starts the kernel.
void launch_rotate_kernel(float* h_torus_vertex, float* h_torus_normals, float* d_torus_vertex, float* d_torus_normals, long numPoints) {

  // Get the current ellapsed time.
  double time = glutGet(GLUT_ELAPSED_TIME);

  // Calculate the number of operations needed.
  long size = sizeof(float) * getSize(numPoints, 4);

	// Copy host memory to device
	cutilSafeCall(cudaMemcpy(d_torus_vertex, h_torus_vertex, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_torus_normals, h_torus_normals, size, cudaMemcpyHostToDevice));
  cudaThreadSynchronize();

  // Calculate a suitable block and grid size for the kernel.
  long block_size = 512;
  long grid_size = numPoints / block_size + 1;

  // Run the kernel.
  rotate_torus_kernel<<<grid_size, block_size>>>(d_torus_vertex, d_torus_normals, numPoints, time);
  cudaThreadSynchronize();

  cutilSafeCall(cudaMemcpy(h_torus_vertex, d_torus_vertex, size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(h_torus_normals, d_torus_normals, size, cudaMemcpyDeviceToHost));
}
