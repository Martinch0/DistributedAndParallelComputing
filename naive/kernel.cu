#include "cutil_inline.h"
#include <iostream>
#include "../custom_matrix.cpp"
#define PI 3.14159265
using namespace std;


///////////////////////////
// Matrix multiplication //
///////////////////////////
// X is the number of ROWS, Y is the number of COLS.
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

__global__ void rotate_torus_kernel(float* d_torus_vertex, float* d_torus_normals, long number_torus_points) {
  float rotmatX[16];
  float rotmatZ[16];
  rotmat_Xk(1, &rotmatX[0]);
  rotmat_Zk(2, &rotmatZ[0]);
  float point[4];
  float normal[4];

  long i = blockIdx.x * blockDim.x + threadIdx.x;
  // Rotate each point
  if (i < number_torus_points) {
    point[getIndex(0, 0, 1)] = d_torus_vertex[getIndex(i, 0, 4)];
    point[getIndex(1, 0, 1)] = d_torus_vertex[getIndex(i, 1, 4)];
    point[getIndex(2, 0, 1)] = d_torus_vertex[getIndex(i, 2, 4)];
    point[getIndex(3, 0, 1)] = d_torus_vertex[getIndex(i, 3, 4)];

    normal[getIndex(0, 0, 1)] = d_torus_normals[getIndex(i, 0, 4)];
    normal[getIndex(1, 0, 1)] = d_torus_normals[getIndex(i, 1, 4)];
    normal[getIndex(2, 0, 1)] = d_torus_normals[getIndex(i, 2, 4)];
    normal[getIndex(3, 0, 1)] = d_torus_normals[getIndex(i, 3, 4)];

    float combo[16]; 
    float newpoint[4]; 
    float newNormal[4];

    matrix_mulk(&rotmatZ[0], &rotmatX[0], 4, 4, 4, 4, &combo[0]);
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

void launch_rotate_kernel(float* h_torus_vertex, float* h_torus_normals, float* d_torus_vertex, float* d_torus_normals, long numPoints) {

  long size = sizeof(float) * getSize(numPoints, 4);
	// copy host memory to device
	cutilSafeCall(cudaMemcpy(d_torus_vertex, h_torus_vertex, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_torus_normals, h_torus_normals, size, cudaMemcpyHostToDevice));
  cudaThreadSynchronize();

  long block_size = 512;
  long grid_size = numPoints / block_size + 1;
  rotate_torus_kernel<<<grid_size, block_size>>>(d_torus_vertex, d_torus_normals, numPoints);
  cudaThreadSynchronize();



  cutilSafeCall(cudaMemcpy(h_torus_vertex, d_torus_vertex, size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(h_torus_normals, d_torus_normals, size, cudaMemcpyDeviceToHost));
}