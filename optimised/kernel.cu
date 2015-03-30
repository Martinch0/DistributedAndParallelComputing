
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
void rotmat_X(float angle, float* rotation);
void rotmat_Y(float angle, float* rotation);
void rotmat_Z(float angle, float* rotation);
void matrix_mul(float* a, float* b, long arows, long acols, long brows, long bcols, float* out);

__global__ void rotate_torus_kernel(float* d_torus_vertex, float* d_torus_normals, long number_torus_points, float* rotCombined) {

  float point[4];
  float normal[4];

  long i = blockIdx.x * blockDim.x + threadIdx.x;
  // Rotate each point
  if (i < number_torus_points) {
    point[0] = d_torus_vertex[getIndex(i, 0, 4)];
    point[1] = d_torus_vertex[getIndex(i, 1, 4)];
    point[2] = d_torus_vertex[getIndex(i, 2, 4)];
    point[3] = d_torus_vertex[getIndex(i, 3, 4)];

    normal[0] = d_torus_normals[getIndex(i, 0, 4)];
    normal[1] = d_torus_normals[getIndex(i, 1, 4)];
    normal[2] = d_torus_normals[getIndex(i, 2, 4)];
    normal[3] = d_torus_normals[getIndex(i, 3, 4)];

    float newpoint[4]; 
    float newNormal[4];

    matrix_mulk(&rotCombined[0], &point[0], 4, 4, 4, 1, &newpoint[0]);
    matrix_mulk(&rotCombined[0], &normal[0], 4, 4, 4, 1, &newNormal[0]);

    d_torus_vertex[getIndex(i, 0, 4)] = newpoint[0];
    d_torus_vertex[getIndex(i, 1, 4)] = newpoint[1];
    d_torus_vertex[getIndex(i, 2, 4)] = newpoint[2];
    d_torus_vertex[getIndex(i, 3, 4)] = newpoint[3];

    d_torus_normals[getIndex(i, 0, 4)] = newNormal[0];
    d_torus_normals[getIndex(i, 1, 4)] = newNormal[1];
    d_torus_normals[getIndex(i, 2, 4)] = newNormal[2];
    d_torus_normals[getIndex(i, 3, 4)] = newNormal[3];
  }
}

void launch_rotate_kernel(float* h_torus_vertex, float* h_torus_normals, float* d_torus_vertex, float* d_torus_normals, long numPoints) {

  float rotmatX[16];
  float rotmatZ[16];
  float rotCombined[16];
  rotmat_X(1, &rotmatX[0]);
  rotmat_Z(2, &rotmatZ[0]);

  matrix_mul(&rotmatZ[0], &rotmatX[0], 4, 4, 4, 4, &rotCombined[0]);
  float *d_rotmat;
  
  cutilSafeCall(cudaMalloc((void **) &d_rotmat, sizeof(float) * 16));

  long size = sizeof(float) * getSize(numPoints, 4);
	// copy host memory to device
	cutilSafeCall(cudaMemcpy(d_torus_vertex, h_torus_vertex, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_torus_normals, h_torus_normals, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_rotmat, rotCombined, sizeof(float) * 16, cudaMemcpyHostToDevice));
  cudaThreadSynchronize();

  long block_size = 512;
  long grid_size = numPoints / block_size + 1;
  rotate_torus_kernel<<<grid_size, block_size>>>(d_torus_vertex, d_torus_normals, numPoints, d_rotmat);
  cudaThreadSynchronize();



  cutilSafeCall(cudaMemcpy(h_torus_vertex, d_torus_vertex, size, cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(h_torus_normals, d_torus_normals, size, cudaMemcpyDeviceToHost));
}
