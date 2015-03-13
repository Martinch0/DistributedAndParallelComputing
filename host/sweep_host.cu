/// COMMENT

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
using namespace std;

 #define PI 3.14159265

float** h_ellipse_vertex;
float* h_torus_vertex;
float* h_torus_surface;
float* sweep_origin;

int number_sweep_steps = 10;
int number_ellipse_points;

char SEMI_COLON_CHAR = 59;

///////////////////////////////
// Count the lines in a file //
///////////////////////////////
int countLines(char *filename) {
  ifstream fin(filename);
  int input_size = count(istreambuf_iterator<char>(fin), istreambuf_iterator<char>(), '\n');
  fin.seekg(ios::beg);
  return input_size;
}

////////////////////
// Read from file //
////////////////////
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

///////////////////
// Write to file //
///////////////////
// Uses templates
template <typename T>
void writeToFile(char *filename, T** arr, int row, int col)
{
  ofstream fout(filename);
  for(int i = 0; i<row; i++) {
    for(int j=0; j<col; j++) {
      fout<<arr[i][j]<<' ';
    }
    fout<< ' ' << SEMI_COLON_CHAR << ' ' <<endl;
  }
  fout.flush();
  fout.close();
}

//////////////////////
// Write to console //
//////////////////////
// Uses templates
template <typename T>
void writeToConsole(T** arr, int row, int col) {
  for(int i = 0; i<row; i++) {
    for(int j=0; j<col; j++) {
      cout<<arr[i][j]<<' ';
    }
    cout<< ' ' << SEMI_COLON_CHAR << ' ' <<endl;
  }
}

///////////////////////////
// Matrix multiplication //
///////////////////////////
// X is the number of ROWS, Y is the number of COLS.
float** matrix_mul(float **a, float **b, int arows, int acols, int brows, int bcols)
{
  float** result = new float * [acols];
  //init array
  for(int i=0; i < acols; i++) {
    result[i] = new float[brows];
  }
  //for every row in the result
  for(int i=0; i < acols; i++) {
    //for every column in the result
    for(int j=0; j < brows; j++) {
      float sum = 0;
      //find the sum of the multiplied row and column
      for(int k=0; k < acols; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

////////////////////////////////////
// Rotation Transformation Matrix //
////////////////////////////////////
float** rotation_matrix(float angle)
{
  float angle_rad = PI * angle / 180.0;

  float** rotation = new float*[4];
  rotation[0] = new float[4];
  rotation[0][0] = cos(angle_rad);
  rotation[0][1] = 0;
  rotation[0][2] = sin(angle_rad);
  rotation[0][3] = 0;
  rotation[1] = new float[4];
  rotation[1][0] = 0;
  rotation[1][1] = 1;
  rotation[1][2] = 0;
  rotation[1][3] = 0;
  rotation[2] = new float[4];
  rotation[2][0] = -sin(angle_rad);
  rotation[2][1] = 0;
  rotation[2][2] = cos(angle_rad);
  rotation[2][3] = 0;
  rotation[3] = new float[4];
  rotation[3][0] = 0;
  rotation[3][1] = 0;
  rotation[3][2] = 0;
  rotation[3][3] = 1;
  return rotation;
}
  
// Sweep the ellipse
void sweep()
{
  
}

////////////////////////////////
// Generate the surface table //
////////////////////////////////
int**  generateSurfaceTable()
{
  //Yu-Yang: assumming rings are one after the other.
  //assumming matrixes are: arr[ellipse_number][x y z 1]
 
  // we need a surface for every point in the torus
  int number_torus_points = number_sweep_steps * number_ellipse_points;
  int** surface_table = new int * [number_torus_points];
  
  //init array
  for(int i=0; i < number_torus_points; i++) {
    // every surface is made of 4 points
    surface_table[i] = new int[4];
  }

  //for each ring on the torus
  for(int i=0; i < number_sweep_steps; i++) {
    //for each point on the ring
    for(int j=0; j < number_ellipse_points; j++) {
      
      //torus point is the ring number * points in an ellipse plus current point counter.
      int torus_point = (i*number_ellipse_points) + j + 1;

      //if not last ring
      if (torus_point + number_ellipse_points -1 < number_torus_points) {
        
        //last point in a ring
        if (torus_point % number_ellipse_points == 0) {
          //create surface square joining the last 2 points of the rings with the first two
          surface_table[torus_point-1][0] = torus_point;
          surface_table[torus_point-1][1] = torus_point + number_ellipse_points;
          surface_table[torus_point-1][2] = torus_point + 1;
          surface_table[torus_point-1][3] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square        
          surface_table[torus_point-1][0] = torus_point;
          surface_table[torus_point-1][1] = torus_point + number_ellipse_points;
          surface_table[torus_point-1][2] = torus_point + number_ellipse_points + 1;
          surface_table[torus_point-1][3] = torus_point + 1;
        }

      //if last ring
      } else {

        //last point in a ring
        if (torus_point % number_ellipse_points == 0) {
          //create surface square joining the last 2 points of the torus with the first two
          surface_table[torus_point-1][0] = torus_point;
          surface_table[torus_point-1][1] = 0;
          surface_table[torus_point-1][2] = 1;
          surface_table[torus_point-1][3] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square
          surface_table[torus_point-1][0] = torus_point;
          surface_table[torus_point-1][1] = (torus_point + number_ellipse_points) - number_torus_points;
          surface_table[torus_point-1][2] = (torus_point + number_ellipse_points) - number_torus_points + 1;
          surface_table[torus_point-1][3] = torus_point + 1;
        }
      } 
    }
  }

  return surface_table;
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

  float ** arr1 = new float*[1];
  arr1[0] = new float[1];
  arr1[1] = new float[1];
  arr1[2] = new float[1];
  arr1[3] = new float[1];

  arr1[0][0] = 1;
  arr1[1][0] = 2;
  arr1[2][0] = 3;
  arr1[3][0] = 1;


  float ** arr2 = new float*[2];
  arr2[0] = new float[2];
  arr2[1] = new float[2];

  arr2[0][0] = 4;
  arr2[0][1] = 3;
  arr2[1][0] = 2;
  arr2[1][1] = 1;

  float ** ry = rotation_matrix(45);
  float ** arr3 = matrix_mul(ry, arr1, 4, 4, 4, 1);

  writeToConsole(arr3, 4, 1);

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

