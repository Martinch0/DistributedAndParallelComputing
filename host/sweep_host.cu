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
float** h_torus_vertex;
int** h_torus_surface;
float* sweep_origin;

int number_sweep_steps = 100;
int number_ellipse_points;
int number_torus_points;

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
void writeToFile(char *filename, char *varName, T** arr, int row, int col)
{
  ofstream fout(filename);
	fout<<varName<<"=[";
  for(int i = 0; i<row; i++) {
    for(int j=0; j<col; j++) {
      fout<<arr[i][j]<<' ';
    }
    if(i!=row-1)
	    fout<< ' ' << SEMI_COLON_CHAR << ' ' <<endl;
  }
	fout<<"];";
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

/////////////////////////////////////////
// Rotation Transformation Matrix on X //
/////////////////////////////////////////
float** rotmat_Y(float angle)
{
  float angle_rad = PI * angle / 180.0;

  float** rotation = new float*[4];
  rotation[0] = new float[4];
  rotation[0][0] = 1
  rotation[0][1] = 0;
  rotation[0][2] = 0;
  rotation[0][3] = 0;
  rotation[1] = new float[4];
  rotation[1][0] = 0;
  rotation[1][1] = cos(angle_rad);
  rotation[1][2] = -sin(angle_rad);
  rotation[1][3] = 0;
  rotation[2] = new float[4];
  rotation[2][0] = 0;
  rotation[2][1] = sin(angle_rad);
  rotation[2][2] = cos(angle_rad);
  rotation[2][3] = 0;
  rotation[3] = new float[4];
  rotation[3][0] = 0;
  rotation[3][1] = 0;
  rotation[3][2] = 0;
  rotation[3][3] = 1;
  return rotation;
}

/////////////////////////////////////////
// Rotation Transformation Matrix on Y //
/////////////////////////////////////////
float** rotmat_Y(float angle)
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

/////////////////////////////////////////
// Rotation Transformation Matrix on Z //
/////////////////////////////////////////
float** rotmat_Y(float angle)
{
  float angle_rad = PI * angle / 180.0;

  float** rotation = new float*[4];
  rotation[0] = new float[4];
  rotation[0][0] = cos(angle_rad);
  rotation[0][1] = -sin(angle_rad);
  rotation[0][2] = 0;
  rotation[0][3] = 0;
  rotation[1] = new float[4];
  rotation[1][0] = 0;
  rotation[1][1] = 1;
  rotation[1][2] = 0;
  rotation[1][3] = 0;
  rotation[2] = new float[4];
  rotation[2][0] = sin(angle_rad);
  rotation[2][1] = cos(angle_rad);
  rotation[2][2] = 0;
  rotation[2][3] = 0;
  rotation[3] = new float[4];
  rotation[3][0] = 0;
  rotation[3][1] = 0;
  rotation[3][2] = 0;
  rotation[3][3] = 1;
  return rotation;
}

  
void sweep()
{

  float step = 360.0f / number_sweep_steps;
  float angle = 0;
	int curPosition = 0;
	float **rot;
	number_torus_points = number_sweep_steps * number_ellipse_points;
	h_torus_vertex = new float*[number_torus_points];
  
  for(int i = 0; i<number_sweep_steps; i++) {
		rot = rotmat_Y(angle);
    for(int j = 0; j<number_ellipse_points; j++) {
		  float **point = new float*[4];
			point[0] = new float[1];
			point[1] = new float[1];
			point[2] = new float[1];
			point[3] = new float[1];
			point[0][0] = h_ellipse_vertex[j][0];
			point[1][0] = h_ellipse_vertex[j][1];
			point[2][0] = h_ellipse_vertex[j][2];
			point[3][0] = h_ellipse_vertex[j][3];

			// Rotate the point
			float **newPoint = matrix_mul(rot, point, 4, 4, 4, 1);

			h_torus_vertex[curPosition] = new float[4];
			h_torus_vertex[curPosition][0] = newPoint[0][0];
			h_torus_vertex[curPosition][1] = newPoint[1][0];
			h_torus_vertex[curPosition][2] = newPoint[2][0];
			h_torus_vertex[curPosition][3] = newPoint[3][0];
			curPosition++;
			
    }
		angle += step;
  }
}

////////////////////////////////
// Generate the surface table //
////////////////////////////////
int**  generateSurfaceTable()
{
  //Yu-Yang: assumming rings are one after the other.
  //assumming matrixes are: arr[ellipse_number][x y z 1]
 
  // we need a surface for every point in the torus
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
          surface_table[torus_point-1][1] = 1;
          surface_table[torus_point-1][2] = 2;
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

  h_ellipse_vertex = readFromFile("../ellipse_matrix.txt");

	unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));             // create a timer
  cutilCheckError(cutStartTimer(timer));               // start the timer

	sweep();

	h_torus_surface = generateSurfaceTable();


	cutilCheckError(cutStopTimer(timer));
	double dSeconds = cutGetTimerValue(timer)/(1000.0);

	//Log througput
	printf("Seconds: %.4f \n", dSeconds);

  writeToFile("vertex_table.m", "vTable", h_torus_vertex, number_torus_points, 4);
  writeToFile("surface_table.m", "faces", h_torus_surface, number_torus_points, 4);

  //
  // INIT DATA HERE
  //
  
  // print information
  cout << "Number of ellipse vertices : " << number_ellipse_points << endl;
  cout << "Number of rotational sweep steps : " << number_sweep_steps << endl;
  //cout << "Rotational sweep origin : " << "[" << sweep_origin[0] << ", " << sweep_origin[1] << ", " << sweep_origin[2] << "]" << endl;

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finish
  cudaThreadSynchronize();

  cutilCheckError(cutStopTimer(timer));

  // exit and clean up device status
  cudaThreadExit();

  return 0;
}

