/////////////////////////////////////////////////////////////////////////////
// 
//  MAIN CPP SOURCE FOR NAIVE IMPLEMENTATION OF TORUS ROTATION
//  DISTRIBUTED AND PARALLEL COMPUTATION MINIPROJECT
//  2014-2015
//  
//  This is the main cpp code for the naive device implementation of torus rotation.
//  This code uses OpenGL to render the torus, and calls the kernel to calculate the new torus to render.
//  The torus is generated using a sweep function and a matrix file it reads.
//  For more information, refer to the README file attached.
//  
//  authors: Martin Mihov (1229174) and Yu-Yang Lin (1228863)
//
//////////////////////////////////////////////////////////////////////////////

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <sys/timeb.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

using namespace std;

#define PI 3.14159265

float* h_ellipse_vertex; // The points of the initial ellipse.
float* h_ellipse_normals; // The normals of the initial ellipse.

float* h_torus_vertex;  // The points of the generated torus.
float* h_torus_normals; // The normals of the generated torus.

// Device arrays
float* d_torus_vertex;  // The points of the generated torus.
float* d_torus_normals; // The normals of the generated torus.
long* h_torus_surface; // The surface table of the generated torus.

long number_sweep_steps = 500; // How many sweep steps will be done in a 360 rotations.
long number_ellipse_points; // The total number of points of the initial ellipse.
long number_torus_points; // The total number of points of the generated torus.
long torus_rotation[] = {1, 0, 2}; // The X, Y and Z rotation of the torus in degrees, performed each frame.

double nbFrames = 0; // Current frames per second.
double lastTime = 0; // Last update of FPS.

char SEMI_COLON_CHAR = 59;


// Forward definitions of some functions from included files.
long getIndex(long row, long col, long row_size);
long getSize(long row_size, long col_size);
void launch_rotate_kernel(float* h_torus_vertex, float* h_torus_normals, float* d_torus_vertex, float* d_torus_normals, long numPoints);

///////////////////////////////
// Count the lines in a file //
///////////////////////////////
long countLines(char *filename) {
  ifstream fin(filename);
  long input_size = count(istreambuf_iterator<char>(fin), istreambuf_iterator<char>(), '\n');
  fin.seekg(ios::beg);
  return input_size;
}

////////////////////
// Read from file //
////////////////////
// Reads the points of the predifined ellipse. Must be 3D points.
float* readFromFile(char *filename)
{
  ifstream fin(filename);
  long len = countLines(filename);
  number_ellipse_points = len;
  float* arr = new float[getSize(len, 4)];
  for(long i=0; i<len; i++) {
    fin>>arr[getIndex(i, 0, 4)]>>arr[getIndex(i, 1, 4)]>>arr[getIndex(i, 2, 4)];
    arr[getIndex(i, 3, 4)] = 1.0f;
  }
  return arr;
}

///////////////////
// Write to file //
///////////////////
// Uses templates
template <typename T>
void writeToFile(char *filename, char *varName, T* arr, long row, long col)
{
  ofstream fout(filename);
  fout<<varName<<"=[";
  for(long i = 0; i<row; i++) {
    for(long j=0; j<col; j++) {
      fout<<arr[getIndex(i, j, col)]<<' ';
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
void writeToConsole(T* arr, long row, long col) {
  for(long i = 0; i<row; i++) {
    for(long j=0; j<col; j++) {
      cout<<arr[getIndex(i, j, col)]<<' ';
    }
    cout<< ' ' << SEMI_COLON_CHAR << ' ' <<endl;
  }
}

///////////////////////////
// Matrix multiplication //
///////////////////////////
// X is the number of ROWS, Y is the number of COLS.
void matrix_mul(float* a, float* b, long arows, long acols, long brows, long bcols, float* out)
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
void rotmat_X(float angle, float* rotation)
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
void rotmat_Y(float angle, float* rotation)
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
void rotmat_Z(float angle, float* rotation)
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

/////////////////////////////////////////////
// Sweep function to generate torus points //
/////////////////////////////////////////////  
void sweep()
{

  // Define the size of the rotations step
  float step = 360.0f / number_sweep_steps;

  // The current angle
  float angle = 0;
  // The current point of the torus
  long curPosition = 0;

  // Preallocating matrices for intermediate results.
  float rot[16];
  float point[4];
  float normal[4];

  // Calculate the number of torus points and allocate the required memory.
  number_torus_points = number_sweep_steps * number_ellipse_points;
  h_torus_vertex = new float[getSize(number_torus_points, 4)];//torus points
  h_torus_normals = new float[getSize(number_torus_points, 4)];//torus normals
  
  // for every sweep step
  for(long i = 0; i<number_sweep_steps; i++) {
    rotmat_Y(angle, &rot[0]);

    // for every ellipse point
    for(long j = 0; j<number_ellipse_points; j++) {
      point[getIndex(0, 0, 1)] = h_ellipse_vertex[getIndex(j, 0, 4)];
      point[getIndex(1, 0, 1)] = h_ellipse_vertex[getIndex(j, 1, 4)];
      point[getIndex(2, 0, 1)] = h_ellipse_vertex[getIndex(j, 2, 4)];
      point[getIndex(3, 0, 1)] = h_ellipse_vertex[getIndex(j, 3, 4)];


      normal[getIndex(0, 0, 1)] = h_ellipse_normals[getIndex(j, 0, 4)];
      normal[getIndex(1, 0, 1)] = h_ellipse_normals[getIndex(j, 1, 4)];
      normal[getIndex(2, 0, 1)] = h_ellipse_normals[getIndex(j, 2, 4)];
      normal[getIndex(3, 0, 1)] = h_ellipse_normals[getIndex(j, 3, 4)];

      float newPoint[4];
      float newNormal[4];

      // Rotate the point and its normal
      matrix_mul(&rot[0], &point[0], 4, 4, 4, 1, &newPoint[0]);
      matrix_mul(&rot[0], &normal[0], 4, 4, 4, 1, &newNormal[0]);

      h_torus_vertex[getIndex(curPosition, 0, 4)] = newPoint[getIndex(0, 0, 1)];
      h_torus_vertex[getIndex(curPosition, 1, 4)] = newPoint[getIndex(1, 0, 1)];
      h_torus_vertex[getIndex(curPosition, 2, 4)] = newPoint[getIndex(2, 0, 1)];
      h_torus_vertex[getIndex(curPosition, 3, 4)] = newPoint[getIndex(3, 0, 1)];
  
      h_torus_normals[getIndex(curPosition, 0, 4)] = newNormal[getIndex(0, 0, 1)];
      h_torus_normals[getIndex(curPosition, 1, 4)] = newNormal[getIndex(1, 0, 1)];
      h_torus_normals[getIndex(curPosition, 2, 4)] = newNormal[getIndex(2, 0, 1)];
      h_torus_normals[getIndex(curPosition, 3, 4)] = newNormal[getIndex(3, 0, 1)];

      curPosition++;
      
    }

    // Increase the current angle.
    angle += step;
  }
}

////////////////////////////////
// Generate the surface table //
////////////////////////////////
void  generateSurfaceTable()
{
  //Yu-Yang: assumming rings are one after the other.
  //assumming matrixes are: arr[ellipse_number][x y z 1]
 
  // we need a surface for every point in the torus
  h_torus_surface = new long[getSize(number_torus_points, 4)];

  //for each ring on the torus
  for(long i=0; i < number_sweep_steps; i++) {
    //for each point on the ring
    for(long j=0; j < number_ellipse_points; j++) {
      
      //torus point is the ring number * points in an ellipse plus current point counter.
      long torus_point = (i*number_ellipse_points) + j + 1;

      //if not last ring
      if (torus_point + number_ellipse_points -1 < number_torus_points) {
        
        //last point in a ring
        if (torus_point % number_ellipse_points == 0) {
          //create surface square joining the last 2 points of the rings with the first two
          h_torus_surface[getIndex(torus_point-1, 0, 4)] = torus_point;
          h_torus_surface[getIndex(torus_point-1, 1, 4)] = torus_point + number_ellipse_points;
          h_torus_surface[getIndex(torus_point-1, 2, 4)] = torus_point + 1;
          h_torus_surface[getIndex(torus_point-1, 3, 4)] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square        
          h_torus_surface[getIndex(torus_point-1, 0, 4)] = torus_point;
          h_torus_surface[getIndex(torus_point-1, 1, 4)] = torus_point + number_ellipse_points;
          h_torus_surface[getIndex(torus_point-1, 2, 4)] = torus_point + number_ellipse_points + 1;
          h_torus_surface[getIndex(torus_point-1, 3, 4)] = torus_point + 1;
        }

      //if last ring
      } else {

        //last point in a ring
        if (torus_point % number_ellipse_points == 0) {
          //create surface square joining the last 2 points of the torus with the first two
          h_torus_surface[getIndex(torus_point-1, 0, 4)] = torus_point;
          h_torus_surface[getIndex(torus_point-1, 1, 4)] = 1;
          h_torus_surface[getIndex(torus_point-1, 2, 4)] = 2;
          h_torus_surface[getIndex(torus_point-1, 3, 4)] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square
          h_torus_surface[getIndex(torus_point-1, 0, 4)] = torus_point;
          h_torus_surface[getIndex(torus_point-1, 1, 4)] = (torus_point + number_ellipse_points) - number_torus_points;
          h_torus_surface[getIndex(torus_point-1, 2, 4)] = (torus_point + number_ellipse_points) - number_torus_points + 1;
          h_torus_surface[getIndex(torus_point-1, 3, 4)] = torus_point + 1;
        }
      } 
    }
  }
}

/////////////////////////
/////// GL CODE /////////
/////////////////////////
GLfloat light_diffuse[] = {1.0, 0.0, 0.0, 0.1};  /* Red diffuse light. */
GLfloat light_position[] = {0.0, 500.0, 1.0, 0.0};  /* Infinite light location. */
GLfloat light_ambient[] = { 0.5, 0.0, 0.0, 1.0 }; /* Ambient light. */

// Draws the torus
void drawTorus()
{
  GLfloat * normal = new GLfloat[4];

  // Draw every surface with its normal
  for (long i = 0; i < number_torus_points; i++) {
    glBegin(GL_QUADS);

    normal[0] = h_torus_normals[getIndex(i, 0, 4)];
    normal[1] = h_torus_normals[getIndex(i, 1, 4)];
    normal[2] = h_torus_normals[getIndex(i, 2, 4)];

    glNormal3fv(&normal[0]);
    glVertex3fv(&h_torus_vertex[getIndex(h_torus_surface[getIndex(i, 0, 4)]-1, 0, 4)]);
    glVertex3fv(&h_torus_vertex[getIndex(h_torus_surface[getIndex(i, 1, 4)]-1, 0, 4)]);
    glVertex3fv(&h_torus_vertex[getIndex(h_torus_surface[getIndex(i, 2, 4)]-1, 0, 4)]);
    glVertex3fv(&h_torus_vertex[getIndex(h_torus_surface[getIndex(i, 3, 4)]-1, 0, 4)]);
    glEnd();
  }

  delete[] normal;
}

// Update the display window. Called every frame.
void display()
{
  // Get the current elapsed time and calculate the Frames Per Second
  double currentTime = glutGet(GLUT_ELAPSED_TIME);
  nbFrames += 1.0;
  if ( currentTime - lastTime >= 1000 ){
    // printf and reset timer
    char buffer[32];
    nbFrames += (currentTime - lastTime) / 1000.0;
    snprintf(buffer, 32, "Naive Torus - FPS: %f", nbFrames);

    glutSetWindowTitle(buffer);
    nbFrames = 0;
    lastTime = currentTime;
  }

  // Launch the kernel to perform a rotation
  launch_rotate_kernel(&h_torus_vertex[0], &h_torus_normals[0], &d_torus_vertex[0], &d_torus_normals[0], number_torus_points);
  
  // Draw the torus.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawTorus();
  glutSwapBuffers();
  glutPostRedisplay();
}

// Initialize the OpenGL
void init()
{
  /* Enable a single OpenGL light. */
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);


  /* Use depth buffering for hidden surface elimination. */
  glEnableClientState(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10000.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 2000.0, 0.0,  /* eye is at (0,2000,0) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 0.0, 1.0);      /* up is in positive Z direction */

}

// Initialize the display window
void displayTorus(int argc, char **argv)
{
  glewInit();
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1000,1000);
  glutCreateWindow("Torus");
  glutDisplayFunc(display);
  init();
  glutMainLoop();
}

/////////////////////////
//// END GL CODE ////////
/////////////////////////

//////////////
// TIMERS  ///
//////////////
long getMilliCount(){
	timeb tb;
	ftime(&tb);
	long nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

long getMilliSpan(long nTimeStart){
	long nSpan = getMilliCount() - nTimeStart;
	if(nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

void initializeTorus() {
  long start;
  long span;

  // Read the initial ellipse points and vertices.
  h_ellipse_vertex = readFromFile("../ellipse_matrix.txt");
  h_ellipse_normals = readFromFile("../ellipse_normals.txt");

  start = getMilliCount(); // Timer

  // Sweep the ellipse to generate the torus.
  sweep();

  span = getMilliSpan(start); // Timer
  cout<<"Generate torus vertex table: "<<span<< " ms; Points: "<<number_torus_points<<endl;

  // Preallocate memory on the device for the torus points and normals.
  cutilSafeCall(cudaMalloc((void **) &d_torus_vertex, sizeof(float) * getSize(number_torus_points, 4)));
  cutilSafeCall(cudaMalloc((void **) &d_torus_normals, sizeof(float) * getSize(number_torus_points, 4)));

  start = getMilliCount(); // Timer

  // Generate the torus surface table
  generateSurfaceTable();

  span = getMilliSpan(start); // Timer
  cout<<"Generate torus surface table: "<<span<< " ms"<<endl;
}

////////////
/// MAIN ///
////////////
int main(int argc, char** argv)
{
  int devID;
	cudaDeviceProp props;

	// get number of SMs on this GPU
	cutilSafeCall(cudaGetDevice(&devID));
	cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  // Initialize the torus.
  initializeTorus();

  // Start the diplaying and the rotation of the torus.
  displayTorus(argc, argv);

	cutilSafeCall(cudaFree(d_torus_vertex));
	cutilSafeCall(cudaFree(d_torus_normals));

	// exit and clean up device status
	cudaThreadExit();
  
  return 0;
}

