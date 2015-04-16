#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <sys/timeb.h>
#include "../custom_matrix.cpp"

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
int* h_torus_surface; // The surface table of the generated torus.

int number_sweep_steps = 500; // How many sweep steps will be done in a 360 rotations.
int number_ellipse_points; // The total number of points of the initial ellipse.
int number_torus_points; // The total number of points of the generated torus.
int torus_rotation[] = {1, 0, 2}; // The X, Y and Z rotation of the torus in degrees, performed each frame.

double nbFrames = 0; // Current frames per second.
double lastTime = 0; // Last update of FPS.

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
// Reads the points of the predifined ellipse. Must be 3D points.
float* readFromFile(char *filename)
{
  ifstream fin(filename);
  int len = countLines(filename);
  number_ellipse_points = len;
  float* arr = new float[getSize(len, 4)];
  for(int i=0; i<len; i++) {
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
void writeToFile(char *filename, char *varName, T* arr, int row, int col)
{
  ofstream fout(filename);
  fout<<varName<<"=[";
  for(int i = 0; i<row; i++) {
    for(int j=0; j<col; j++) {
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
void writeToConsole(T* arr, int row, int col) {
  for(int i = 0; i<row; i++) {
    for(int j=0; j<col; j++) {
      cout<<arr[getIndex(i, j, col)]<<' ';
    }
    cout<< ' ' << SEMI_COLON_CHAR << ' ' <<endl;
  }
}

///////////////////////////
// Matrix multiplication //
///////////////////////////
// arows and brows is the number of ROWS in a and b, respectively
// acols and bcols is the number of COLS.
// The resulting matrix is saved in out, which needs to be preallocated.
void matrix_mul(float* a, float* b, int arows, int acols, int brows, int bcols, float* out)
{
  //matrix_mul(&a[0], &b[0], 4, 1, 1, 4, &c[0]);
  //for every row in the result
  for(int i=0; i < arows; i++) {
    //for every column in the result
    for(int j=0; j < bcols; j++) {
      float sum = 0;
      //find the sum of the multiplied row and column
      for(int k=0; k < acols; k++) {
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
  int curPosition = 0;

  // Preallocating matrices for intermediate results.
  float rot[16];
  float point[4];
  float normal[4];

  // Calculate the number of torus points and allocate the required memory.
  number_torus_points = number_sweep_steps * number_ellipse_points;
  h_torus_vertex = new float[getSize(number_torus_points, 4)];//torus points
  h_torus_normals = new float[getSize(number_torus_points, 4)];//torus normals
  
  // for every sweep step
  for(int i = 0; i<number_sweep_steps; i++) {

    // Get the rotation matrix
    rotmat_Y(angle, &rot[0]);

    // for every ellipse point apply the rotation matrix and save it in the torus points array
    for(int j = 0; j<number_ellipse_points; j++) {
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

      // Rotate the point
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

    // Increase the current angle by the sweep step.
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
  h_torus_surface = new int[getSize(number_torus_points, 4)];

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


////////////////////
// Torus rotation //
////////////////////
void rotateTorus() {

  // Preallocate memory for the rotation matrices
  float rotmatX[16];
  float rotmatY[16];
  float rotmatZ[16];

  // Create random rotation matrices, based on the elapsed time.
  double time = glutGet(GLUT_ELAPSED_TIME);

  rotmat_X(cos(time/5051)*3, &rotmatX[0]);
  rotmat_Y(cos(time/2063)*3, &rotmatY[0]);
  rotmat_Z(cos(time/1433)*2, &rotmatZ[0]);

  float point[4];
  float normal[4];

  // Rotate each point
  for (int i = 0; i < number_torus_points; i++) {
    point[getIndex(0, 0, 1)] = h_torus_vertex[getIndex(i, 0, 4)];
    point[getIndex(1, 0, 1)] = h_torus_vertex[getIndex(i, 1, 4)];
    point[getIndex(2, 0, 1)] = h_torus_vertex[getIndex(i, 2, 4)];
    point[getIndex(3, 0, 1)] = h_torus_vertex[getIndex(i, 3, 4)];

    normal[getIndex(0, 0, 1)] = h_torus_normals[getIndex(i, 0, 4)];
    normal[getIndex(1, 0, 1)] = h_torus_normals[getIndex(i, 1, 4)];
    normal[getIndex(2, 0, 1)] = h_torus_normals[getIndex(i, 2, 4)];
    normal[getIndex(3, 0, 1)] = h_torus_normals[getIndex(i, 3, 4)];

    float temp[16];
    float combo[16];
    float newPoint[4]; 
    float newNormal[4];

    // Calculate combined rotation matrix
    matrix_mul(&rotmatZ[0], &rotmatX[0], 4, 4, 4, 4, &temp[0]);
    matrix_mul(&rotmatY[0], &temp[0], 4, 4, 4, 4, &combo[0]);

    // Rotate the point and its normal
    matrix_mul(&combo[0], &point[0], 4, 4, 4, 1, &newPoint[0]);
    matrix_mul(&combo[0], &normal[0], 4, 4, 4, 1, &newNormal[0]);

    h_torus_vertex[getIndex(i, 0, 4)] = newPoint[getIndex(0, 0, 1)];
    h_torus_vertex[getIndex(i, 1, 4)] = newPoint[getIndex(1, 0, 1)];
    h_torus_vertex[getIndex(i, 2, 4)] = newPoint[getIndex(2, 0, 1)];
    h_torus_vertex[getIndex(i, 3, 4)] = newPoint[getIndex(3, 0, 1)];

    h_torus_normals[getIndex(i, 0, 4)] = newNormal[getIndex(0, 0, 1)];
    h_torus_normals[getIndex(i, 1, 4)] = newNormal[getIndex(1, 0, 1)];
    h_torus_normals[getIndex(i, 2, 4)] = newNormal[getIndex(2, 0, 1)];
    h_torus_normals[getIndex(i, 3, 4)] = newNormal[getIndex(3, 0, 1)];
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

  // Draw every surface with its corresponding normal
  for (int i = 0; i < number_torus_points; i++) {
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
  // Get the elapsed time and calculate the Frames Per Second
  double currentTime = glutGet(GLUT_ELAPSED_TIME);
  nbFrames += 1.0;
  if ( currentTime - lastTime >= 1000 ){
    // prlongf and reset timer
    char buffer[32];
    nbFrames += (currentTime - lastTime) / 1000.0;
    snprintf(buffer, 32, "Naive Torus - FPS: %f", nbFrames);

    glutSetWindowTitle(buffer);
    nbFrames = 0;
    lastTime = currentTime;
  }

  // Perform the rotation
  rotateTorus();

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

/////////////////////////
//// TIMERS  ////////////
/////////////////////////
int getMilliCount(){
	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

int getMilliSpan(int nTimeStart){
	int nSpan = getMilliCount() - nTimeStart;
	if(nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

////////////
/// MAIN ///
////////////
int main(int argc, char** argv)
{
  int start;
  int span;

  // Read the initial ellipse points and normals
  h_ellipse_vertex = readFromFile("../ellipse_matrix.txt");
  h_ellipse_normals = readFromFile("../ellipse_normals.txt");


  start = getMilliCount(); // Timer

  // Sweep the ellipse to generate the torus.
  sweep();
 
  span = getMilliSpan(start); // Timer
  cout<<"Generate torus vertex table: "<<span<< " ms"<<endl;
  start = getMilliCount();

  // Generate the surface table for the torus
  generateSurfaceTable();

  span = getMilliSpan(start);
  cout<<"Generate torus surface table: "<<span<< " ms"<<endl;

  // Display the generated torus and start rotation.
  displayTorus(argc, argv);
  
  return 0;
}

