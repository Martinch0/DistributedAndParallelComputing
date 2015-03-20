/// COMMENT

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

using namespace std;

 #define PI 3.14159265

float** h_ellipse_vertex;
float** h_ellipse_normals;

float** h_torus_vertex;
float** h_torus_normals;
int** h_torus_surface;

float* sweep_origin; //TODO not sure if we are using this.

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
float** rotmat_X(float angle)
{
  float angle_rad = PI * angle / 180.0;

  float** rotation = new float*[4];
  rotation[0] = new float[4];
  rotation[0][0] = 1;
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
float** rotmat_Z(float angle)
{
  float angle_rad = PI * angle / 180.0;

  float** rotation = new float*[4];
  rotation[0] = new float[4];
  rotation[0][0] = cos(angle_rad);
  rotation[0][1] = -sin(angle_rad);
  rotation[0][2] = 0;
  rotation[0][3] = 0;
  rotation[1] = new float[4];
  rotation[1][0] = sin(angle_rad);
  rotation[1][1] = cos(angle_rad);
  rotation[1][2] = 0;
  rotation[1][3] = 0;
  rotation[2] = new float[4];
  rotation[2][0] = 0;
  rotation[2][1] = 0;
  rotation[2][2] = 1;
  rotation[2][3] = 0;
  rotation[3] = new float[4];
  rotation[3][0] = 0;
  rotation[3][1] = 0;
  rotation[3][2] = 0;
  rotation[3][3] = 1;
  return rotation;
}

/////////////////////////////////////////////
// Sweep function to generate torus points //
/////////////////////////////////////////////  
void sweep()
{

  float step = 360.0f / number_sweep_steps;
  float angle = 0;
  int curPosition = 0;
  float **rot;

  float **point = new float*[4];
  point[0] = new float[1];
  point[1] = new float[1];
  point[2] = new float[1];
  point[3] = new float[1];

  float **normal = new float*[4];
  normal[0] = new float[1];
  normal[1] = new float[1];
  normal[2] = new float[1];
  normal[3] = new float[1];

  number_torus_points = number_sweep_steps * number_ellipse_points;
  h_torus_vertex = new float*[number_torus_points];//torus points
  h_torus_normals = new float*[number_torus_points];//torus normals
  
  // for every sweep step
  for(int i = 0; i<number_sweep_steps; i++) {
    rot = rotmat_Y(angle);

    // for every ellipse point
    for(int j = 0; j<number_ellipse_points; j++) {
      point[0][0] = h_ellipse_vertex[j][0];
      point[1][0] = h_ellipse_vertex[j][1];
      point[2][0] = h_ellipse_vertex[j][2];
      point[3][0] = h_ellipse_vertex[j][3];

      normal[0][0] = h_ellipse_normals[j][0];
      normal[1][0] = h_ellipse_normals[j][1];
      normal[2][0] = h_ellipse_normals[j][2];
      normal[3][0] = h_ellipse_normals[j][3];

      // Rotate the point
      float **newPoint = matrix_mul(rot, point, 4, 4, 4, 1);
      float **newNormal = matrix_mul(rot, normal, 4, 4, 4, 1);

      h_torus_vertex[curPosition] = new float[4];
      h_torus_vertex[curPosition][0] = newPoint[0][0];
      h_torus_vertex[curPosition][1] = newPoint[1][0];
      h_torus_vertex[curPosition][2] = newPoint[2][0];
      h_torus_vertex[curPosition][3] = newPoint[3][0];

      h_torus_normals[curPosition] = new float[4];
      h_torus_normals[curPosition][0] = newNormal[0][0];
      h_torus_normals[curPosition][1] = newNormal[1][0];
      h_torus_normals[curPosition][2] = newNormal[2][0];
      h_torus_normals[curPosition][3] = newNormal[3][0];

      delete newPoint;
      delete newNormal;
      curPosition++;
      
    }
    angle += step;
    delete rot;
  }
  delete point[0];
  delete point[1];
  delete point[2];
  delete point[3];
  delete point;

  delete normal[0];
  delete normal[1];
  delete normal[2];
  delete normal[3];
  delete normal;
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

void rotateTorus() {
  float** rotmatX = rotmat_X(5);
  float** rotmatZ = rotmat_Z(10);
  float** point = new float*[4];
  point[0] = new float[1];
  point[1] = new float[1];
  point[2] = new float[1];
  point[3] = new float[1];

  float **normal = new float*[4];
  normal[0] = new float[1];
  normal[1] = new float[1];
  normal[2] = new float[1];
  normal[3] = new float[1];

  for (int i = 0; i < number_torus_points; i++) {
    point[0][0] = h_torus_vertex[i][0];
    point[1][0] = h_torus_vertex[i][1];
    point[2][0] = h_torus_vertex[i][2];
    point[3][0] = h_torus_vertex[i][3];

    normal[0][0] = h_torus_normals[i][0];
    normal[1][0] = h_torus_normals[i][1];
    normal[2][0] = h_torus_normals[i][2];
    normal[3][0] = h_torus_normals[i][3];

    float** combo = matrix_mul(rotmatZ, rotmatX, 4, 4, 4, 4);
    float** newPoint = matrix_mul(combo, point, 4, 4, 4, 1);
    float **newNormal = matrix_mul(combo, normal, 4, 4, 4, 1);

    h_torus_vertex[i][0] = newPoint[0][0];
    h_torus_vertex[i][1] = newPoint[1][0];
    h_torus_vertex[i][2] = newPoint[2][0];
    h_torus_vertex[i][3] = newPoint[3][0];

    h_torus_normals[i][0] = newNormal[0][0];
    h_torus_normals[i][1] = newNormal[1][0];
    h_torus_normals[i][2] = newNormal[2][0];
    h_torus_normals[i][3] = newNormal[3][0];

    delete[] combo[0];
    delete[] combo[1];
    delete[] combo[2];
    delete[] combo[3];
    delete[] combo;

    delete[] newPoint[0];
    delete[] newPoint[1];
    delete[] newPoint[2];
    delete[] newPoint[3];
    delete[] newPoint;

    delete[] newNormal[0];
    delete[] newNormal[1];
    delete[] newNormal[2];
    delete[] newNormal[3];
    delete[] newNormal;
  }

  delete[] rotmatX[0];
  delete[] rotmatX[1];
  delete[] rotmatX[2];
  delete[] rotmatX[3];
  delete[] rotmatX;

  delete[] rotmatZ[0];
  delete[] rotmatZ[1];
  delete[] rotmatZ[2];
  delete[] rotmatZ[3];
  delete[] rotmatZ;

  delete[] point[0];
  delete[] point[1];
  delete[] point[2];
  delete[] point[3];
  delete[] point;

  delete normal[0];
  delete normal[1];
  delete normal[2];
  delete normal[3];
  delete normal;
}

/////////////////////////
/////// GL CODE /////////
/////////////////////////
GLfloat light_diffuse[] = {1.0, 0.0, 0.0, 1.0};  /* Red diffuse light. */
GLfloat light_position[] = {500.0, 500.0, 1.0, 0.0};  /* Infinite light location. */
GLfloat light_ambient[] = { 0.5, 0.0, 0.0, 1.0 };

void drawBox()
{

  GLfloat * normal = new GLfloat[4];

  for (int i = 0; i < number_torus_points; i++) {
    glBegin(GL_QUADS);

    normal[0] = h_torus_normals[i][0];
    normal[1] = h_torus_normals[i][1];
    normal[2] = h_torus_normals[i][2];

    glNormal3fv(&normal[0]);
    glVertex3fv(&h_torus_vertex[h_torus_surface[i][0]-1][0]);
    glVertex3fv(&h_torus_vertex[h_torus_surface[i][1]-1][0]);
    glVertex3fv(&h_torus_vertex[h_torus_surface[i][2]-1][0]);
    glVertex3fv(&h_torus_vertex[h_torus_surface[i][3]-1][0]);
    glEnd();
  }
}

void display()
{
  rotateTorus();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
  glutPostRedisplay();
}

void init()
{
  /* Enable a single OpenGL light. */
  glEnable(GL_DEPTH_TEST);
  glShadeModel (GL_SMOOTH);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);


  /* Use depth buffering for hidden surface elimination. */
  glEnableClientState(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 100000.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 2000.0, 0.0,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 0.0, 1.0);      /* up is in positive Y direction */

}

void displayTorus(int argc, char **argv)
{
  glewInit();
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("Torus");
  glutDisplayFunc(display);
  init();
  glutMainLoop();
}

/////////////////////////
//// END GL CODE ////////
/////////////////////////


////////////
/// MAIN ///
////////////
int main(int argc, char** argv)
{
  //CUDA properties
  int devID;
  cudaDeviceProp props;
  
  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  h_ellipse_vertex = readFromFile("../ellipse_matrix.txt");
  h_ellipse_normals = readFromFile("../ellipse_normals.txt");

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));             // create a timer
  cutilCheckError(cutStartTimer(timer));               // start the timer

  sweep();

  h_torus_surface = generateSurfaceTable();


  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0);

  displayTorus(argc, argv);

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

