/// COMMENT

#include "cutil_inline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include "custom_matrix.cpp"

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

using namespace std;

#define PI 3.14159265

CustomMatrix<float>* h_ellipse_vertex;
CustomMatrix<float>* h_ellipse_normals;

CustomMatrix<float>* h_torus_vertex;
CustomMatrix<float>* h_torus_normals;
CustomMatrix<int>* h_torus_surface;

int number_sweep_steps = 1000;
int number_ellipse_points;
int number_torus_points;
int torus_rotation[] = {1, 0, 2};

int nbFrames = 0;
double lastTime = 0;


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
CustomMatrix<float>* readFromFile(char *filename)
{
  ifstream fin(filename);
  int len = countLines(filename);
  number_ellipse_points = len;
  CustomMatrix<float>* arr = new CustomMatrix<float>(len, 4);
  for(int i=0; i<len; i++) {
    fin>>arr->matrix[i][0]>>arr->matrix[i][1]>>arr->matrix[i][2];
    arr->matrix[i][3] = 1.0f;
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
void matrix_mul(CustomMatrix<float>* a, CustomMatrix<float>* b, int arows, int acols, int brows, int bcols, CustomMatrix<float>* out)
{
  //for every row in the result
  for(int i=0; i < acols; i++) {
    //for every column in the result
    for(int j=0; j < brows; j++) {
      float sum = 0;
      //find the sum of the multiplied row and column
      for(int k=0; k < acols; k++) {
        sum += a->matrix[i][k] * b->matrix[k][j];
      }
      out->matrix[i][j] = sum;
    }
  }
}

/////////////////////////////////////////
// Rotation Transformation Matrix on X //
/////////////////////////////////////////
CustomMatrix<float>* rotmat_X(float angle)
{
  float angle_rad = PI * angle / 180.0;

  CustomMatrix<float>* rotation = new CustomMatrix<float>(4,4);

  //First row.
  rotation->matrix[0][0] = 1;
  rotation->matrix[0][1] = 0;
  rotation->matrix[0][2] = 0;
  rotation->matrix[0][3] = 0;

  //Second row.
  rotation->matrix[1][0] = 0;
  rotation->matrix[1][1] = cos(angle_rad);
  rotation->matrix[1][2] = -sin(angle_rad);
  rotation->matrix[1][3] = 0;

  //Third row.
  rotation->matrix[2][0] = 0;
  rotation->matrix[2][1] = sin(angle_rad);
  rotation->matrix[2][2] = cos(angle_rad);
  rotation->matrix[2][3] = 0;

  //Fourth row.
  rotation->matrix[3][0] = 0;
  rotation->matrix[3][1] = 0;
  rotation->matrix[3][2] = 0;
  rotation->matrix[3][3] = 1;
  return rotation;
}

/////////////////////////////////////////
// Rotation Transformation Matrix on Y //
/////////////////////////////////////////
CustomMatrix<float>* rotmat_Y(float angle)
{
  float angle_rad = PI * angle / 180.0;

  CustomMatrix<float>* rotation = new CustomMatrix<float>(4,4);

  //First row.
  rotation->matrix[0][0] = cos(angle_rad);
  rotation->matrix[0][1] = 0;
  rotation->matrix[0][2] = sin(angle_rad);
  rotation->matrix[0][3] = 0;

  //Second row.
  rotation->matrix[1][0] = 0;
  rotation->matrix[1][1] = 1;
  rotation->matrix[1][2] = 0;
  rotation->matrix[1][3] = 0;

  //Third row.
  rotation->matrix[2][0] = -sin(angle_rad);
  rotation->matrix[2][1] = 0;
  rotation->matrix[2][2] = cos(angle_rad);
  rotation->matrix[2][3] = 0;

  //Fourth row.
  rotation->matrix[3][0] = 0;
  rotation->matrix[3][1] = 0;
  rotation->matrix[3][2] = 0;
  rotation->matrix[3][3] = 1;
  return rotation;
}

/////////////////////////////////////////
// Rotation Transformation Matrix on Z //
/////////////////////////////////////////
CustomMatrix<float>* rotmat_Z(float angle)
{
  float angle_rad = PI * angle / 180.0;

  CustomMatrix<float>* rotation = new CustomMatrix<float>(4,4);

  //First row.
  rotation->matrix[0][0] = cos(angle_rad);
  rotation->matrix[0][1] = -sin(angle_rad);
  rotation->matrix[0][2] = 0;
  rotation->matrix[0][3] = 0;

  //Second row.
  rotation->matrix[1][0] = sin(angle_rad);
  rotation->matrix[1][1] = cos(angle_rad);
  rotation->matrix[1][2] = 0;
  rotation->matrix[1][3] = 0;

  //Third row.
  rotation->matrix[2][0] = 0;
  rotation->matrix[2][1] = 0;
  rotation->matrix[2][2] = 1;
  rotation->matrix[2][3] = 0;

  //Fourth row.
  rotation->matrix[3][0] = 0;
  rotation->matrix[3][1] = 0;
  rotation->matrix[3][2] = 0;
  rotation->matrix[3][3] = 1;
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

  CustomMatrix<float>* rot;
  CustomMatrix<float> point(4,1);
  CustomMatrix<float> normal(4,1);

  number_torus_points = number_sweep_steps * number_ellipse_points;
  h_torus_vertex = new CustomMatrix<float>(number_torus_points, 4);//torus points
  h_torus_normals = new CustomMatrix<float>(number_torus_points, 4);//torus normals
  
  // for every sweep step
  for(int i = 0; i<number_sweep_steps; i++) {
    rot = rotmat_Y(angle);

    // for every ellipse point
    for(int j = 0; j<number_ellipse_points; j++) {
      point.matrix[0][0] = h_ellipse_vertex->matrix[j][0];
      point.matrix[1][0] = h_ellipse_vertex->matrix[j][1];
      point.matrix[2][0] = h_ellipse_vertex->matrix[j][2];
      point.matrix[3][0] = h_ellipse_vertex->matrix[j][3];

      normal.matrix[0][0] = h_ellipse_normals->matrix[j][0];
      normal.matrix[1][0] = h_ellipse_normals->matrix[j][1];
      normal.matrix[2][0] = h_ellipse_normals->matrix[j][2];
      normal.matrix[3][0] = h_ellipse_normals->matrix[j][3];


      CustomMatrix<float> newPoint(4,1);
      CustomMatrix<float> newNormal(4,1);

      // Rotate the point
      matrix_mul(rot, &point, 4, 4, 4, 1, &newPoint);
      matrix_mul(rot, &normal, 4, 4, 4, 1, &newNormal);

      h_torus_vertex->matrix[curPosition][0] = newPoint.matrix[0][0];
      h_torus_vertex->matrix[curPosition][1] = newPoint.matrix[1][0];
      h_torus_vertex->matrix[curPosition][2] = newPoint.matrix[2][0];
      h_torus_vertex->matrix[curPosition][3] = newPoint.matrix[3][0];

      h_torus_normals->matrix[curPosition][0] = newNormal.matrix[0][0];
      h_torus_normals->matrix[curPosition][1] = newNormal.matrix[1][0];
      h_torus_normals->matrix[curPosition][2] = newNormal.matrix[2][0];
      h_torus_normals->matrix[curPosition][3] = newNormal.matrix[3][0];

      curPosition++;
      
    }
    angle += step;
  }
  delete rot;
}

////////////////////////////////
// Generate the surface table //
////////////////////////////////
void  generateSurfaceTable()
{
  //Yu-Yang: assumming rings are one after the other.
  //assumming matrixes are: arr[ellipse_number][x y z 1]
 
  // we need a surface for every point in the torus
  h_torus_surface = new CustomMatrix<int>(number_torus_points, 4);

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
          h_torus_surface->matrix[torus_point-1][0] = torus_point;
          h_torus_surface->matrix[torus_point-1][1] = torus_point + number_ellipse_points;
          h_torus_surface->matrix[torus_point-1][2] = torus_point + 1;
          h_torus_surface->matrix[torus_point-1][3] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square        
          h_torus_surface->matrix[torus_point-1][0] = torus_point;
          h_torus_surface->matrix[torus_point-1][1] = torus_point + number_ellipse_points;
          h_torus_surface->matrix[torus_point-1][2] = torus_point + number_ellipse_points + 1;
          h_torus_surface->matrix[torus_point-1][3] = torus_point + 1;
        }

      //if last ring
      } else {

        //last point in a ring
        if (torus_point % number_ellipse_points == 0) {
          //create surface square joining the last 2 points of the torus with the first two
          h_torus_surface->matrix[torus_point-1][0] = torus_point;
          h_torus_surface->matrix[torus_point-1][1] = 1;
          h_torus_surface->matrix[torus_point-1][2] = 2;
          h_torus_surface->matrix[torus_point-1][3] = torus_point + 1 - number_ellipse_points;
        } else {
          //create surface square
          h_torus_surface->matrix[torus_point-1][0] = torus_point;
          h_torus_surface->matrix[torus_point-1][1] = (torus_point + number_ellipse_points) - number_torus_points;
          h_torus_surface->matrix[torus_point-1][2] = (torus_point + number_ellipse_points) - number_torus_points + 1;
          h_torus_surface->matrix[torus_point-1][3] = torus_point + 1;
        }
      } 
    }
  }
}


////////////////////
// Torus rotation //
////////////////////
void rotateTorus() {
  CustomMatrix<float>* rotmatX = rotmat_X(torus_rotation[0]);
  CustomMatrix<float>* rotmatZ = rotmat_Z(torus_rotation[2]);
  CustomMatrix<float> point(4,1);
  CustomMatrix<float> normal(4,1);

  for (int i = 0; i < number_torus_points; i++) {
    point.matrix[0][0] = h_torus_vertex->matrix[i][0];
    point.matrix[1][0] = h_torus_vertex->matrix[i][1];
    point.matrix[2][0] = h_torus_vertex->matrix[i][2];
    point.matrix[3][0] = h_torus_vertex->matrix[i][3];

    normal.matrix[0][0] = h_torus_normals->matrix[i][0];
    normal.matrix[1][0] = h_torus_normals->matrix[i][1];
    normal.matrix[2][0] = h_torus_normals->matrix[i][2];
    normal.matrix[3][0] = h_torus_normals->matrix[i][3];

    CustomMatrix<float> combo(4,4); 
    CustomMatrix<float> newPoint(4,1); 
    CustomMatrix<float> newNormal(4,1);

    matrix_mul(rotmatZ, rotmatX, 4, 4, 4, 4, &combo);
    matrix_mul(&combo, &point, 4, 4, 4, 1, &newPoint);
    matrix_mul(&combo, &normal, 4, 4, 4, 1, &newNormal);

    h_torus_vertex->matrix[i][0] = newPoint.matrix[0][0];
    h_torus_vertex->matrix[i][1] = newPoint.matrix[1][0];
    h_torus_vertex->matrix[i][2] = newPoint.matrix[2][0];
    h_torus_vertex->matrix[i][3] = newPoint.matrix[3][0];

    h_torus_normals->matrix[i][0] = newNormal.matrix[0][0];
    h_torus_normals->matrix[i][1] = newNormal.matrix[1][0];
    h_torus_normals->matrix[i][2] = newNormal.matrix[2][0];
    h_torus_normals->matrix[i][3] = newNormal.matrix[3][0];
  }

  delete rotmatX;
  delete rotmatZ;
}

/////////////////////////
/////// GL CODE /////////
/////////////////////////
GLfloat light_diffuse[] = {1.0, 0.0, 0.0, 0.1};  /* Red diffuse light. */
GLfloat light_position[] = {0.0, 500.0, 1.0, 0.0};  /* Infinite light location. */
GLfloat light_ambient[] = { 0.5, 0.0, 0.0, 1.0 };

void drawBox()
{

  GLfloat * normal = new GLfloat[4];

  for (int i = 0; i < number_torus_points; i++) {
    glBegin(GL_QUADS);

    normal[0] = h_torus_normals->matrix[i][0];
    normal[1] = h_torus_normals->matrix[i][1];
    normal[2] = h_torus_normals->matrix[i][2];

    glNormal3fv(&normal[0]);
    glVertex3fv(&h_torus_vertex->matrix[h_torus_surface->matrix[i][0]-1][0]);
    glVertex3fv(&h_torus_vertex->matrix[h_torus_surface->matrix[i][1]-1][0]);
    glVertex3fv(&h_torus_vertex->matrix[h_torus_surface->matrix[i][2]-1][0]);
    glVertex3fv(&h_torus_vertex->matrix[h_torus_surface->matrix[i][3]-1][0]);
    glEnd();
  }

  delete[] normal;
}

void display()
{
  double currentTime = glutGet(GLUT_ELAPSED_TIME);
  cout<<currentTime<<endl;
  nbFrames++;
  if ( currentTime - lastTime >= 1000 ){ // If last prinf() was more than 1 sec ago
    // printf and reset timer
    char buffer[16];
    snprintf(buffer, 16, "FPS: %d", nbFrames);

    glutSetWindowTitle(buffer);
    nbFrames = 0;
    lastTime = currentTime;
  }
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

  generateSurfaceTable();


  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0);

  displayTorus(argc, argv);

  //Log througput
  printf("Seconds: %.4f \n", dSeconds);

  //writeToFile("vertex_table.m", "vTable", h_torus_vertex->matrix, number_torus_points, 4);
  //writeToFile("surface_table.m", "faces", h_torus_surface->matrix, number_torus_points, 4);

  //
  // INIT DATA HERE
  //
  
  // print information
  cout << "Number of ellipse vertices : " << number_ellipse_points << endl;
  cout << "Number of rotational sweep steps : " << number_sweep_steps << endl;

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finish
  cudaThreadSynchronize();

  cutilCheckError(cutStopTimer(timer));

  // exit and clean up device status
  cudaThreadExit();

  return 0;
}

