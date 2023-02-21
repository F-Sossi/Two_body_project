//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
// 
// This program simulates the motion of two bodies in space. The bodies are
// represented by spheres and the motion is simulated by using the Euler
// method.
//
// Complile with cmake: 
//
// mkdir build
// cd build
// cmake ..
// make
// ./my_project
// 
// To compile: nvcc kernel.cu -o tbp.exe -lGL -lGLU -lglut
// To run: ./lab3
// Note: install glut and opengl libraries
// sudo apt-get install mesa-common-dev libglu1-mesa-dev freeglut3-dev
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "two_body.h"
#include "viewer.h"

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("OpenGL Test");
    glutDisplayFunc(display);
    init();
    glutMainLoop();
    return 0;
}